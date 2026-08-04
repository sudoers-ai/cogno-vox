"""
cogno_vox.synthesizer — async text-to-speech backends + fallback chain.

Adapted clean-room from the parent ``cogno.audio.synthesizer``, made async and
decoupled from the host model_ladder/tenant-key context. Backend shapes:

  - ``OpenAICompatSynthesizer`` — any ``POST /v1/audio/speech`` server
    (Kokoro local, OpenAI).
  - ``GrokSynthesizer`` — xAI ``/v1/tts`` (text/voice_id/output_format object).
  - ``ElevenLabsSynthesizer`` — ``/v1/text-to-speech/{voice}`` (xi-api-key).
  - ``GeminiSynthesizer`` — ``generateContent`` AUDIO modality (raw PCM →
    Opus via the ffmpeg helper in ``audio_utils``).

All return ``b""`` on failure so the chain fails over; ``FallbackSynthesizer``
raises ``SynthesisError`` only when every tier fails.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Optional

import httpx

from cogno_homeo import (
    CircuitBreaker,
    MetricsSink,
    NoCandidateAvailable,
    RetryPolicy,
    resilient_call,
)

from cogno_vox.audio_utils import pcm_to_opus, wav_to_opus
from cogno_vox.ports import SynthesisError, SynthesizerBackend
from cogno_vox.text_prep import apply_emotion, clean_text_for_tts, strip_emotion_tags
from cogno_vox.types import SynthesisResult

log = logging.getLogger(__name__)


# ── OpenAI-compatible (Kokoro local / OpenAI) ─────────────────────────────

class OpenAICompatSynthesizer:
    """Any OpenAI-compatible ``/v1/audio/speech`` server (Kokoro, OpenAI)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        voice: str = "alloy",
        response_format: str = "opus",
        timeout: float = 45.0,
        opus_via_wav: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._voice = voice
        self._fmt = response_format
        self._timeout = timeout
        # Kokoro-fastapi's own opus encoder truncates the TAIL of the audio (~1.5s lost — a
        # voice note ends mid-sentence). When set and opus is wanted, request WAV from the
        # server and encode Opus-in-Ogg here (proper flush). The factory sets it for the
        # local tier only; cloud providers encode opus correctly themselves.
        self._opus_via_wav = opus_via_wav and response_format == "opus"

    @property
    def fmt(self) -> str:
        return self._fmt

    @property
    def name(self) -> str:
        local = any(s in self._base_url for s in ("localhost", "host.docker", "kokoro"))
        return f"{'local' if local else 'openai'}:{self._model}"

    async def synthesize(self, text: str) -> bytes:
        if not text:
            return b""
        url = f"{self._base_url}/audio/speech"
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "input": text,
            "voice": self._voice,
            "response_format": "wav" if self._opus_via_wav else self._fmt,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                if self._opus_via_wav:
                    # off-loop: ffmpeg is a blocking subprocess (up to 30s) — run in a
                    # thread so concurrent turns/health checks never stall on an encode.
                    # b"" on encoder failure → the chain fails this tier over (never WAV
                    # mislabeled as the opus the caller promised the channel).
                    import asyncio
                    return await asyncio.to_thread(wav_to_opus, resp.content)
                return resp.content
        except Exception as exc:
            log.warning("stage=TTS event=tier_failed tier=%s error=%s", self.name, exc)
            return b""


# ── Grok (xAI /v1/tts — NOT OpenAI-compatible) ────────────────────────────

class GrokSynthesizer:
    """xAI Grok TTS (``text``/``voice_id``/``output_format`` object).

    Voices: eve, ara, leo, rex, sal.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        model: str = "grok-2-tts",
        voice: str = "eve",
        response_format: str = "opus",
        sample_rate: int = 24000,
        timeout: float = 45.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._fmt = response_format
        self._sample_rate = sample_rate
        self._timeout = timeout

    @property
    def fmt(self) -> str:
        return self._fmt

    @property
    def name(self) -> str:
        return f"grok:{self._model}"

    async def synthesize(self, text: str) -> bytes:
        if not text:
            return b""
        url = f"{self._base_url}/tts"
        payload = {
            "text": text,
            "voice_id": self._voice,
            "model": self._model,
            "output_format": {
                "codec": self._fmt,
                "sample_rate": self._sample_rate,
                "bit_rate": 128000,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    url, headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
                resp.raise_for_status()
                return resp.content
        except Exception as exc:
            log.warning("stage=TTS event=tier_failed tier=%s error=%s", self.name, exc)
            return b""


# ── ElevenLabs ────────────────────────────────────────────────────────────

class ElevenLabsSynthesizer:
    """ElevenLabs TTS (``/v1/text-to-speech/{voice}``, ``xi-api-key`` header)."""

    _FORMAT_MAP = {
        "opus": "opus_16000",
        "mp3": "mp3_44100_128",
        "pcm": "pcm_16000",
        "ulaw": "ulaw_8000",
    }

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.elevenlabs.io/v1",
        model: str = "eleven_multilingual_v2",
        voice: str = "JBFqnCBsd6RMkjVDRZzb",
        response_format: str = "opus",
        timeout: float = 45.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._fmt = response_format
        self._el_fmt = self._FORMAT_MAP.get(response_format, response_format)
        self._timeout = timeout

    @property
    def fmt(self) -> str:
        return self._fmt

    @property
    def name(self) -> str:
        return f"elevenlabs:{self._model}"

    async def synthesize(self, text: str) -> bytes:
        if not text:
            return b""
        url = f"{self._base_url}/text-to-speech/{self._voice}"
        headers = {"xi-api-key": self._api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "model_id": self._model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    url, headers=headers, json=payload,
                    params={"output_format": self._el_fmt},
                )
                resp.raise_for_status()
                return resp.content
        except Exception as exc:
            log.warning("stage=TTS event=tier_failed tier=%s error=%s", self.name, exc)
            return b""


# ── Gemini (generateContent AUDIO modality → PCM → Opus) ──────────────────

class GeminiSynthesizer:
    """Gemini TTS — returns raw Linear16 PCM @ 24kHz, converted to Opus.

    Voices: Kore, Charon, Fenrir, Puck, Aoede, Leda, Orus, Zephyr.
    """

    _API_BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-preview-tts",
        voice: str = "Kore",
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._timeout = timeout

    @property
    def fmt(self) -> str:
        return "opus"

    @property
    def name(self) -> str:
        return f"gemini:{self._model}"

    async def synthesize(self, text: str) -> bytes:
        if not text:
            return b""
        url = f"{self._API_BASE}/models/{self._model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": self._voice}
                    }
                },
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                # key in the x-goog-api-key HEADER, not params: httpx merges params into request.url
                # and embeds the url in its error message, so `key=...` would leak the tenant's key
                # into the warning log below on any routine 400/429/5xx.
                resp = await client.post(
                    url, json=payload,
                    headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key},
                )
                resp.raise_for_status()
                cands = resp.json().get("candidates", [])
                if not cands:
                    return b""
                parts = cands[0].get("content", {}).get("parts", [])
                if not parts:
                    return b""
                audio_b64 = parts[0].get("inlineData", {}).get("data", "")
                if not audio_b64:
                    return b""
                return pcm_to_opus(base64.b64decode(audio_b64), sample_rate=24000)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", "")
            log.warning("stage=TTS event=tier_failed tier=%s error=%s status=%s",
                        self.name, type(exc).__name__, status)
            return b""


# ── Fallback chain ────────────────────────────────────────────────────────

class FallbackSynthesizer:
    """Tries each tier in order; first non-empty audio wins.

    The failover loop runs over :func:`cogno_homeo.resilient_call`, so a host can
    opt into a circuit breaker / retry / metrics seam; with none supplied it
    behaves like the historical "first non-empty tier wins" chain. Empty audio
    counts as a failure (failover + breaker trip).
    """

    def __init__(
        self,
        backends: list[SynthesizerBackend],
        *,
        breaker: Optional[CircuitBreaker] = None,
        policy: Optional[RetryPolicy] = None,
        metrics: Optional[MetricsSink] = None,
    ) -> None:
        if not backends:
            raise SynthesisError("No synthesizer tiers configured.")
        self.backends = backends
        self._breaker = breaker
        self._policy = policy
        self._metrics = metrics

    @property
    def name(self) -> str:
        return self.backends[0].name

    async def synthesize(self, text: str, *, emotion: str = "") -> SynthesisResult:
        # Strip emoji / markdown decoration ONCE here so every tier speaks the words, not the
        # symbols (a reply is written for the eye — 😊, **bold**, a raw link URL). ``chars`` then
        # reflects what is actually spoken (also the honest number to meter).
        #
        # ``emotion`` is an engine-agnostic hint ("chuckle") — decorated PER TIER below in that
        # tier's dialect, so a failover to a plain engine never carries a tag it would read out
        # loud. Any tag already inline in the reply is a leak (the pipeline hands emotion as a
        # hint, not text) and is stripped up front for the same reason.
        text = strip_emotion_tags(clean_text_for_tts(text))
        if not text:
            raise SynthesisError("Empty text.")

        async def attempt(backend: SynthesizerBackend) -> SynthesisResult:
            dialect = getattr(backend, "emotion_dialect", "")
            spoken = apply_emotion(text, emotion, dialect) if emotion and dialect else text
            t0 = time.perf_counter()
            audio = await backend.synthesize(spoken)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if audio:
                log.info("stage=TTS tier=%s ms=%.0f bytes=%d",
                         backend.name, elapsed_ms, len(audio))
            else:
                log.warning("stage=TTS event=tier_empty tier=%s reason=failover", backend.name)
            fmt = getattr(backend, "fmt", "opus")
            return SynthesisResult(audio=audio, fmt=fmt, tier=backend.name,
                                   elapsed_ms=elapsed_ms, chars=len(spoken))

        try:
            return await resilient_call(
                self.backends, attempt,
                key=lambda b: b.name, is_success=lambda r: bool(r.audio),
                breaker=self._breaker, policy=self._policy, metrics=self._metrics,
            )
        except NoCandidateAvailable:
            raise SynthesisError("All synthesizer tiers failed.")
