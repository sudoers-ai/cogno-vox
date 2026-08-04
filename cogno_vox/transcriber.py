"""
cogno_vox.transcriber — async speech-to-text backends + fallback chain.

Adapted clean-room from the parent ``cogno.audio.transcriber``, made async
(httpx.AsyncClient) and decoupled from the host's model_ladder/tenant-key
context. Three backend shapes:

  - ``OpenAICompatTranscriber`` — any ``POST /v1/audio/transcriptions`` server
    (faster-whisper-server local, Groq, OpenAI — distinguished by base_url/key).
  - ``GeminiTranscriber`` — multimodal ``generateContent`` (no Whisper path).
  - ``BedrockTranscriber`` — AWS Converse API with audio blocks (e.g. Voxtral);
    boto3 is lazy-imported (optional ``[bedrock]`` extra).

All backends return ``""`` on failure so the chain transparently fails over;
``FallbackTranscriber`` raises ``TranscriptionError`` only when every tier fails.
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

from cogno_vox.ports import TranscriberBackend, TranscriptionError
from cogno_vox.types import SUPPORTED_INPUT_FORMATS, TranscriptionResult

log = logging.getLogger(__name__)

_TRANSCRIBE_PROMPT = (
    "Transcribe this audio exactly as spoken. "
    "Return ONLY the transcribed text, nothing else. "
    "No commentary, no formatting, no labels."
)

_AUDIO_MIME_TYPES = {
    "ogg": "audio/ogg",
    "opus": "audio/ogg",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "webm": "audio/webm",
    "flac": "audio/flac",
}

_BEDROCK_AUDIO_FORMATS = {
    "ogg": "ogg",
    "opus": "ogg",
    "mp3": "mp3",
    "wav": "wav",
    "m4a": "mp4",
    "webm": "webm",
    "flac": "flac",
}


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else "ogg"


# ── OpenAI-compatible (faster-whisper-server / Groq / OpenAI) ─────────────

class OpenAICompatTranscriber:
    """Any OpenAI-compatible ``/v1/audio/transcriptions`` server.

    Covers the local faster-whisper-server (no api_key) and the Groq / OpenAI
    clouds (api_key set) — they differ only by ``base_url`` and credentials.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        if self._base_url.endswith("/v1"):
            self._base_url = self._base_url[:-3]
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    @property
    def name(self) -> str:
        if not self._api_key:
            return f"local:{self._model}"
        provider = "groq" if "groq" in self._base_url else "openai"
        return f"{provider}:{self._model}"

    async def transcribe(self, audio: bytes, filename: str = "audio.ogg") -> str:
        if not audio:
            return ""
        url = f"{self._base_url}/v1/audio/transcriptions"
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    url,
                    headers=headers,
                    files={"file": (filename, audio)},
                    data={"model": self._model},
                )
                resp.raise_for_status()
                return resp.json().get("text", "")
        except Exception as exc:
            log.warning("stage=STT event=tier_failed tier=%s error=%s", self.name, exc)
            return ""


# ── Gemini (multimodal generateContent — no Whisper protocol) ─────────────

class GeminiTranscriber:
    """Transcribes via Gemini multimodal ``generateContent`` (inline audio)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    @property
    def name(self) -> str:
        return f"gemini:{self._model}"

    async def transcribe(self, audio: bytes, filename: str = "audio.ogg") -> str:
        if not audio or not self._api_key:
            return ""
        mime = _AUDIO_MIME_TYPES.get(_ext(filename), "audio/ogg")
        body = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": mime,
                                     "data": base64.b64encode(audio).decode()}},
                    {"text": _TRANSCRIBE_PROMPT},
                ]
            }]
        }
        # The key goes in the x-goog-api-key HEADER, never the URL query string: httpx embeds the
        # full URL in its error message, and this except logs the failure — `?key=...` would write
        # the tenant's BYOK key into the logs on any routine 400/429/5xx (a bad audio note is enough).
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self._model}:generateContent"
        )
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    url, json=body,
                    headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key},
                )
                resp.raise_for_status()
                cands = resp.json().get("candidates", [])
                if cands:
                    parts = cands[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
                return ""
        except Exception as exc:
            # log the exception TYPE + status only — str(exc) may still carry a request URL.
            status = getattr(getattr(exc, "response", None), "status_code", "")
            log.warning("stage=STT event=tier_failed tier=%s error=%s status=%s",
                        self.name, type(exc).__name__, status)
            return ""


# ── Bedrock (Voxtral via Converse API audio blocks) ───────────────────────

class BedrockTranscriber:
    """Transcribes via AWS Bedrock Converse API (audio content blocks).

    boto3 is lazy-imported (optional ``[bedrock]`` extra). Credentials come
    from the standard AWS env vars / chain, not from per-call config.
    """

    def __init__(
        self,
        model: str = "mistral.voxtral-mini-3b-2507",
        region: str = "us-east-1",
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._region = region
        self._timeout = timeout
        self._client = None

    @property
    def name(self) -> str:
        return f"bedrock:{self._model}"

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "boto3 not installed. Run: pip install 'cogno-vox[bedrock]'"
            ) from exc
        self._client = boto3.client(
            "bedrock-runtime",
            config=Config(
                region_name=self._region,
                read_timeout=self._timeout,
                connect_timeout=15,
                retries={"max_attempts": 0},
            ),
        )
        return self._client

    async def transcribe(self, audio: bytes, filename: str = "audio.ogg") -> str:
        if not audio:
            return ""
        import asyncio

        fmt = _BEDROCK_AUDIO_FORMATS.get(_ext(filename), "ogg")

        def _call() -> str:
            client = self._get_client()
            resp = client.converse(
                modelId=self._model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"audio": {"format": fmt, "source": {"bytes": audio}}},
                        {"text": _TRANSCRIBE_PROMPT},
                    ],
                }],
            )
            out = resp.get("output", {}).get("message", {})
            for block in out.get("content", []):
                if "text" in block:
                    return block["text"].strip()
            return ""

        try:
            # boto3 is sync; keep the event loop free under concurrency.
            return await asyncio.to_thread(_call)
        except Exception as exc:
            log.warning("stage=STT event=tier_failed tier=%s error=%s", self.name, exc)
            return ""


# ── Fallback chain ────────────────────────────────────────────────────────

class FallbackTranscriber:
    """Tries each tier in order; first non-empty transcription wins.

    The failover loop runs over :func:`cogno_homeo.resilient_call`, so a host can
    opt into a circuit breaker / retry / metrics seam by passing them in; with
    none supplied it behaves like the historical "first non-empty tier wins"
    chain. An empty transcription counts as a failure (failover + breaker trip).
    """

    def __init__(
        self,
        backends: list[TranscriberBackend],
        *,
        breaker: Optional[CircuitBreaker] = None,
        policy: Optional[RetryPolicy] = None,
        metrics: Optional[MetricsSink] = None,
    ) -> None:
        if not backends:
            raise TranscriptionError(
                "No transcriber tiers configured. Provide at least one tier "
                "(local server URL or a cloud API key)."
            )
        self.backends = backends
        self._breaker = breaker
        self._policy = policy
        self._metrics = metrics

    @property
    def name(self) -> str:
        return self.backends[0].name

    async def transcribe(
        self, audio: bytes, filename: str = "audio.ogg"
    ) -> TranscriptionResult:
        if not audio:
            raise TranscriptionError("Empty audio bytes.")
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext and ext not in SUPPORTED_INPUT_FORMATS:
            raise TranscriptionError(
                f"Unsupported format '{ext}'. Use: "
                f"{', '.join(sorted(SUPPORTED_INPUT_FORMATS))}"
            )

        async def attempt(backend: TranscriberBackend) -> TranscriptionResult:
            t0 = time.perf_counter()
            text = await backend.transcribe(audio, filename)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if text:
                log.info("stage=STT tier=%s ms=%.0f chars=%d",
                         backend.name, elapsed_ms, len(text))
            else:
                log.warning("stage=STT event=tier_empty tier=%s reason=failover", backend.name)
            return TranscriptionResult(text=text, tier=backend.name, elapsed_ms=elapsed_ms)

        try:
            return await resilient_call(
                self.backends, attempt,
                key=lambda b: b.name, is_success=lambda r: bool(r.text),
                breaker=self._breaker, policy=self._policy, metrics=self._metrics,
            )
        except NoCandidateAvailable:
            raise TranscriptionError("All transcriber tiers failed.")
