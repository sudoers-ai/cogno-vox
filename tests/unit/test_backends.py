"""Backend HTTP wiring, mocked with httpx.MockTransport (no network)."""

from __future__ import annotations

import httpx

import cogno_vox.synthesizer as syn_mod
import cogno_vox.transcriber as tr_mod
from cogno_vox import (
    GrokSynthesizer,
    OpenAICompatSynthesizer,
    OpenAICompatTranscriber,
)


def _patch_async_client(monkeypatch, module, handler):
    """Route every AsyncClient in ``module`` through a MockTransport handler."""
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        kwargs.pop("timeout", None)
        return real(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", factory)


# ── transcriber ───────────────────────────────────────────────────────────

async def test_openai_compat_transcriber_parses_text(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"text": "bom dia"})

    _patch_async_client(monkeypatch, tr_mod, handler)
    backend = OpenAICompatTranscriber(
        base_url="https://api.groq.com/openai/v1", model="whisper-large-v3-turbo",
        api_key="gsk_test",
    )

    text = await backend.transcribe(b"\x00\x01", "voice.ogg")

    assert text == "bom dia"
    assert captured["url"].endswith("/v1/audio/transcriptions")
    assert captured["auth"] == "Bearer gsk_test"
    assert backend.name == "groq:whisper-large-v3-turbo"


async def test_openai_compat_transcriber_local_name_no_key():
    backend = OpenAICompatTranscriber(base_url="http://localhost:8000", model="small")
    assert backend.name == "local:small"


async def test_transcriber_returns_empty_on_http_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _patch_async_client(monkeypatch, tr_mod, handler)
    backend = OpenAICompatTranscriber(base_url="http://x", model="m", api_key="k")

    assert await backend.transcribe(b"\x00\x01", "voice.ogg") == ""


async def test_transcriber_empty_audio_short_circuits():
    backend = OpenAICompatTranscriber(base_url="http://x", model="m")
    assert await backend.transcribe(b"", "voice.ogg") == ""


# ── synthesizer ─────────────────────────────────────────────────────────────

async def test_openai_compat_synthesizer_returns_bytes(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"OPUSBYTES")

    _patch_async_client(monkeypatch, syn_mod, handler)
    backend = OpenAICompatSynthesizer(
        base_url="http://localhost:8880/v1", model="kokoro",
        voice="af_alloy", response_format="opus",
    )

    audio = await backend.synthesize("olá")

    assert audio == b"OPUSBYTES"
    assert captured["url"].endswith("/audio/speech")
    assert captured["body"]["input"] == "olá"
    assert captured["body"]["voice"] == "af_alloy"
    assert backend.name == "local:kokoro"
    assert backend.fmt == "opus"


async def test_grok_synthesizer_uses_grok_schema(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"GROKAUDIO")

    _patch_async_client(monkeypatch, syn_mod, handler)
    backend = GrokSynthesizer(api_key="xai_test", voice="eve")

    audio = await backend.synthesize("texto")

    assert audio == b"GROKAUDIO"
    assert captured["url"].endswith("/tts")
    # Grok-specific field names, not the OpenAI schema.
    assert captured["body"]["text"] == "texto"
    assert captured["body"]["voice_id"] == "eve"
    assert captured["body"]["output_format"]["codec"] == "opus"


async def test_synthesizer_returns_empty_on_http_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    _patch_async_client(monkeypatch, syn_mod, handler)
    backend = OpenAICompatSynthesizer(base_url="http://x", model="tts-1", api_key="k")

    assert await backend.synthesize("hi") == b""


async def test_synthesizer_empty_text_short_circuits():
    backend = OpenAICompatSynthesizer(base_url="http://x", model="tts-1")
    assert await backend.synthesize("") == b""


async def test_opus_via_wav_requests_wav_and_reencodes(monkeypatch):
    # Kokoro-fastapi's opus encoder truncates the audio tail — the workaround requests WAV
    # and encodes Opus locally, still reporting fmt="opus" to the caller.
    from cogno_vox import synthesizer as synth_mod

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"WAVBYTES")

    _patch_async_client(monkeypatch, synth_mod, handler)
    monkeypatch.setattr(synth_mod, "wav_to_opus", lambda b: b"OGG:" + b)

    backend = OpenAICompatSynthesizer(
        base_url="http://localhost:8880/v1", model="kokoro",
        voice="pf_dora", response_format="opus", opus_via_wav=True,
    )
    audio = await backend.synthesize("olá, tudo bem?")

    assert captured["body"]["response_format"] == "wav"   # WAV requested from the server
    assert audio == b"OGG:WAVBYTES"                       # encoded locally
    assert backend.fmt == "opus"                          # caller still sees opus


async def test_opus_via_wav_inert_for_other_formats(monkeypatch):
    from cogno_vox import synthesizer as synth_mod

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"WAVBYTES")

    _patch_async_client(monkeypatch, synth_mod, handler)
    backend = OpenAICompatSynthesizer(
        base_url="http://localhost:8881/v1", model="dia-ptbr",
        response_format="wav", opus_via_wav=True,
    )
    audio = await backend.synthesize("olá")
    assert captured["body"]["response_format"] == "wav"
    assert audio == b"WAVBYTES"                           # untouched — not an opus request
