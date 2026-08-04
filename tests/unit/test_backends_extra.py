"""Coverage for the Gemini / ElevenLabs / Bedrock backends + edge branches."""

from __future__ import annotations

import sys
import types

import httpx

import cogno_vox.synthesizer as syn_mod
import cogno_vox.transcriber as tr_mod
from cogno_vox import (
    BedrockTranscriber,
    ElevenLabsSynthesizer,
    GeminiSynthesizer,
    GeminiTranscriber,
)
from cogno_vox.chunker import split_text_for_tts


def _patch(monkeypatch, module, handler):
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        kwargs.pop("timeout", None)
        return real(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", factory)


# ── Gemini transcriber ─────────────────────────────────────────────────────

async def test_gemini_transcriber_extracts_text(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": " olá  "}]}}]
        })

    _patch(monkeypatch, tr_mod, handler)
    backend = GeminiTranscriber(api_key="g", model="gemini-2.5-flash")
    assert await backend.transcribe(b"\x00\x01", "v.ogg") == "olá"
    assert backend.name == "gemini:gemini-2.5-flash"


async def test_gemini_transcriber_no_key_short_circuits():
    backend = GeminiTranscriber(api_key="")
    assert await backend.transcribe(b"\x00\x01", "v.ogg") == ""


async def test_gemini_transcriber_key_is_a_header_never_in_url(monkeypatch):
    # SECURITY (audit 2026-08-04): the BYOK key must ride in the x-goog-api-key header, NOT the
    # URL query string — httpx embeds the url in its error, so `?key=...` leaked the key into the
    # tier_failed warning log on any routine 4xx/5xx.
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["key_header"] = request.headers.get("x-goog-api-key")
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    _patch(monkeypatch, tr_mod, handler)
    await GeminiTranscriber(api_key="super-secret-key").transcribe(b"\x00", "v.ogg")
    assert "super-secret-key" not in seen["url"] and "key=" not in seen["url"]
    assert seen["key_header"] == "super-secret-key"


async def test_gemini_synth_key_is_a_header_never_in_url(monkeypatch):
    import base64
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["key_header"] = request.headers.get("x-goog-api-key")
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"inlineData": {"data": base64.b64encode(b"P").decode()}}]}}]})

    _patch(monkeypatch, syn_mod, handler)
    monkeypatch.setattr(syn_mod, "pcm_to_opus", lambda pcm, sample_rate=24000: b"OGG")
    await GeminiSynthesizer(api_key="super-secret-key").synthesize("hi")
    assert "super-secret-key" not in seen["url"] and "key=" not in seen["url"]
    assert seen["key_header"] == "super-secret-key"


async def test_gemini_transcriber_empty_candidates(monkeypatch):
    _patch(monkeypatch, tr_mod, lambda r: httpx.Response(200, json={"candidates": []}))
    assert await GeminiTranscriber(api_key="g").transcribe(b"\x00", "v.ogg") == ""


# ── Gemini synthesizer (PCM → opus mocked) ─────────────────────────────────

async def test_gemini_synthesizer_returns_converted_audio(monkeypatch):
    import base64

    pcm_b64 = base64.b64encode(b"RAWPCM").decode()

    def handler(request):
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"inlineData": {"data": pcm_b64}}]}}]
        })

    _patch(monkeypatch, syn_mod, handler)
    monkeypatch.setattr(syn_mod, "pcm_to_opus", lambda pcm, sample_rate=24000: b"OGG:" + pcm)

    backend = GeminiSynthesizer(api_key="g")
    audio = await backend.synthesize("oi")
    assert audio == b"OGG:RAWPCM"
    assert backend.fmt == "opus"
    assert backend.name.startswith("gemini:")


async def test_gemini_synthesizer_no_audio_data(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"inlineData": {}}]}}]
        })

    _patch(monkeypatch, syn_mod, handler)
    assert await GeminiSynthesizer(api_key="g").synthesize("oi") == b""


# ── ElevenLabs synthesizer ─────────────────────────────────────────────────

async def test_elevenlabs_maps_format_and_headers(monkeypatch):
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["key"] = request.headers.get("xi-api-key")
        return httpx.Response(200, content=b"ELAUDIO")

    _patch(monkeypatch, syn_mod, handler)
    backend = ElevenLabsSynthesizer(api_key="el_test", voice="VOICEID",
                                    response_format="opus")
    audio = await backend.synthesize("texto")
    assert audio == b"ELAUDIO"
    assert "/text-to-speech/VOICEID" in captured["url"]
    assert "output_format=opus_16000" in captured["url"]
    assert captured["key"] == "el_test"
    assert backend.fmt == "opus"


# ── Bedrock transcriber (boto3 mocked) ─────────────────────────────────────

async def test_bedrock_transcriber_uses_converse(monkeypatch):
    captured = {}

    class FakeClient:
        def converse(self, **kwargs):
            captured["model"] = kwargs["modelId"]
            return {"output": {"message": {"content": [{"text": " transcrito "}]}}}

    fake_boto3 = types.SimpleNamespace(client=lambda *a, **k: FakeClient())
    fake_botocore = types.ModuleType("botocore")
    fake_config_mod = types.ModuleType("botocore.config")
    fake_config_mod.Config = lambda **k: None
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_config_mod)

    backend = BedrockTranscriber(model="mistral.voxtral-mini-3b-2507")
    text = await backend.transcribe(b"\x00\x01", "v.ogg")
    assert text == "transcrito"
    assert captured["model"] == "mistral.voxtral-mini-3b-2507"
    assert backend.name == "bedrock:mistral.voxtral-mini-3b-2507"


async def test_bedrock_transcriber_empty_audio():
    assert await BedrockTranscriber().transcribe(b"", "v.ogg") == ""


# ── chunker: single over-long sentence hard-split ──────────────────────────

def test_chunker_hard_splits_long_sentence():
    # No sentence boundary, 30 words, max 10 -> 3 hard-split segments.
    text = " ".join(f"w{i}" for i in range(30))
    segs = split_text_for_tts(text, max_words=10)
    assert len(segs) == 3
    assert all(len(s.split()) <= 10 for s in segs)
