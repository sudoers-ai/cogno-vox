"""Fallback-chain behaviour: failover, result telemetry, empty/format guards."""

from __future__ import annotations

import pytest

from cogno_vox import (
    FallbackSynthesizer,
    FallbackTranscriber,
    SynthesisError,
    TranscriptionError,
)
from tests.conftest import StubSynthesizer, StubTranscriber


# ── transcriber ───────────────────────────────────────────────────────────

async def test_transcriber_first_tier_wins():
    t1 = StubTranscriber("local:small", "hello world")
    t2 = StubTranscriber("openai:whisper-1", "should not run")
    chain = FallbackTranscriber([t1, t2])

    result = await chain.transcribe(b"\x00\x01", "voice.ogg")

    assert result.text == "hello world"
    assert result.tier == "local:small"
    assert result.chars == len("hello world")
    assert result.elapsed_ms >= 0
    assert t2.calls == 0  # second tier never touched


async def test_transcriber_fails_over_on_empty():
    t1 = StubTranscriber("local:small", "")        # fails → ""
    t2 = StubTranscriber("groq:turbo", "recovered")
    chain = FallbackTranscriber([t1, t2])

    result = await chain.transcribe(b"\x00\x01", "voice.ogg")

    assert result.text == "recovered"
    assert result.tier == "groq:turbo"
    assert t1.calls == 1 and t2.calls == 1


async def test_transcriber_all_fail_raises():
    chain = FallbackTranscriber([StubTranscriber("a", ""), StubTranscriber("b", "")])
    with pytest.raises(TranscriptionError):
        await chain.transcribe(b"\x00\x01", "voice.ogg")


async def test_transcriber_empty_audio_raises():
    chain = FallbackTranscriber([StubTranscriber("a", "x")])
    with pytest.raises(TranscriptionError):
        await chain.transcribe(b"", "voice.ogg")


async def test_transcriber_unsupported_format_raises():
    chain = FallbackTranscriber([StubTranscriber("a", "x")])
    with pytest.raises(TranscriptionError):
        await chain.transcribe(b"\x00\x01", "clip.aiff")


def test_transcriber_requires_tiers():
    with pytest.raises(TranscriptionError):
        FallbackTranscriber([])


# ── synthesizer ─────────────────────────────────────────────────────────────

async def test_synthesizer_first_tier_wins():
    s1 = StubSynthesizer("local:kokoro", b"OPUSDATA")
    s2 = StubSynthesizer("openai:tts-1", b"NOPE")
    chain = FallbackSynthesizer([s1, s2])

    result = await chain.synthesize("olá mundo")

    assert result.audio == b"OPUSDATA"
    assert result.tier == "local:kokoro"
    assert result.fmt == "opus"
    assert result.nbytes == len(b"OPUSDATA")
    assert result.chars == len("olá mundo")   # TTS billable unit (input text length)
    assert s2.calls == 0


async def test_synthesizer_fails_over_on_empty():
    s1 = StubSynthesizer("local:kokoro", b"")
    s2 = StubSynthesizer("grok:grok-2-tts", b"AUDIO")
    chain = FallbackSynthesizer([s1, s2])

    result = await chain.synthesize("texto")

    assert result.audio == b"AUDIO"
    assert result.tier == "grok:grok-2-tts"


async def test_synthesizer_all_fail_raises():
    chain = FallbackSynthesizer([StubSynthesizer("a", b""), StubSynthesizer("b", b"")])
    with pytest.raises(SynthesisError):
        await chain.synthesize("texto")


async def test_synthesizer_empty_text_raises():
    chain = FallbackSynthesizer([StubSynthesizer("a", b"x")])
    with pytest.raises(SynthesisError):
        await chain.synthesize("")


def test_synthesizer_requires_tiers():
    with pytest.raises(SynthesisError):
        FallbackSynthesizer([])
