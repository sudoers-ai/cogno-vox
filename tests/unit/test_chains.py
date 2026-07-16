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


# ── emotion threading (per-tier dialect) ─────────────────────────────────

class RecordingSynthesizer(StubSynthesizer):
    """Stub that records the exact text it was asked to speak."""

    def __init__(self, name: str, audio: bytes, dialect: str = "") -> None:
        super().__init__(name, audio)
        self.emotion_dialect = dialect
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        self.spoken.append(text)
        return await super().synthesize(text)


async def test_emotion_decorates_only_capable_tier():
    dia = RecordingSynthesizer("local:dia", b"AUDIO", dialect="dia")
    chain = FallbackSynthesizer([dia])
    result = await chain.synthesize("Que ótima notícia! Parabéns.", emotion="laugh")
    assert dia.spoken == ["Que ótima notícia! (laughs) Parabéns."]
    assert result.chars == len(dia.spoken[0])       # metering reflects what was spoken


async def test_emotion_never_reaches_plain_tier_on_failover():
    dia = RecordingSynthesizer("local:dia", b"", dialect="dia")      # falha → failover
    kokoro = RecordingSynthesizer("local:kokoro", b"AUDIO")          # sem dialeto
    chain = FallbackSynthesizer([dia, kokoro])
    result = await chain.synthesize("Que ótima notícia! Parabéns.", emotion="laugh")
    assert "(laughs)" in dia.spoken[0]              # tier capaz recebeu a tag
    assert kokoro.spoken == ["Que ótima notícia! Parabéns."]   # o fallback NÃO
    assert result.tier == "local:kokoro"


async def test_leaked_inline_tags_are_stripped_for_all_tiers():
    kokoro = RecordingSynthesizer("local:kokoro", b"AUDIO")
    chain = FallbackSynthesizer([kokoro])
    await chain.synthesize("Oi (laughs) tudo bem? <sigh> Certo.")
    assert kokoro.spoken == ["Oi tudo bem? Certo."]   # nunca lê "laughs" em voz alta


async def test_no_emotion_is_the_old_behaviour():
    kokoro = RecordingSynthesizer("local:kokoro", b"AUDIO")
    chain = FallbackSynthesizer([kokoro])
    await chain.synthesize("Olá! Tudo certo.")
    assert kokoro.spoken == ["Olá! Tudo certo."]
