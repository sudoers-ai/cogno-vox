"""
Live integration tests against real local STT/TTS servers.

Gated on env vars (auto-skip when unset), mirroring cogno-engram's
ENGRAM_TEST_DSN pattern — so this suite is a no-op in plain CI but runs
end-to-end when pointed at live servers:

    VOX_TEST_WHISPER_URL=http://localhost:8000 \
    VOX_TEST_KOKORO_URL=http://localhost:8880/v1 \
    pytest tests/integration -q

Validated against fedirz/faster-whisper-server (Systran/faster-whisper-small)
and ghcr.io/remsky/kokoro-fastapi-cpu. The first Whisper call may download the
model — hence the generous transcriber timeout.
"""

from __future__ import annotations

import os
import re

import pytest

from cogno_vox import (
    DeliveryProfile,
    TierConfig,
    VoxConfig,
    create_synthesizer,
    create_transcriber,
    split_text_for_tts,
)

WHISPER_URL = os.environ.get("VOX_TEST_WHISPER_URL")
KOKORO_URL = os.environ.get("VOX_TEST_KOKORO_URL")
WHISPER_MODEL = os.environ.get("VOX_TEST_WHISPER_MODEL", "Systran/faster-whisper-small")
KOKORO_VOICE = os.environ.get("VOX_TEST_KOKORO_VOICE", "af_alloy")

needs_kokoro = pytest.mark.skipif(not KOKORO_URL, reason="set VOX_TEST_KOKORO_URL")
needs_whisper = pytest.mark.skipif(not WHISPER_URL, reason="set VOX_TEST_WHISPER_URL")
needs_both = pytest.mark.skipif(
    not (KOKORO_URL and WHISPER_URL), reason="set VOX_TEST_KOKORO_URL + VOX_TEST_WHISPER_URL"
)


def _synthesizer(fmt: str = "wav"):
    return create_synthesizer(VoxConfig(synthesize_tiers=(
        TierConfig(provider="local", model="kokoro", base_url=KOKORO_URL or "",
                   voice=KOKORO_VOICE, response_format=fmt, timeout=60),
    )))


def _transcriber():
    return create_transcriber(VoxConfig(transcribe_tiers=(
        TierConfig(provider="local", model=WHISPER_MODEL, base_url=WHISPER_URL or "",
                   timeout=180),  # first call may download the model
    )))


def _words(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]", "", text.lower()).split())


@needs_kokoro
async def test_kokoro_synthesizes_audio():
    result = await _synthesizer("wav").synthesize("Hello from cogno vox.")
    assert result.audio  # non-empty bytes
    assert result.fmt == "wav"
    assert result.tier == "local:kokoro"
    assert result.elapsed_ms > 0
    assert result.nbytes == len(result.audio)


@needs_kokoro
async def test_kokoro_opus_format():
    result = await _synthesizer("opus").synthesize("Voice note in opus.")
    assert result.audio
    assert result.fmt == "opus"


@needs_both
async def test_round_trip_tts_then_stt():
    phrase = "The quick brown fox jumps over the lazy dog."
    spoken = await _synthesizer("wav").synthesize(phrase)
    assert spoken.audio

    heard = await _transcriber().transcribe(spoken.audio, "roundtrip.wav")
    assert heard.text.strip()
    assert heard.tier == f"local:{WHISPER_MODEL}"

    # ASR need not be byte-exact; assert the distinctive content words survive.
    expected = {"quick", "brown", "fox", "jumps", "lazy", "dog"}
    assert expected <= _words(heard.text)


@needs_both
async def test_round_trip_segmented():
    text = ("This is the first sentence of a longer reply. "
            "Here is a second sentence with more words. "
            "And a third one to be safe.")
    transcriber = _transcriber()
    synthesizer = _synthesizer("wav")

    heard_all: set[str] = set()
    for segment in split_text_for_tts(text, max_words=8):
        spoken = await synthesizer.synthesize(segment)
        heard = await transcriber.transcribe(spoken.audio, "seg.wav")
        heard_all |= _words(heard.text)

    assert {"first", "second", "third", "sentence"} <= heard_all


# ── delivery profile, against a REAL engine ───────────────────────────────
#
# The unit suite proves the contract against stubs. What it cannot prove is the claim the whole
# design rests on: that sending `instructions` to a server that never heard of the field is
# harmless. Kokoro is exactly that server — OpenAI-compatible in shape, without the shaping
# extension — so this is the live check that fail-open is real and not an assumption about how
# strangers parse JSON.

@needs_kokoro
async def test_a_delivery_profile_does_not_break_an_engine_that_ignores_it():
    """The fail-open claim, live. A 400 here (strict schema on an unknown key) would mean the
    profile has to be opt-in per engine rather than sent freely — the design decision this
    test exists to keep honest."""
    shaped = await _synthesizer("wav").synthesize(
        "Bom dia, tudo bem?", delivery=DeliveryProfile(style="warm", pace="slow", energy="low"))
    assert shaped.audio and len(shaped.audio) > 1000
    assert shaped.tier.startswith("local:")


@needs_both
async def test_shaping_does_not_cost_the_WORDS():
    """A profile changes HOW, never WHAT. If an engine honours it by whispering or rushing, the
    listener pays — and the round trip is the cheapest place that shows up.

    Asserted as a COMPARISON between the shaped and plain runs, not against a fixed word set.
    The first version checked a Portuguese phrase against expected Portuguese words and failed
    on this server — which transcribes pt-BR audio into English — measuring the ASR's language
    behaviour instead of the thing under test. What the claim is actually about is a delta, so
    the plain run is the baseline and any content word it recovers must survive shaping.
    """
    phrase = "The quick brown fox jumps over the lazy dog."
    synthesizer, transcriber = _synthesizer("wav"), _transcriber()

    plain = await synthesizer.synthesize(phrase)
    shaped = await synthesizer.synthesize(
        phrase, delivery=DeliveryProfile(style="warm", pace="slow"))
    assert plain.audio and shaped.audio

    heard_plain = _words((await transcriber.transcribe(plain.audio, "plain.wav")).text)
    heard_shaped = _words((await transcriber.transcribe(shaped.audio, "shaped.wav")).text)

    content = {"quick", "brown", "fox", "jumps", "lazy", "dog"} & heard_plain
    assert content, f"baseline itself lost the words: {heard_plain}"
    assert content <= heard_shaped, f"shaping lost {content - heard_shaped} (heard {heard_shaped})"
