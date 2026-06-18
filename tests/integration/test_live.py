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
