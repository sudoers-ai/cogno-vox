"""
Integration: real round-trip WER against OpenAI-compatible STT/TTS servers.

Auto-skips unless both a TTS and an STT endpoint are configured AND reachable, so
the suite stays green in CI/dev without audio models. Point it at a local
faster-whisper-server + Kokoro (or the OpenAI/Groq clouds) via env::

    VOX_TTS_BASE_URL  VOX_TTS_MODEL  VOX_TTS_API_KEY  VOX_TTS_VOICE
    VOX_STT_BASE_URL  VOX_STT_MODEL  VOX_STT_API_KEY
"""

import os

import httpx
import pytest

from cognobench.cases import CASES
from cognobench.runner import run_bench
from cogno_vox.synthesizer import OpenAICompatSynthesizer
from cogno_vox.transcriber import OpenAICompatTranscriber

TTS_URL = os.getenv("VOX_TTS_BASE_URL", "")
STT_URL = os.getenv("VOX_STT_BASE_URL", "")


def _reachable(url: str) -> bool:
    if not url:
        return False
    try:
        httpx.get(url, timeout=3)
        return True
    except Exception:
        return False


requires_audio = pytest.mark.skipif(
    not (_reachable(TTS_URL) and _reachable(STT_URL)),
    reason="VOX_TTS_BASE_URL / VOX_STT_BASE_URL not configured or unreachable",
)


def _synth():
    return OpenAICompatSynthesizer(
        base_url=TTS_URL, model=os.getenv("VOX_TTS_MODEL", "kokoro"),
        api_key=os.getenv("VOX_TTS_API_KEY", ""), voice=os.getenv("VOX_TTS_VOICE", "alloy"),
        response_format="wav",
    )


def _stt():
    return OpenAICompatTranscriber(
        base_url=STT_URL, model=os.getenv("VOX_STT_MODEL", "Systran/faster-whisper-small"),
        api_key=os.getenv("VOX_STT_API_KEY", ""),
    )


@requires_audio
async def test_round_trip_produces_audio_and_transcribes():
    report = await run_bench(_synth(), _stt(), cases=CASES[:3])
    # every tier produced audio and some transcription
    assert report.empty_outputs == 0


@requires_audio
async def test_round_trip_wer_floor():
    # Clear short utterances through a competent STT should round-trip well.
    # Ceiling at 0.40 mean WER catches gross regressions / misconfiguration
    # without being brittle to one-off recognition slips.
    report = await run_bench(_synth(), _stt(), cases=CASES)
    assert report.mean_wer <= 0.40, f"round-trip WER too high: {report.mean_wer:.3f}"
