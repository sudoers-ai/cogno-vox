"""
Live integration: the Dia wrapper server end-to-end through the vox chain.

Gated on ``VOX_TEST_DIA_URL`` (auto-skip when unset), mirroring test_live.py::

    # terminal 1 — the wrapper over a real checkpoint (see cogno_vox.dia_server)
    python -m cogno_vox.dia_server --config config.json \
        --checkpoint Dia1.6-Portuguese-v1-merged.safetensors --port 8881

    # terminal 2
    VOX_TEST_DIA_URL=http://localhost:8881/v1 pytest tests/integration/test_dia_live.py -q

Validates the full emotion contract on real audio: the chain decorates the text in
the Dia dialect, the wrapper voices it, non-empty WAV comes back.
"""

from __future__ import annotations

import os

import pytest

from cogno_vox import TierConfig, VoxConfig, create_synthesizer

DIA_URL = os.environ.get("VOX_TEST_DIA_URL", "")

pytestmark = pytest.mark.skipif(not DIA_URL, reason="VOX_TEST_DIA_URL not set")


def _config() -> VoxConfig:
    return VoxConfig(synthesize_tiers=(
        TierConfig(provider="local", model="dia-ptbr", base_url=DIA_URL,
                   response_format="wav", timeout=180.0, emotion_dialect="dia"),
    ))


@pytest.mark.asyncio
async def test_emotion_synthesis_roundtrip():
    chain = create_synthesizer(_config())
    result = await chain.synthesize("Que ótima notícia! Parabéns pelo cadastro.",
                                    emotion="laugh")
    assert result.audio and result.audio[:4] == b"RIFF"      # non-empty WAV
    # chars metered over the DECORATED text (the tag was injected for this tier)
    assert result.chars == len("Que ótima notícia! (laughs) Parabéns pelo cadastro.")


@pytest.mark.asyncio
async def test_plain_synthesis_has_no_tag():
    chain = create_synthesizer(_config())
    result = await chain.synthesize("Seu horário foi confirmado para amanhã às onze.")
    assert result.audio and result.chars == len(
        "Seu horário foi confirmado para amanhã às onze.")
