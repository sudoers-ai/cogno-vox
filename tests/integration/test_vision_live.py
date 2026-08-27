"""
Live integration tests against real VLLM vision endpoints (Ollama / vLLM / OpenAI).

Gated on env vars (auto-skip when unset), mirroring cogno-vox's STT/TTS pattern:

    VOX_TEST_OLLAMA_URL=http://localhost:11434/v1 \
    VOX_TEST_OLLAMA_VISION_MODEL=qwen2.5-vl-7b \
    pytest tests/integration -q
"""

from __future__ import annotations

import os
import pytest

from cogno_vox import (
    OpenAICompatVisionAnalyzer,
    TierConfig,
    VisionAnalysisResult,
    create_vision_analyzer,
    VoxConfig,
)

OLLAMA_URL = os.environ.get("VOX_TEST_OLLAMA_URL")
VISION_MODEL = os.environ.get("VOX_TEST_OLLAMA_VISION_MODEL", "qwen2.5-vl-7b")

needs_ollama_vision = pytest.mark.skipif(
    not OLLAMA_URL, reason="set VOX_TEST_OLLAMA_URL to run live vision integration tests"
)

# 1x1 Red PNG image
SAMPLE_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


@needs_ollama_vision
async def test_live_ollama_vision_analysis():
    cfg = VoxConfig(
        vision_tiers=(
            TierConfig(provider="local", model=VISION_MODEL, base_url=OLLAMA_URL or "http://localhost:11434/v1"),
        )
    )
    analyzer = create_vision_analyzer(cfg)
    result = await analyzer.analyze(SAMPLE_PNG_BYTES, "test.png")

    assert isinstance(result, VisionAnalysisResult)
    assert result.summary
    assert result.tier == f"local:{VISION_MODEL}"
    assert result.elapsed_ms > 0
