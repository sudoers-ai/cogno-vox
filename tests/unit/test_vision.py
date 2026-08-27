"""Unit tests for cogno_vox vision analyzers and fallback chains."""

import pytest
import httpx
from cogno_vox import (
    FallbackVisionAnalyzer,
    OpenAICompatVisionAnalyzer,
    TierConfig,
    VisionError,
    VoxConfig,
    create_vision_analyzer,
)


@pytest.mark.asyncio
async def test_openai_compat_vision_analyzer_success():
    mock_json_response = {
        "choices": [
            {
                "message": {
                    "content": '```json\n{"summary": "Comprovante PIX de R$ 150,00", "category": "PIX_RECEIPT", "extracted_data": {"amount": 150.0}, "confidence": 0.99}\n```'
                }
            }
        ]
    }

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=mock_json_response)
    )

    async with httpx.AsyncClient(transport=transport) as client:
        tier = TierConfig(provider="local", model="qwen2.5-vl-7b", base_url="http://localhost:11434/v1")
        analyzer = OpenAICompatVisionAnalyzer(tier, client=client)

        result = await analyzer.analyze(b"FAKE_PNG_BYTES", "receipt.png")
        assert result is not None
        assert result.summary == "Comprovante PIX de R$ 150,00"
        assert result.category == "PIX_RECEIPT"
        assert result.extracted_data == {"amount": 150.0}
        assert result.confidence == 0.99
        assert result.tier == "local:qwen2.5-vl-7b"


@pytest.mark.asyncio
async def test_openai_compat_vision_analyzer_non_json_fallback():
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "Uma foto de um gato sentado na mesa."
                }
            }
        ]
    }

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=mock_response)
    )

    async with httpx.AsyncClient(transport=transport) as client:
        tier = TierConfig(provider="local", model="moondream2")
        analyzer = OpenAICompatVisionAnalyzer(tier, client=client)

        result = await analyzer.analyze(b"FAKE_JPG_BYTES", "cat.jpg")
        assert result is not None
        assert result.summary == "Uma foto de um gato sentado na mesa."
        assert result.category == "GENERAL_IMAGE"
        assert result.extracted_data == {}
        assert result.confidence == 0.8


@pytest.mark.asyncio
async def test_openai_compat_vision_analyzer_error():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(500, text="Internal Server Error")
    )

    async with httpx.AsyncClient(transport=transport) as client:
        tier = TierConfig(provider="local", model="qwen2.5-vl")
        analyzer = OpenAICompatVisionAnalyzer(tier, client=client)

        result = await analyzer.analyze(b"FAKE_PNG_BYTES", "image.png")
        assert result is None


@pytest.mark.asyncio
async def test_fallback_vision_analyzer_chain():
    mock_fail_transport = httpx.MockTransport(
        lambda request: httpx.Response(500, text="Error")
    )
    mock_success_transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "Resumo fallback ok"}}]})
    )

    async with httpx.AsyncClient(transport=mock_fail_transport) as client_fail, \
               httpx.AsyncClient(transport=mock_success_transport) as client_ok:

        tier1 = TierConfig(provider="local", model="failing-model")
        tier2 = TierConfig(provider="openai", model="gpt-4o-vision")

        analyzer1 = OpenAICompatVisionAnalyzer(tier1, client=client_fail)
        analyzer2 = OpenAICompatVisionAnalyzer(tier2, client=client_ok)

        chain = FallbackVisionAnalyzer([analyzer1, analyzer2])
        res = await chain.analyze(b"TEST_BYTES", "test.png")

        assert res.summary == "Resumo fallback ok"
        assert res.tier == "openai:gpt-4o-vision"


@pytest.mark.asyncio
async def test_fallback_vision_analyzer_all_fail():
    mock_fail = httpx.MockTransport(lambda req: httpx.Response(500, text="Failed"))

    async with httpx.AsyncClient(transport=mock_fail) as client:
        analyzer = OpenAICompatVisionAnalyzer(TierConfig(provider="local", model="m"), client=client)
        chain = FallbackVisionAnalyzer([analyzer])

        with pytest.raises(VisionError, match="All vision tiers failed"):
            await chain.analyze(b"TEST_BYTES", "test.png")


def test_create_vision_analyzer_factory():
    cfg = VoxConfig(
        vision_tiers=(
            TierConfig(provider="local", model="qwen2.5-vl-7b"),
            TierConfig(provider="openai", model="gpt-4o-mini-vision"),
        )
    )
    chain = create_vision_analyzer(cfg)
    assert len(chain.backends) == 2
    assert chain.backends[0].name == "local:qwen2.5-vl-7b"
    assert chain.backends[1].name == "openai:gpt-4o-mini-vision"
