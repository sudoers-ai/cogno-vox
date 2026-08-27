"""Unit tests for cogno_vox vision analyzers and fallback chains."""

import httpx
import pytest

from cogno_vox import (
    FallbackVisionAnalyzer,
    OpenAICompatVisionAnalyzer,
    TierConfig,
    VisionError,
    VoxConfig,
    create_vision_analyzer,
)
from cogno_vox.vision import _detect_mime_type


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


@pytest.mark.parametrize(
    "filename,media_bytes,expected",
    [
        ("foto.png", b"", "image/png"),
        ("foto.PNG", b"", "image/png"),
        ("foto.jpg", b"", "image/jpeg"),
        ("foto.jpeg", b"", "image/jpeg"),
        ("foto.JPEG", b"", "image/jpeg"),
        ("foto.webp", b"", "image/webp"),
        ("foto.gif", b"", "image/gif"),
        ("video.mp4", b"", "video/mp4"),
        ("video.webm", b"", "video/webm"),
        ("", b"\x89PNG\r\n\x1a\n1234", "image/png"),
        ("", b"\xff\xd8\xff1234", "image/jpeg"),
        ("", b"RIFF1234WEBPextra", "image/webp"),
        ("", b"DESCONHECIDO", "image/png"),  # Afirma valor por omissão ("image/png") por extenso
    ],
)
def test_detect_mime_type(filename, media_bytes, expected):
    assert _detect_mime_type(filename, media_bytes) == expected


@pytest.mark.asyncio
async def test_openai_compat_vision_analyzer_empty_media_bytes():
    analyzer = OpenAICompatVisionAnalyzer(TierConfig(provider="local", model="m"))
    assert await analyzer.analyze(b"", "image.png") is None


@pytest.mark.asyncio
async def test_openai_compat_vision_analyzer_video_keyframe_extraction(monkeypatch):
    """Cobre o caminho de processamento de vídeo com e sem keyframes extraídos."""
    from cogno_vox import vision as vs_mod

    # 1. Caso com keyframes extraídos
    monkeypatch.setattr(vs_mod, "extract_keyframes", lambda b, max_frames: [b"kf1", b"kf2"])
    captured_payload = {}

    def handler(request: httpx.Request):
        import json
        captured_payload.update(json.loads(request.read()))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Resumo do vídeo ok"}}]})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        analyzer = OpenAICompatVisionAnalyzer(TierConfig(provider="local", model="m"), client=client)
        res = await analyzer.analyze(b"fake_video_bytes", "sample.mp4")
        assert res is not None
        assert res.summary == "Resumo do vídeo ok"
        messages = captured_payload.get("messages", [])
        content = messages[0]["content"]
        assert len(content) == 3  # text prompt + 2 keyframes image_url

    # 2. Caso sem keyframes extraídos (fallback para raw bytes como jpeg)
    monkeypatch.setattr(vs_mod, "extract_keyframes", lambda b, max_frames: [])
    captured_payload.clear()

    async with httpx.AsyncClient(transport=transport) as client:
        analyzer = OpenAICompatVisionAnalyzer(TierConfig(provider="local", model="m"), client=client)
        res = await analyzer.analyze(b"raw_video_bytes", "sample.mp4")
        assert res is not None
        messages = captured_payload.get("messages", [])
        content = messages[0]["content"]
        assert len(content) == 2  # text prompt + 1 fallback raw image_url


@pytest.mark.asyncio
async def test_openai_compat_vision_analyzer_api_key_header():
    """Cobre a adição do cabeçalho Authorization quando api_key é informada."""
    auth_header = []

    def handler(request: httpx.Request):
        auth_header.append(request.headers.get("Authorization"))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        tier = TierConfig(provider="openai", model="gpt-4o", api_key="sk-test-secret-key")
        analyzer = OpenAICompatVisionAnalyzer(tier, client=client)
        res = await analyzer.analyze(b"PNG_BYTES", "img.png")
        assert res is not None
        assert auth_header == ["Bearer sk-test-secret-key"]


@pytest.mark.asyncio
async def test_openai_compat_vision_analyzer_self_managed_client(monkeypatch):
    """Cobre linhas 132-133 e 162: quando client é None, instancia e fecha o AsyncClient automaticamente."""
    from cogno_vox import vision as vs_mod

    closed = False

    class MockAsyncClient:
        def __init__(self, timeout=None):
            pass

        async def post(self, url, json=None, headers=None):
            return httpx.Response(200, json={"choices": [{"message": {"content": "self managed client ok"}}]})

        async def aclose(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr(vs_mod.httpx, "AsyncClient", MockAsyncClient)

    tier = TierConfig(provider="local", model="moondream")
    analyzer = OpenAICompatVisionAnalyzer(tier, client=None)
    res = await analyzer.analyze(b"IMG_BYTES", "img.png")
    assert res is not None
    assert res.summary == "self managed client ok"
    assert closed is True, "AsyncClient criado automaticamente deve ser fechado no finally"


@pytest.mark.asyncio
async def test_openai_compat_vision_analyzer_markdown_block_without_json_tag():
    """Cobre a linha 171: bloco de código markdown sem a tag 'json' explicitada."""
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": '```\n{"summary": "gato no sofá", "category": "PET"}\n```'
                }
            }
        ]
    }
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=mock_response))
    async with httpx.AsyncClient(transport=transport) as client:
        analyzer = OpenAICompatVisionAnalyzer(TierConfig(provider="local", model="m"), client=client)
        res = await analyzer.analyze(b"BYTES", "img.png")
        assert res is not None
        assert res.summary == "gato no sofá"
        assert res.category == "PET"

    """Cobre respostas sem choices ou com texto vazio."""
    # 1. Resposta sem choices
    transport1 = httpx.MockTransport(lambda req: httpx.Response(200, json={"choices": []}))
    async with httpx.AsyncClient(transport=transport1) as client:
        analyzer = OpenAICompatVisionAnalyzer(TierConfig(provider="local", model="m"), client=client)
        assert await analyzer.analyze(b"BYTES", "img.png") is None

    # 2. Choices com conteúdo vazio
    transport2 = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]})
    )
    async with httpx.AsyncClient(transport=transport2) as client:
        analyzer = OpenAICompatVisionAnalyzer(TierConfig(provider="local", model="m"), client=client)
        assert await analyzer.analyze(b"BYTES", "img.png") is None


@pytest.mark.asyncio
async def test_openai_compat_vision_analyzer_http_exception_returns_none(caplog):
    """Cobre exceções de rede durante a chamada HTTP (linhas 157-159)."""
    import logging

    def handler(request: httpx.Request):
        raise httpx.ConnectError("Conexão recusada pelo servidor VLLM")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        analyzer = OpenAICompatVisionAnalyzer(TierConfig(provider="local", model="m"), client=client)
        with caplog.at_level(logging.WARNING):
            res = await analyzer.analyze(b"BYTES", "img.png")

        assert res is None
        logs = "\n".join(r.getMessage() for r in caplog.records)
        assert "vision local:m failed" in logs
        assert "Conexão recusada" in logs


def test_fallback_vision_analyzer_empty_backends():
    """Cobre a validação de lista de backends vazia (linha 205)."""
    with pytest.raises(ValueError, match="FallbackVisionAnalyzer requires at least one backend"):
        FallbackVisionAnalyzer([])


@pytest.mark.asyncio
async def test_fallback_vision_analyzer_empty_media_bytes():
    """Cobre a validação de bytes vazios na cadeia de fallback (linha 220)."""
    analyzer = OpenAICompatVisionAnalyzer(TierConfig(provider="local", model="m"))
    chain = FallbackVisionAnalyzer([analyzer])
    with pytest.raises(VisionError, match="Empty media bytes provided for vision analysis"):
        await chain.analyze(b"", "img.png")


@pytest.mark.asyncio
async def test_fallback_vision_analyzer_all_tiers_fail_with_detailed_error_messages():
    """Cobre o tratamento de erros na cadeia de fallback (linhas 229-231 e 233).
    
    Verifica que uma falha total diz exatamente o que aconteceu em cada tier na mensagem de exceção.
    """
    class FailingBackend1:
        @property
        def name(self) -> str:
            return "tier1:local"

        async def analyze(self, media_bytes, filename_or_mime="image.png", *, prompt=""):
            return None  # Devolve nulo (resposta vazia)

    class FailingBackend2:
        @property
        def name(self) -> str:
            return "tier2:cloud"

        async def analyze(self, media_bytes, filename_or_mime="image.png", *, prompt=""):
            raise RuntimeError("Erro de autenticação API Key invalida")

    chain = FallbackVisionAnalyzer([FailingBackend1(), FailingBackend2()])

    with pytest.raises(VisionError) as exc_info:
        await chain.analyze(b"IMG_BYTES", "img.png")

    msg = str(exc_info.value)
    assert "All vision tiers failed" in msg
    assert "tier1:local: returned empty result" in msg
    assert "tier2:cloud: Erro de autenticação API Key invalida" in msg

