"""Unit tests for cogno_vox vision port types and protocols."""

from dataclasses import is_dataclass

from cogno_vox import (
    SUPPORTED_VISION_FORMATS,
    VisionAnalysisResult,
    VisionAnalyzerBackend,
    VisionError,
    VoxConfig,
    VoxError,
)


def test_vision_error_hierarchy():
    assert issubclass(VisionError, VoxError)
    err = VisionError("All vision tiers failed")
    assert str(err) == "All vision tiers failed"


def test_supported_vision_formats():
    assert "png" in SUPPORTED_VISION_FORMATS
    assert "jpg" in SUPPORTED_VISION_FORMATS
    assert "mp4" in SUPPORTED_VISION_FORMATS
    assert "webm" in SUPPORTED_VISION_FORMATS


def test_vision_analysis_result():
    res = VisionAnalysisResult(
        summary="Comprovante de pagamento PIX",
        category="PIX_RECEIPT",
        extracted_data={"amount": 100.0, "recipient": "João"},
        confidence=0.95,
        tier="local:qwen2.5-vl-7b",
        elapsed_ms=120.5,
    )
    assert is_dataclass(res)
    assert res.summary == "Comprovante de pagamento PIX"
    assert res.category == "PIX_RECEIPT"
    assert res.extracted_data["amount"] == 100.0
    assert res.confidence == 0.95
    assert res.tier == "local:qwen2.5-vl-7b"
    assert res.elapsed_ms == 120.5


def test_vox_config_vision_tiers():
    cfg = VoxConfig()
    assert cfg.vision_tiers == ()


class DummyVisionBackend:
    @property
    def name(self) -> str:
        return "dummy:vision"

    async def analyze(
        self,
        media_bytes: bytes,
        filename_or_mime: str = "image.png",
        *,
        prompt: str = ""
    ) -> VisionAnalysisResult | None:
        if not media_bytes:
            return None
        return VisionAnalysisResult(
            summary="Test image analysis",
            category="GENERAL_IMAGE",
            tier=self.name,
        )


def test_vision_analyzer_backend_protocol():
    dummy = DummyVisionBackend()
    assert isinstance(dummy, VisionAnalyzerBackend)
    assert dummy.name == "dummy:vision"
