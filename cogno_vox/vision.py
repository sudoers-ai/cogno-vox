"""
cogno_vox.vision — Vision and multimodal analysis backends + fallback chain.

Converts images and video keyframes into structured VisionAnalysisResult instances
behind provider-agnostic fallback chains.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Sequence, Optional

import httpx

from cogno_vox.ports import VisionAnalyzerBackend, VisionError
from cogno_vox.types import TierConfig, VisionAnalysisResult
from cogno_vox.video_sampler import extract_keyframes

logger = logging.getLogger(__name__)

# Default prompt used when analyzing images/documents if none is provided
DEFAULT_VISION_PROMPT = (
    "Analise esta imagem ou mídia detalhadamente. Identifique o tipo de documento ou conteúdo "
    "(ex: PIX_RECEIPT, ID_DOCUMENT, GENERAL_IMAGE, VIDEO). Retorne um JSON válido no formato:\n"
    "{\n"
    '  "summary": "Descrição clara e concisa do conteúdo em português",\n'
    '  "category": "CATEGORIA",\n'
    '  "extracted_data": {"chave": "valor"},\n'
    '  "confidence": 0.95\n'
    "}"
)


def _detect_mime_type(filename_or_mime: str, media_bytes: bytes) -> str:
    """Infer MIME type from filename/MIME string or magic bytes."""
    val = filename_or_mime.lower()
    if "pdf" in val or val.endswith(".pdf"):
        return "application/pdf"
    if "jpeg" in val or val.endswith(".jpg") or val.endswith(".jpeg"):
        return "image/jpeg"
    if "png" in val or val.endswith(".png"):
        return "image/png"
    if "webp" in val or val.endswith(".webp"):
        return "image/webp"
    if "gif" in val or val.endswith(".gif"):
        return "image/gif"
    if "mp4" in val or val.endswith(".mp4"):
        return "video/mp4"
    if "webm" in val or val.endswith(".webm"):
        return "video/webm"

    # Check magic numbers
    if media_bytes.startswith(b"%PDF"):
        return "application/pdf"
    if media_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if media_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if media_bytes.startswith(b"RIFF") and media_bytes[8:12] == b"WEBP":
        return "image/webp"

    return "image/png"


class OpenAICompatVisionAnalyzer:
    """Vision backend driving OpenAI-compatible VLLM endpoints (Ollama, Qwen2.5-VL, vLLM)."""

    def __init__(self, tier: TierConfig, client: Optional[httpx.AsyncClient] = None) -> None:
        self._tier = tier
        self._base_url = (tier.base_url or "http://localhost:11434/v1").rstrip("/")
        self._model = tier.model
        self._api_key = tier.api_key
        self._timeout = tier.timeout
        self._client = client

    @property
    def name(self) -> str:
        return f"{self._tier.provider}:{self._model}"

    async def analyze(
        self,
        media_bytes: bytes,
        filename_or_mime: str = "image.png",
        *,
        prompt: str = ""
    ) -> Optional[VisionAnalysisResult]:
        if not media_bytes:
            return None

def _prepare_vision_media(
    media_bytes: bytes, filename_or_mime: str
) -> tuple[list[tuple[str, str]], int, int]:
    """Prepare media bytes as a list of (mime_type, base64_str) tuples, plus PDF page counts."""
    mime = _detect_mime_type(filename_or_mime, media_bytes)
    images_b64: list[tuple[str, str]] = []
    pdf_pages_read = 0
    pdf_total_pages = 0

    if mime == "application/pdf":
        try:
            import io
            import pypdfium2 as pdfium
        except ImportError as err:
            raise VisionError(
                "PDF vision conversion requires 'pypdfium2'. Please install with: pip install cogno-vox[vision] or pip install pypdfium2"
            ) from err

        try:
            pdf = pdfium.PdfDocument(media_bytes)
            pdf_total_pages = len(pdf)
            max_pages = 5
            pdf_pages_read = min(pdf_total_pages, max_pages)

            for i in range(pdf_pages_read):
                image = pdf[i].render(scale=2).to_pil()
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                images_b64.append(("image/jpeg", b64))
        except Exception as exc:
            if isinstance(exc, VisionError):
                raise
            raise VisionError(f"Failed to render PDF pages for vision analysis: {exc}") from exc

    elif mime.startswith("video/"):
        keyframes = extract_keyframes(media_bytes, max_frames=8)
        if not keyframes:
            b64 = base64.b64encode(media_bytes).decode("ascii")
            images_b64.append(("image/jpeg", b64))
        else:
            for kf in keyframes:
                b64 = base64.b64encode(kf).decode("ascii")
                images_b64.append(("image/jpeg", b64))
    else:
        b64 = base64.b64encode(media_bytes).decode("ascii")
        images_b64.append((mime, b64))

    return images_b64, pdf_pages_read, pdf_total_pages


def _parse_vision_response(
    tier_name: str,
    raw_text: str,
    elapsed_ms: float,
    *,
    tokens_in: int = 0,
    tokens_out: int = 0,
    pdf_pages_read: int = 0,
    pdf_total_pages: int = 0,
) -> VisionAnalysisResult:
    """Attempt JSON extraction or fallback to plain summary."""
    text = raw_text
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            extracted = parsed.get("extracted_data")
            extracted_data: dict[str, object] = extracted if isinstance(extracted, dict) else {}
            return VisionAnalysisResult(
                summary=str(parsed.get("summary") or raw_text),
                category=str(parsed.get("category") or "GENERAL_IMAGE"),
                extracted_data=extracted_data,
                confidence=float(parsed.get("confidence") or 1.0),
                tier=tier_name,
                elapsed_ms=elapsed_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                pdf_pages_read=pdf_pages_read,
                pdf_total_pages=pdf_total_pages,
            )
    except Exception:
        pass

    return VisionAnalysisResult(
        summary=raw_text,
        category="GENERAL_IMAGE",
        extracted_data={},
        confidence=0.8,
        tier=tier_name,
        elapsed_ms=elapsed_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        pdf_pages_read=pdf_pages_read,
        pdf_total_pages=pdf_total_pages,
    )


class OpenAICompatVisionAnalyzer:
    """Vision backend driving OpenAI-compatible VLLM endpoints (Ollama, Qwen2.5-VL, vLLM)."""

    def __init__(self, tier: TierConfig, client: Optional[httpx.AsyncClient] = None) -> None:
        self._tier = tier
        self._base_url = (tier.base_url or "http://localhost:11434/v1").rstrip("/")
        self._model = tier.model
        self._api_key = tier.api_key
        self._timeout = tier.timeout
        self._client = client

    @property
    def name(self) -> str:
        return f"{self._tier.provider}:{self._model}"

    async def analyze(
        self,
        media_bytes: bytes,
        filename_or_mime: str = "image.png",
        *,
        prompt: str = ""
    ) -> Optional[VisionAnalysisResult]:
        if not media_bytes:
            return None

        images_b64, pdf_pages_read, pdf_total_pages = _prepare_vision_media(media_bytes, filename_or_mime)

        content_payload: list[dict[str, object]] = [
            {"type": "text", "text": prompt or DEFAULT_VISION_PROMPT}
        ]
        for mime_type, raw_b64 in images_b64:
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{raw_b64}"}
            })

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": content_payload}],
            "temperature": 0.1,
        }

        url = f"{self._base_url}/chat/completions"
        start_t = time.perf_counter()

        close_client = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=self._timeout)
            close_client = True

        try:
            resp = await client.post(url, json=payload, headers=headers)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            if resp.status_code != 200:
                logger.warning(
                    "vision %s HTTP %d",
                    self.name, resp.status_code
                )
                return None

            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                return None

            usage = data.get("usage") or {}
            tokens_in = int(usage.get("prompt_tokens") or usage.get("prompt_eval_count") or 0)
            tokens_out = int(usage.get("completion_tokens") or usage.get("eval_count") or 0)

            raw_text = choices[0].get("message", {}).get("content", "").strip()
            if not raw_text:
                return None

            return _parse_vision_response(
                self.name,
                raw_text,
                elapsed_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                pdf_pages_read=pdf_pages_read,
                pdf_total_pages=pdf_total_pages,
            )

        except Exception as exc:
            # SECURITY (Test 9): NEVER log self._api_key or full headers/payload!
            logger.warning("vision %s failed: %s", self.name, exc)
            return None
        finally:
            if close_client and client is not None:
                await client.aclose()


class AnthropicVisionAnalyzer:
    """Vision backend driving Anthropic Messages API (Claude 3 / 3.5)."""

    def __init__(self, tier: TierConfig, client: Optional[httpx.AsyncClient] = None) -> None:
        self._tier = tier
        self._base_url = (tier.base_url or "https://api.anthropic.com").rstrip("/")
        self._model = tier.model
        self._api_key = tier.api_key
        self._timeout = tier.timeout
        self._client = client

    @property
    def name(self) -> str:
        return f"{self._tier.provider}:{self._model}"

    async def analyze(
        self,
        media_bytes: bytes,
        filename_or_mime: str = "image.png",
        *,
        prompt: str = ""
    ) -> Optional[VisionAnalysisResult]:
        if not media_bytes:
            return None

        images_b64, pdf_pages_read, pdf_total_pages = _prepare_vision_media(media_bytes, filename_or_mime)

        content_payload: list[dict[str, object]] = [
            {"type": "text", "text": prompt or DEFAULT_VISION_PROMPT}
        ]
        for mime_type, raw_b64 in images_b64:
            content_payload.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": raw_b64,
                }
            })

        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key

        payload = {
            "model": self._model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": content_payload}],
        }

        url = f"{self._base_url}/v1/messages" if not self._base_url.endswith("/v1") else f"{self._base_url}/messages"
        start_t = time.perf_counter()

        close_client = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=self._timeout)
            close_client = True

        try:
            resp = await client.post(url, json=payload, headers=headers)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            if resp.status_code != 200:
                logger.warning(
                    "vision %s HTTP %d",
                    self.name, resp.status_code
                )
                return None

            data = resp.json()
            content_blocks = data.get("content") or []
            raw_text = "".join(
                b.get("text", "") for b in content_blocks if isinstance(b, dict) and b.get("type") == "text"
            ).strip()

            if not raw_text:
                return None

            usage = data.get("usage") or {}
            tokens_in = int(usage.get("input_tokens") or 0)
            tokens_out = int(usage.get("output_tokens") or 0)

            return _parse_vision_response(
                self.name,
                raw_text,
                elapsed_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                pdf_pages_read=pdf_pages_read,
                pdf_total_pages=pdf_total_pages,
            )

        except Exception as exc:
            # SECURITY (Test 9): NEVER log self._api_key or full headers/payload!
            logger.warning("vision %s failed: %s", self.name, exc)
            return None
        finally:
            if close_client and client is not None:
                await client.aclose()


class FallbackVisionAnalyzer:
    """Vision analyzer fallback chain — tries backends in order, first non-None wins."""

    def __init__(self, backends: Sequence[VisionAnalyzerBackend]) -> None:
        if not backends:
            raise ValueError("FallbackVisionAnalyzer requires at least one backend")
        self._backends = tuple(backends)

    @property
    def backends(self) -> tuple[VisionAnalyzerBackend, ...]:
        return self._backends

    async def analyze(
        self,
        media_bytes: bytes,
        filename_or_mime: str = "image.png",
        *,
        prompt: str = ""
    ) -> VisionAnalysisResult:
        if not media_bytes:
            raise VisionError("Empty media bytes provided for vision analysis")

        errors: list[str] = []
        for backend in self._backends:
            try:
                res = await backend.analyze(media_bytes, filename_or_mime, prompt=prompt)
                if res is not None and res.summary:
                    return res
                errors.append(f"{backend.name}: returned empty result")
            except Exception as exc:
                logger.warning("vision chain backend %s failed: %s", backend.name, exc)
                errors.append(f"{backend.name}: {exc}")

        raise VisionError(f"All vision tiers failed: {'; '.join(errors)}")
