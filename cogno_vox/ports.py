"""
cogno_vox.ports — the two capability-scoped protocols + error types.

Structurally typed (``runtime_checkable`` ``Protocol``, no inheritance needed),
mirroring ``LLMBackend``/``Embedder`` in cogno-anima and the three ports in
cogno-engram. A backend is anything that has ``name`` and the one async method.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from cogno_vox.types import DeliveryProfile, VisionAnalysisResult



class VoxError(Exception):
    """Base error for the voice I/O edge."""


class TranscriptionError(VoxError):
    """Raised when transcription fails across every configured tier."""


class SynthesisError(VoxError):
    """Raised when synthesis fails across every configured tier."""


class VisionError(VoxError):
    """Raised when vision analysis fails across every configured tier."""


@runtime_checkable
class TranscriberBackend(Protocol):
    """A single speech-to-text backend (one tier of a fallback chain)."""

    @property
    def name(self) -> str: ...

    async def transcribe(self, audio: bytes, filename: str = "audio.ogg") -> str:
        """Return transcribed text, or ``""`` on failure (chain fails over)."""
        ...


@runtime_checkable
class SynthesizerBackend(Protocol):
    """A single text-to-speech backend (one tier of a fallback chain)."""

    @property
    def name(self) -> str: ...

    async def synthesize(self, text: str) -> bytes:
        """Return encoded audio bytes, or ``b""`` on failure (chain fails over)."""
        ...


@runtime_checkable
class VisionAnalyzerBackend(Protocol):
    """A single vision/multimodal analysis backend (one tier of a fallback chain)."""

    @property
    def name(self) -> str: ...

    async def analyze(
        self,
        media_bytes: bytes,
        filename_or_mime: str = "image.png",
        *,
        prompt: str = ""
    ) -> Optional[VisionAnalysisResult]:
        """Return VisionAnalysisResult, or ``None`` on failure (chain fails over)."""
        ...


@runtime_checkable
class DeliveryAwareBackend(Protocol):
    """OPTIONAL: a backend that can also be told HOW to say the words.

    Deliberately a second protocol rather than a widened ``SynthesizerBackend``, for the same
    reason ``ToolCallingBackend`` is separate from ``LLMBackend`` in the sibling libs: most
    engines cannot shape delivery at all, and a base protocol every implementation must satisfy
    would force each of them to carry a method that does nothing. A tier is probed with
    ``isinstance`` and falls back to plain ``synthesize`` — so a chain can mix a shaping engine
    with a plain one and a failover never turns "unshaped" into "failed".
    """

    async def synthesize_shaped(self, text: str, delivery: "DeliveryProfile") -> bytes:
        """Same contract as ``synthesize``: audio bytes, or ``b""`` on failure."""
        ...

