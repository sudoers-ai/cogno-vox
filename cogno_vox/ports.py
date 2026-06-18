"""
cogno_vox.ports — the two capability-scoped protocols + error types.

Structurally typed (``runtime_checkable`` ``Protocol``, no inheritance needed),
mirroring ``LLMBackend``/``Embedder`` in cogno-anima and the three ports in
cogno-engram. A backend is anything that has ``name`` and the one async method.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class VoxError(Exception):
    """Base error for the voice I/O edge."""


class TranscriptionError(VoxError):
    """Raised when transcription fails across every configured tier."""


class SynthesisError(VoxError):
    """Raised when synthesis fails across every configured tier."""


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
