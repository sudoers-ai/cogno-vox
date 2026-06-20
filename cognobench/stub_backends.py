"""
Deterministic, network-free TTS+STT doubles for ``--stub`` runs.

The stub "synthesizes" text by UTF-8 encoding it (with a tiny header so the bytes
look like an opaque container) and "transcribes" by decoding it back — a lossless
round-trip, so WER is 0. This guards the bench harness (case loading, round-trip
wiring, WER, reporting) in CI without any audio model, mirroring the stub mode of
the sibling cognobenches.
"""

from __future__ import annotations

_HEADER = b"STUBWAV\x00"


class StubSynthesizer:
    """``SynthesizerBackend`` that encodes text into opaque bytes."""

    name = "stub:tts"
    fmt = "wav"

    async def synthesize(self, text: str) -> bytes:
        if not text:
            return b""
        return _HEADER + text.encode("utf-8")


class StubTranscriber:
    """``TranscriberBackend`` that decodes the stub synthesizer's bytes back to text."""

    name = "stub:stt"

    async def transcribe(self, audio: bytes, filename: str = "audio.wav") -> str:
        if not audio or not audio.startswith(_HEADER):
            return ""
        return audio[len(_HEADER):].decode("utf-8")
