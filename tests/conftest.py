"""Shared test doubles for cogno-vox unit tests (zero network)."""

from __future__ import annotations

import pytest


class StubTranscriber:
    """A scripted transcriber backend — returns ``text`` (or "" to fail over)."""

    def __init__(self, name: str, text: str) -> None:
        self._name = name
        self._text = text
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def transcribe(self, audio: bytes, filename: str = "audio.ogg") -> str:
        self.calls += 1
        return self._text


class StubSynthesizer:
    """A scripted synthesizer backend — returns ``audio`` (or b"" to fail over)."""

    fmt = "opus"

    def __init__(self, name: str, audio: bytes) -> None:
        self._name = name
        self._audio = audio
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def synthesize(self, text: str) -> bytes:
        self.calls += 1
        return self._audio


@pytest.fixture
def stub_transcriber():
    return StubTranscriber


@pytest.fixture
def stub_synthesizer():
    return StubSynthesizer
