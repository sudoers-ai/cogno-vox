"""pcm_to_opus degradation behaviour (no ffmpeg dependency in CI)."""

from __future__ import annotations

import cogno_vox.audio_utils as au
from cogno_vox.audio_utils import pcm_to_opus


def test_pcm_to_opus_empty():
    assert pcm_to_opus(b"") == b""


def test_pcm_to_opus_degrades_without_ffmpeg(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(au.subprocess, "run", boom)
    pcm = b"\x00\x01\x02\x03"
    # Falls back to returning the raw PCM unchanged instead of raising.
    assert pcm_to_opus(pcm) == pcm


def test_pcm_to_opus_returns_ffmpeg_stdout(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = b"OGGOPUS"
        stderr = b""

    monkeypatch.setattr(au.subprocess, "run", lambda *a, **k: FakeProc())
    assert pcm_to_opus(b"\x00\x01") == b"OGGOPUS"


def test_pcm_to_opus_degrades_on_nonzero_exit(monkeypatch):
    class FakeProc:
        returncode = 1
        stdout = b""
        stderr = b"bad input"

    monkeypatch.setattr(au.subprocess, "run", lambda *a, **k: FakeProc())
    pcm = b"\x00\x01"
    assert pcm_to_opus(pcm) == pcm
