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


def test_wav_to_opus_fails_closed_without_ffmpeg(monkeypatch):
    # STRICT degrade: the caller reports fmt="opus" to the channel — WAV under that label
    # breaks playback, so a failed encode must return b"" (tier fails over), never the input.
    import subprocess as sp

    from cogno_vox import audio_utils

    def _boom(*a, **k):
        raise FileNotFoundError("ffmpeg")
    monkeypatch.setattr(sp, "run", _boom)
    assert audio_utils.wav_to_opus(b"RIFFxxxx") == b""

    def _rc1(*a, **k):
        class R:
            returncode, stdout, stderr = 1, b"", b"err"
        return R()
    monkeypatch.setattr(sp, "run", _rc1)
    assert audio_utils.wav_to_opus(b"RIFFxxxx") == b""
