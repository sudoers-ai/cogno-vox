"""
cogno_vox.audio_utils — audio container helpers.

Currently: raw PCM → Opus-in-Ogg via an ffmpeg subprocess. Gemini TTS returns
Linear16 PCM @ 24kHz; Telegram (and most voice-note channels) expect Opus.

ffmpeg is a *soft* dependency: if it is not on PATH the function degrades by
returning the raw PCM unchanged (the caller/channel may still accept it, or the
fallback chain moves on) rather than raising.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

log = logging.getLogger(__name__)


def ffmpeg_available() -> bool:
    """True when the ``ffmpeg`` binary is on PATH. Used for a boot-time check: the local Kokoro
    tier re-encodes opus via ffmpeg (``wav_to_opus``), and its degrade is STRICT (``b""``), so a
    missing ffmpeg silently kills all voice output — worth surfacing at startup, not per turn."""
    return shutil.which("ffmpeg") is not None


def pcm_to_opus(pcm: bytes, sample_rate: int = 24000) -> bytes:
    """Convert raw signed-16-bit mono PCM to Opus-in-Ogg via ffmpeg.

    Returns the original ``pcm`` unchanged if ffmpeg is unavailable or fails.
    """
    if not pcm:
        return b""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "s16le",           # signed 16-bit little-endian
                "-ar", str(sample_rate),
                "-ac", "1",              # mono
                "-i", "pipe:0",
                "-c:a", "libopus",
                "-b:a", "32k",
                "-f", "ogg",
                "pipe:1",
            ],
            input=pcm,
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
        log.warning("ffmpeg PCM->Opus failed: rc=%d stderr=%s",
                    result.returncode, result.stderr[:200])
        return pcm
    except FileNotFoundError:
        log.warning("ffmpeg not found — returning raw PCM")
        return pcm
    except Exception as exc:
        log.warning("PCM->Opus conversion error: %s", exc)
        return pcm


def wav_to_opus(audio: bytes) -> bytes:
    """Convert a container-carrying audio blob (WAV/anything ffmpeg sniffs) to Opus-in-Ogg.

    Companion to :func:`pcm_to_opus` for sources that return a full container instead of raw
    PCM — e.g. requesting ``wav`` from a Kokoro server (whose own ``opus`` encoder truncates
    the tail of the audio) and encoding properly here.

    Degrade contract is STRICT (unlike ``pcm_to_opus``): on ffmpeg failure it returns ``b""``.
    The caller promised opus (``fmt`` is reported to the channel, which labels the voice note
    ``audio/ogg``) — handing back WAV under that label breaks playback outright, whereas an
    empty result makes the tier fail over to the next engine (or degrade to a text reply).
    """
    if not audio:
        return b""
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", "pipe:0",
             "-c:a", "libopus", "-b:a", "32k", "-f", "ogg", "pipe:1"],
            input=audio,
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
        log.warning("ffmpeg WAV->Opus failed: rc=%d stderr=%s",
                    result.returncode, result.stderr[:200])
        return b""
    except FileNotFoundError:
        log.warning("ffmpeg not found — WAV->Opus unavailable, failing the tier over")
        return b""
    except Exception as exc:
        log.warning("WAV->Opus conversion error: %s", exc)
        return b""
