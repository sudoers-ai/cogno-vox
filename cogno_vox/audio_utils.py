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
import subprocess

log = logging.getLogger(__name__)


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
