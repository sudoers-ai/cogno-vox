"""
cogno_vox.chunker — split text into TTS-sized segments (pure word math).

Ported from the parent ``cogno.gateways.chunker.split_text_for_tts``. Kept in
the library because it is pure text/word arithmetic with no I/O — useful right
before synthesis so each voice note stays ~30s. The channel-specific *delivery*
(separate voice notes, inter-message delay) stays in the host.
"""

from __future__ import annotations

import re

# ~130 words/min in pt-BR → ~65 words ≈ 30 seconds of speech.
TTS_MAX_WORDS_PER_SEGMENT = 65
TTS_MAX_SEGMENTS = 10  # safety cap


def split_text_for_tts(
    text: str,
    max_words: int = TTS_MAX_WORDS_PER_SEGMENT,
    max_segments: int = TTS_MAX_SEGMENTS,
) -> list[str]:
    """Split ``text`` into segments of ``≤ max_words`` words each.

    Splits at sentence boundaries to avoid cutting mid-sentence; a single
    over-long sentence is hard-split at word boundaries. Overflow beyond
    ``max_segments`` is merged into the last segment.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    words = text.split()
    if len(words) <= max_words:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)

    segments: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sw = sentence.split()

        if len(sw) > max_words:
            if current:
                segments.append(" ".join(current))
                current = []
            for i in range(0, len(sw), max_words):
                segments.append(" ".join(sw[i:i + max_words]))
            continue

        if len(current) + len(sw) > max_words:
            if current:
                segments.append(" ".join(current))
            current = sw[:]
        else:
            current.extend(sw)

    if current:
        segments.append(" ".join(current))

    if len(segments) > max_segments:
        kept = segments[:max_segments - 1]
        kept.append(" ".join(segments[max_segments - 1:]))
        segments = kept

    return segments
