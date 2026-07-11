"""
cogno_vox.text_prep — normalise text for speech synthesis (pure, no I/O).

A cognitive reply is written for the *eye*: it carries emoji, markdown emphasis,
bullet dashes and link syntax. Fed verbatim to a TTS engine those are spoken out
loud — "sorriso" for 😊, "asterisco" for **, the raw URL of a link. ``clean_text_for_tts``
strips the non-verbal decoration while keeping the words and the punctuation that
shapes prosody (``. , ! ? ; : … -``), so the voice reads what a human would say.

Run it right before synthesis. It is idempotent and safe on already-clean text.
"""

from __future__ import annotations

import re
import unicodedata

# Emoji / pictographs / dingbats / symbols that a TTS engine would try to verbalise. Deliberately
# scoped to symbol blocks — it must NOT touch letters, digits, or the general-punctuation block
# (that holds normal spaces, en/em dashes and the … ellipsis, which carry prosody).
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"   # regional-indicator flags
    "\U0001F300-\U0001FAFF"   # emoticons, symbols & pictographs, transport, supplemental, ext-A
    "\U0001F000-\U0001F0FF"   # mahjong / dominoes / playing cards
    "\U00002600-\U000027BF"   # misc symbols + dingbats (☀ ✂ ✅ ✈ ❤ ➜ …)
    "\U00002B00-\U00002BFF"   # misc symbols & arrows (⬛ ⭐ …)
    "\U00002300-\U000023FF"   # misc technical (⌚ ⏰ ⏳ ⚙ …)
    "\U000025A0-\U000025FF"   # geometric shapes (▪ ● ◆ …)
    "\U00002190-\U000021FF"   # arrows (↔ ⇒ …)
    "\U0000FE00-\U0000FE0F"   # variation selectors (emoji-style ️)
    "\U0000200D"              # zero-width joiner (glues emoji sequences)
    "\U000020E3"              # combining enclosing keycap (the box in 1️⃣)
    "\U00002122\U00002139"    # ™ ℹ
    "\U0000203C\U00002049"    # ‼ ⁉
    "]+",
    flags=re.UNICODE,
)


def clean_text_for_tts(text: str) -> str:
    """Return ``text`` stripped of emoji and markdown decoration, ready for synthesis.

    Keeps words and sentence punctuation; removes emoji/pictographs, markdown emphasis
    (``* _ ` ~``), heading/blockquote/bullet markers, and reduces a ``[label](url)`` link to
    its label. Collapses the whitespace the removals leave behind. Idempotent."""
    if not text:
        return ""

    # markdown link / image → just the visible label (drop the URL a TTS would read out)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # emoji, pictographs, dingbats, arrows, variation selectors, ZWJ
    text = _EMOJI_RE.sub("", text)
    # heading (#) and blockquote (>) markers at the start of a line
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*>+\s?", "", text)
    # bullet-list markers at the start of a line ("- ", "* ", "• ") → nothing
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    # inline emphasis / code / strike markers (the characters themselves are non-verbal)
    text = re.sub(r"[*`~]+", "", text)
    # any stray symbol left from a partial pictograph sequence
    text = "".join(
        ch for ch in text
        if not (unicodedata.category(ch) == "So" and ord(ch) > 0x2000)
    )
    # collapse the gaps the removals leave
    text = re.sub(r"[ \t]{2,}", " ", text)          # runs of spaces
    text = re.sub(r" +([.,!?;:…])", r"\1", text)    # space before punctuation
    text = re.sub(r"(?m)[ \t]+$", "", text)         # trailing spaces per line
    text = re.sub(r"\n{3,}", "\n\n", text)          # excess blank lines
    return text.strip()
