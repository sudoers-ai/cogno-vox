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


# ── emotion tags (expressive TTS) ─────────────────────────────────────────
#
# Some engines voice non-verbal cues from inline tags — each in its OWN dialect:
# Dia (nari-labs) reads parenthetical cues like ``(laughs)``; Orpheus (Canopy)
# reads angle-bracket tags like ``<laugh>``. The pipeline hands emotion around as
# an engine-agnostic HINT (``"chuckle"``); the fallback chain calls
# :func:`apply_emotion` per tier with that tier's declared dialect
# (``TierConfig.emotion_dialect``) so a failover to a plain engine never carries
# a tag it would read out loud ("laughs, parabéns!").

_EMOTION_DIALECTS: dict[str, dict[str, str]] = {
    # Dia — parenthetical cues (the ``[S1]`` speaker prefix is the server's concern, not a tag).
    "dia": {
        "laugh": "(laughs)", "chuckle": "(chuckle)", "sigh": "(sighs)",
        "gasp": "(gasps)", "cough": "(coughs)", "clear_throat": "(clears throat)",
    },
    # Orpheus — angle-bracket tags.
    "orpheus": {
        "laugh": "<laugh>", "chuckle": "<chuckle>", "sigh": "<sigh>",
        "gasp": "<gasp>", "cough": "<cough>", "groan": "<groan>",
        "yawn": "<yawn>", "sniffle": "<sniffle>",
    },
}

# Emotions voiced at the START of the reply (a sigh/gasp precedes the words); the rest
# lands after the first sentence (a laugh reacts to what was just said).
_LEADING_EMOTIONS = frozenset({"sigh", "gasp"})
_SENTENCE_END_RE = re.compile(r"[.!?…]+(?=\s)")

# Every known tag across dialects, longest first so "(clears throat)" wins over any prefix.
_TAG_STRIP_RE = re.compile(
    "|".join(re.escape(t) for d in _EMOTION_DIALECTS.values()
             for t in sorted(d.values(), key=len, reverse=True)),
    re.IGNORECASE,
)


def apply_emotion(text: str, emotion: str, dialect: str) -> str:
    """Decorate ``text`` with ``dialect``'s tag for the engine-agnostic ``emotion`` hint.

    Deterministic placement: leading for breath-like cues (sigh/gasp), after the first
    sentence for reactive ones (laugh/chuckle), appended when there is no boundary.
    Unknown emotion or dialect → ``text`` unchanged (fail-open: a hint must never
    block speech). Run :func:`strip_emotion_tags` first when the text may carry tags."""
    tag = _EMOTION_DIALECTS.get(dialect, {}).get(emotion, "")
    if not tag or not text:
        return text
    if emotion in _LEADING_EMOTIONS:
        return f"{tag} {text}"
    m = _SENTENCE_END_RE.search(text)
    if m:
        return f"{text[:m.end()]} {tag}{text[m.end():]}"
    return f"{text} {tag}"


def strip_emotion_tags(text: str) -> str:
    """Remove every known emotion tag (all dialects) from ``text``.

    Whitelist-based — only the exact tags above are removed, so ordinary parentheses
    ("(11) 99999-1234") survive. Safety net for two paths: a tag the LLM leaked into
    the reply, and a tagged text reaching an engine with no ``emotion_dialect`` (which
    would read "laughs" out loud)."""
    if not text:
        return ""
    out = _TAG_STRIP_RE.sub("", text)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" +([.,!?;:…])", r"\1", out)
    return out.strip()
