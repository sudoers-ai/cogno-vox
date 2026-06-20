"""
Word Error Rate (WER) — pure, dependency-free.

WER = (substitutions + insertions + deletions) / reference_word_count, computed by
word-level Levenshtein. Text is normalized first (lowercase, strip punctuation,
collapse whitespace) so scoring is about words, not formatting. Returned WER is a
ratio in ``[0, 1+]`` (it can exceed 1 when the hypothesis is much longer).
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> List[str]:
    """Lowercase, drop punctuation/diacritics, collapse whitespace → word tokens."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT.sub(" ", text.lower())
    text = _WS.sub(" ", text).strip()
    return text.split(" ") if text else []


def _levenshtein(ref: List[str], hyp: List[str]) -> int:
    """Edit distance over word tokens (substitution/insertion/deletion = 1)."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        curr = [i]
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate of ``hypothesis`` against ``reference`` (0.0 = perfect).

    Two empty strings score 0.0; an empty reference with a non-empty hypothesis
    scores 1.0 (all insertions, normalized by 1 to avoid divide-by-zero).
    """
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)
