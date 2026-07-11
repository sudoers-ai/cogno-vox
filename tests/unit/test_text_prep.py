"""Unit tests for TTS text normalisation (cogno_vox.text_prep.clean_text_for_tts)."""

from __future__ import annotations

import pytest

from cogno_vox import clean_text_for_tts, split_text_for_tts


def test_strips_emoji_from_a_real_secretary_reply():
    # the exact shape that was being read out loud (emoji verbalised as words)
    reply = "Prontinho, Vinicius! ✅ Sua consulta ficou marcada para 20/07 às 10h. 📅 😊"
    out = clean_text_for_tts(reply)
    assert "✅" not in out and "📅" not in out and "😊" not in out
    assert out == "Prontinho, Vinicius! Sua consulta ficou marcada para 20/07 às 10h."


@pytest.mark.parametrize("emoji", ["😊", "👋", "📅", "✅", "❤️", "🙏", "🏴‍☠️", "1️⃣", "🇧🇷", "⏰", "⭐", "➡️"])
def test_common_emoji_are_removed(emoji):
    out = clean_text_for_tts(f"bom dia {emoji} tudo bem")
    assert emoji.replace("️", "") not in out
    assert "bom dia" in out and "tudo bem" in out


def test_keeps_words_accents_and_prosody_punctuation():
    src = "Olá! Tudo bem? São 15h — não se preocupe: está tudo certo… ok."
    out = clean_text_for_tts(src)
    # accents + the punctuation that shapes speech survive intact
    assert out == "Olá! Tudo bem? São 15h — não se preocupe: está tudo certo… ok."


def test_strips_markdown_emphasis_and_code():
    out = clean_text_for_tts("isso é **muito** importante e `código` aqui ~~riscado~~")
    assert "*" not in out and "`" not in out and "~" not in out
    assert "muito" in out and "código" in out and "riscado" in out


def test_markdown_link_reduced_to_label():
    out = clean_text_for_tts("veja [nosso site](https://exemplo.com/x?y=1) agora")
    assert "http" not in out and "exemplo.com" not in out
    assert "nosso site" in out


def test_strips_heading_blockquote_and_bullets():
    src = "# Título\n> citação\n- primeiro item\n- segundo item"
    out = clean_text_for_tts(src)
    assert "#" not in out and out.lstrip()[0] != ">"
    assert "Título" in out and "citação" in out
    assert "primeiro item" in out and "segundo item" in out
    assert not out.lstrip().startswith("-")


def test_idempotent():
    src = "Tudo certo! 😊 veja **isto**"
    once = clean_text_for_tts(src)
    assert clean_text_for_tts(once) == once


def test_empty_and_emoji_only():
    assert clean_text_for_tts("") == ""
    assert clean_text_for_tts("   ") == ""
    assert clean_text_for_tts("😊🎉👍") == ""       # nothing to say


def test_chunker_cleans_before_splitting():
    # a caller that chunks first (bypassing FallbackSynthesizer) still gets clean segments
    text = "Oi! 😊 " + "palavra " * 80 + "fim ✅"
    segs = split_text_for_tts(text)
    joined = " ".join(segs)
    assert "😊" not in joined and "✅" not in joined
    assert segs and all(s.strip() for s in segs)


@pytest.mark.asyncio
async def test_fallback_synthesizer_speaks_cleaned_text():
    # the canonical path: the backend must receive text WITHOUT emoji, and chars must reflect it
    from cogno_vox import FallbackSynthesizer

    seen = {}

    class _Spy:
        name = "spy"
        fmt = "opus"

        async def synthesize(self, text: str) -> bytes:
            seen["text"] = text
            return b"AUDIO"

    r = await FallbackSynthesizer([_Spy()]).synthesize("Oi 😊, marcado ✅ para 10h!")
    assert "😊" not in seen["text"] and "✅" not in seen["text"]
    assert seen["text"] == "Oi, marcado para 10h!"
    assert r.chars == len(seen["text"])       # metered on what is spoken, not the raw emoji text
