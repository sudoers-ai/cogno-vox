"""
Minimal host wiring for cogno-vox: audio in -> text, text -> audio out.

This mirrors how a host (e.g. cogno's Telegram/WhatsApp gateway) would sit
cogno-vox on the two edges of the cognitive pipeline:

    audio bytes --[vox STT]--> text --> cogno-anima ... --> reply --[vox TTS]--> audio

Requires a reachable local Whisper + Kokoro server (or swap the tiers for cloud
ones with API keys). Run:  python examples/host_min.py path/to/voice.ogg
"""

from __future__ import annotations

import asyncio
import sys

from cogno_vox import (
    TierConfig,
    VoxConfig,
    create_synthesizer,
    create_transcriber,
    split_text_for_tts,
)

config = VoxConfig(
    transcribe_tiers=(
        # Tier 1: local faster-whisper-server; Tier 2: Groq cloud (needs key).
        TierConfig(provider="local", model="Systran/faster-whisper-small",
                   base_url="http://localhost:8000"),
        # TierConfig(provider="groq", model="whisper-large-v3-turbo",
        #            base_url="https://api.groq.com/openai/v1", api_key="gsk_..."),
    ),
    synthesize_tiers=(
        # Tier 1: local Kokoro; Tier 2: OpenAI cloud (needs key).
        TierConfig(provider="local", model="kokoro",
                   base_url="http://localhost:8880/v1", voice="af_alloy"),
        # TierConfig(provider="openai", model="tts-1", voice="alloy",
        #            base_url="https://api.openai.com/v1", api_key="sk-..."),
    ),
    default_tts_format="opus",
)


async def main(audio_path: str) -> None:
    transcriber = create_transcriber(config)
    synthesizer = create_synthesizer(config)

    with open(audio_path, "rb") as fh:
        audio_bytes = fh.read()

    # 1) STT: audio -> text (this would feed cogno-anima NOUMENO).
    stt = await transcriber.transcribe(audio_bytes, filename=audio_path)
    print(f"[STT {stt.tier} {stt.elapsed_ms:.0f}ms] {stt.text!r}")

    # ... cogno-anima pipeline runs here, producing a reply ...
    reply = f"Você disse: {stt.text}"

    # 2) TTS: reply -> audio, one voice note per ~30s segment.
    for i, segment in enumerate(split_text_for_tts(reply)):
        out = await synthesizer.synthesize(segment)
        path = f"reply_{i}.{out.fmt if out.fmt != 'opus' else 'ogg'}"
        with open(path, "wb") as fh:
            fh.write(out.audio)
        print(f"[TTS {out.tier} {out.elapsed_ms:.0f}ms] -> {path} ({out.nbytes} bytes)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python examples/host_min.py path/to/voice.ogg")
        raise SystemExit(1)
    asyncio.run(main(sys.argv[1]))
