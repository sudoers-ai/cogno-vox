# cogno-vox

**Voice/audio I/O edge for the [Cogno](https://github.com/sudoers-ai/cogno-anima) cognitive pipeline** — speech-to-text in, text-to-speech out, behind provider-agnostic fallback chains.

`cogno-vox` is the *mouth and ears* of Cogno. Where [`cogno-anima`](https://github.com/sudoers-ai/cogno-anima) is the *mind* (pure cognition) and [`cogno-engram`](https://github.com/sudoers-ai/cogno-engram) is the *body* (memory/persistence), `cogno-vox` is the **transport edge** that sits on the two boundaries of the pipeline:

```
gateway ──▶ [vox STT] ──▶ cogno-anima (NOUMENO…SUPEREGO) ──▶ [vox TTS] ──▶ gateway
             audio→text                                        text→audio
```

It is the most decoupled of the libraries: the core imports **neither** `cogno-anima` **nor** `cogno-engram` — it only converts audio ⇆ text.

> Status: **alpha** — core (ports + OpenAI-compatible backends + fallback chains) and the unit suite are in place.

## Two capability-scoped ports

Like `cogno-anima`'s `LLMBackend`/`Embedder` and `cogno-engram`'s three ports, each direction is a structurally-typed `Protocol` with its own fallback chain (try tier 1→N, first non-empty wins, all-fail → raises):

| Port | Async method | Reference backends |
| --- | --- | --- |
| `TranscriberBackend` (STT) | `transcribe(audio, filename) -> str` | OpenAI-compatible (faster-whisper-server / Groq / OpenAI), Gemini multimodal, Bedrock/Voxtral |
| `SynthesizerBackend` (TTS) | `synthesize(text) -> bytes` | OpenAI-compatible (Kokoro / OpenAI), Grok, ElevenLabs, Gemini |

`OpenAICompat*` covers any `/v1/audio/transcriptions` or `/v1/audio/speech` server — local or cloud differ only by `base_url`/`api_key`, no SDK needed (just `httpx`).

## Async by design

Everything is `async` (`httpx.AsyncClient`) to run under concurrent request load alongside `cogno-anima`/`cogno-engram`. Provider SDKs are lazy-imported behind extras (`pip install "cogno-vox[bedrock]"`); the OpenAI-compatible HTTP path needs no extra.

## Host-injected config (no YAML, no env contextvar)

The host owns tier selection (model ladder, per-tenant routing, BYOK). You hand the factory a `VoxConfig`:

```python
from cogno_vox import VoxConfig, TierConfig, create_transcriber, create_synthesizer

config = VoxConfig(
    transcribe_tiers=(
        TierConfig(provider="local", model="Systran/faster-whisper-small",
                   base_url="http://localhost:8000"),
        TierConfig(provider="groq", model="whisper-large-v3-turbo",
                   base_url="https://api.groq.com/openai/v1", api_key="gsk_..."),
    ),
    synthesize_tiers=(
        TierConfig(provider="local", model="kokoro",
                   base_url="http://localhost:8880/v1", voice="af_alloy"),
        TierConfig(provider="openai", model="tts-1", voice="alloy",
                   base_url="https://api.openai.com/v1", api_key="sk-..."),
    ),
    default_tts_format="opus",   # mandatory for Telegram native voice notes
)

stt = await create_transcriber(config).transcribe(audio_bytes, "voice.ogg")
print(stt.text, stt.tier, stt.elapsed_ms)

out = await create_synthesizer(config).synthesize("Olá, tudo bem?")
# out.audio (bytes), out.fmt, out.tier, out.elapsed_ms
```

`split_text_for_tts()` slices long replies into ~30s segments (≈65 words) at sentence boundaries — one voice note each.

## What stays in the host

Provider/model selection, RBAC premium gating, BYOK & key rotation, the channel delivery (`sendVoice`/`sendMedia`) and the channel-mandated format choice. `cogno-vox` offers `opus`/`mp3`; the host decides.

## Develop

```bash
pip install -e ".[dev]"
pytest tests/unit -q          # mocked HTTP, no network
ruff check cogno_vox && mypy cogno_vox
```

### Live integration tests (gated, auto-skip)

`tests/integration` runs end-to-end against real local servers and **auto-skips**
unless pointed at them (same pattern as `cogno-engram`'s `ENGRAM_TEST_DSN`).
Validated against `fedirz/faster-whisper-server` (`Systran/faster-whisper-small`)
and `ghcr.io/remsky/kokoro-fastapi-cpu`:

```bash
docker run -d -p 127.0.0.1:8000:8000 fedirz/faster-whisper-server:latest-cpu
docker run -d -p 127.0.0.1:8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest

VOX_TEST_WHISPER_URL=http://localhost:8000 \
VOX_TEST_KOKORO_URL=http://localhost:8880/v1 \
pytest tests/integration -q   # synth (Kokoro) -> transcribe (Whisper) round-trip
```

## License

Apache-2.0.
