# cogno-vox

**Voice/audio I/O edge for the [Cogno](https://github.com/sudoers-ai/cogno-anima) cognitive pipeline** — speech-to-text in, text-to-speech out, behind provider-agnostic fallback chains.

`cogno-vox` is the *mouth and ears* of Cogno. Where [`cogno-anima`](https://github.com/sudoers-ai/cogno-anima) is the *mind* (pure cognition) and [`cogno-engram`](https://github.com/sudoers-ai/cogno-engram) is the *memory* (persistence substrate), `cogno-vox` is the **transport edge** that sits on the two boundaries of the pipeline:

```
gateway ──▶ [vox STT] ──▶ cogno-anima (NOUMENO…SUPEREGO) ──▶ [vox TTS] ──▶ gateway
             audio→text                                        text→audio
```

It is the most decoupled of the libraries: the core imports **neither** `cogno-anima` **nor** `cogno-engram` — it only converts audio ⇆ text.

> Status: **alpha** — core (ports + OpenAI-compatible backends + fallback chains) and the unit suite are in place.

## Three capability-scoped ports

Like `cogno-anima`'s `LLMBackend`/`Embedder` and `cogno-engram`'s three ports, each direction is a structurally-typed `Protocol` with its own fallback chain (try tier 1→N, first non-empty wins, all-fail → raises):

| Port | Async method | Reference backends |
| --- | --- | --- |
| `TranscriberBackend` (STT) | `transcribe(audio, filename) -> str` | OpenAI-compatible (faster-whisper-server / Groq / OpenAI), Gemini multimodal, Bedrock/Voxtral |
| `SynthesizerBackend` (TTS) | `synthesize(text) -> bytes` | OpenAI-compatible (Kokoro / OpenAI), Grok, ElevenLabs, Gemini |
| `VisionAnalyzerPort` (Vision) | `analyze(media_bytes, mime_type) -> VisionAnalysisResult` | Qwen2.5-VL (local/vLLM), Moondream2, GOT-OCR2.0, Gemini / GPT-4o Vision |
| `DeliveryAwareBackend` (TTS, **optional**) | `synthesize_shaped(text, delivery) -> bytes` | OpenAI-compatible (`instructions`), ElevenLabs (`voice_settings`) |

`OpenAICompat*` covers any `/v1/audio/transcriptions` or `/v1/audio/speech` server — local or cloud differ only by `base_url`/`api_key`, no SDK needed (just `httpx`).

## Two ways to shape a voice

Both are engine-agnostic in, engine-specific out, and both are **preferences** — an engine that cannot honour one speaks the same words and the call succeeds.

| | What it is | How it travels |
| --- | --- | --- |
| **emotion** | one discrete non-verbal cue — `chuckle`, `sigh` | an inline tag in the tier's dialect (`(chuckle)` for Dia, `<chuckle>` for Orpheus) |
| **delivery** | how the WHOLE utterance is said — three independent axes | engine config: `instructions` prose, or `voice_settings` numbers |

```python
from cogno_vox import DeliveryProfile

await synthesizer.synthesize(
    "Bom dia, tudo bem?",
    emotion="chuckle",                                   # a cue, if the tier has a dialect
    delivery=DeliveryProfile(style="warm", pace="slow"),  # a shape, if the tier can be shaped
)
```

`style` (`warm|reserved|empathetic`), `pace` (`fast|steady|slow`), `energy` (`high|normal|low`) — every axis optional, `""` meaning "engine default". Not every engine has a lever for every axis: on ElevenLabs `pace` maps to `stability` as the closest approximation (stated, not hidden), and `energy` maps to **nothing** — the obvious candidate, `similarity_boost`, is voice *adherence*, so `high` would hold the read closer to its reference, the opposite of what was asked. Between a wrong mapping and none, none. Three flat axes rather than one mood label because the host derives them from separate signals and collapsing them would force it to pick a winner where there is none.

`sanitize_delivery(raw) -> (profile, dropped)` is pure and total: anything in, a valid profile out, never an exception. An unknown value is **dropped, never guessed** — correcting `friendly` to `warm` would put a tone on a voice nobody asked for, while dropping falls back to the engine default. Log `dropped` once per configuration, not per turn.

**Who can be shaped takes two answers, and both are required.** The *adapter* must be able to carry a profile — `DeliveryAwareBackend`, a second protocol like `ToolCallingBackend` beside `LLMBackend` in `cogno-anima` — **and** the *engine* must declare a dialect, `TierConfig(delivery_dialect="instructions"|"voice_settings")`.

The second is not inferable, and a review measured the hole: `OpenAICompatSynthesizer` drives OpenAI, Kokoro, Dia **and** Orpheus over the same HTTP shape, so asking the class alone answered "which transport is this" — a different question wearing the right question's clothes — and `instructions` was being sent to two tag engines whose own docs ignore it. Declared per tier, exactly like `emotion_dialect`.

Both are checked per tier, so a chain can mix a shaping engine with a plain one and a failover degrades the *delivery* instead of the call. A profile that sets no axis takes the byte-identical path the code took before the feature existed.

`sanitize_delivery` is also the guard the **renderers** use, not just the entry point: a plain `dict` reaching `as_instructions` used to raise `AttributeError` from inside the backend's call and outside every `return b""` — which lost a voice note on a single-tier chain and recorded a circuit-breaker fault against a *healthy* engine on a mixed one.

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

Deciding **what** the delivery should be is host-side too: `cogno-vox` owns only the translation into what a given engine can be told. The persona's traits, the contact's state and the turn's outcome are the host's to read — see `DeliveryProfile` above for the vocabulary it must produce.

## The Cogno ecosystem

`cogno-vox` is one organ of **[Cogno](https://github.com/sudoers-ai)** — a family of
small, composable, Apache-2.0 libraries that together form a complete
conversational-agent platform. Each library owns a single concern and stays
infra-agnostic; a **host** assembles them into a running agent:

![The Cogno ecosystem](docs/assets/cogno-ecosystem.svg)

The open-source libraries are the organs; the **host is the body** that joins
them. Our reference host — `cogno-host`, with its `cogno-ui` dashboard — is the
private product layer, but it holds no special powers: everything it does rides
on the public seams documented in each library's `docs/HOST_INTEGRATION.md`, so
you can assemble a body of your own.

## Develop

```bash
pip install -e ".[dev]"
pytest tests/unit -q          # mocked HTTP, no network
ruff check cogno_vox && mypy cogno_vox
```

### Does shaping cost intelligibility?

A delivery profile changes HOW, never WHAT — and the one thing that can go wrong unnoticed is an engine honouring it by whispering or rushing. The round-trip bench measures exactly that:

```bash
python3 voxbench.py --delivery "style=warm,pace=slow"     # WER must not regress
```

The report says whether the engine **APPLIED** or **IGNORED** the profile, because "the WER did not move" means two different things and only one of them is a result.

### Live integration tests (gated, auto-skip)

`tests/integration` runs end-to-end against real local servers and **auto-skips**
unless pointed at them (same pattern as `cogno-engram`'s `ENGRAM_TEST_DSN`).
Validated against `fedirz/faster-whisper-server` (`Systran/faster-whisper-small`)
and `ghcr.io/remsky/kokoro-fastapi-cpu`. The bundled `docker-compose.yml` brings
both up:

```bash
docker compose up -d           # whisper :8000, kokoro :8880

VOX_TEST_WHISPER_URL=http://localhost:8000 \
VOX_TEST_KOKORO_URL=http://localhost:8880/v1 \
pytest tests/integration -q   # synth (Kokoro) -> transcribe (Whisper) round-trip
```

See [`docs/HOST_INTEGRATION.md`](docs/HOST_INTEGRATION.md) for wiring `cogno-vox`
onto the two edges of the pipeline.

## License

Apache-2.0.
