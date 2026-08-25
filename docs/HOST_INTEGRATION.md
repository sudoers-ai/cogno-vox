# Host integration

How a host (e.g. the Cogno SaaS gateway) wires `cogno-vox` onto the two edges of
the cognitive pipeline. `cogno-vox` only converts audio ⇆ text; everything else
(persona, RBAC, model ladder, key rotation, channel delivery) stays in the host.

```
inbound voice note ──▶ [vox STT] ──▶ cogno-anima ─…─▶ reply text ──▶ [vox TTS] ──▶ outbound voice note
                        transcribe                                    synthesize
```

## 1. Configure the chains (host owns tier selection)

The host builds a `VoxConfig` — the library never reads a YAML file or an env
contextvar. Tiers are tried in order; the first non-empty result wins, and the
chain raises only when **every** tier fails.

```python
from cogno_vox import VoxConfig, TierConfig, create_transcriber, create_synthesizer

config = VoxConfig(
    transcribe_tiers=(
        TierConfig(provider="local", model="Systran/faster-whisper-small",
                   base_url="http://localhost:8000"),
        TierConfig(provider="groq", model="whisper-large-v3-turbo",
                   base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY),
    ),
    synthesize_tiers=(
        TierConfig(provider="local", model="kokoro",
                   base_url="http://localhost:8880/v1", voice="af_alloy"),
        TierConfig(provider="openai", model="tts-1", voice="alloy",
                   base_url="https://api.openai.com/v1", api_key=OPENAI_KEY),
    ),
    default_tts_format="opus",   # mandatory for Telegram native voice notes
)

transcriber = create_transcriber(config)   # build once, reuse
synthesizer = create_synthesizer(config)
```

Per-tenant routing / model ladders / BYOK: the host composes a `VoxConfig` per
request (or caches per tenant). The library is stateless.

## 2. STT — inbound audio → text

```python
result = await transcriber.transcribe(audio_bytes, filename="voice.ogg")
ctx_text = result.text        # feed this into cogno-anima NOUMENO
#   result.tier      -> "local:Systran/faster-whisper-small" (which tier won)
#   result.elapsed_ms, result.chars  -> fold into your telemetry
```

`TranscriptionError` is raised for empty audio, an unsupported container, or all
tiers failing. Supported input containers: `ogg, mp3, wav, m4a, webm, flac, opus`.

> **Cold start:** the first call to a fresh `faster-whisper-server` downloads the
> model and can exceed the default 60s tier timeout. Bump `TierConfig.timeout`
> (the integration suite uses 180) or pre-warm the server.

## 3. TTS — reply text → audio

Slice long replies into ~30s voice notes, synthesize each, hand the bytes to the
channel:

```python
from cogno_vox import split_text_for_tts

for segment in split_text_for_tts(reply_text):
    out = await synthesizer.synthesize(segment)
    await channel.send_voice(out.audio, mime=out.fmt)   # host delivery
```

`out.fmt` is the container/codec (`opus` by default). `SynthesisError` is raised
only when every tier fails.

## What stays in the host

| Concern | Owner |
| --- | --- |
| Provider/model selection, per-tenant routing | host (builds `VoxConfig`) |
| RBAC premium gating (e.g. a `voice_response` skill) | host |
| BYOK & API-key rotation | host |
| Channel delivery (`sendVoice`/`sendMedia`) + ordering/delay | host |
| Channel-mandated format choice | host (lib offers `opus`/`mp3`) |
| Audio ⇆ text conversion + provider fallback | **cogno-vox** |

## Local servers

`docker-compose.yml` brings up the reference STT/TTS servers used by the live
integration suite:

```bash
docker compose up -d           # whisper :8000, kokoro :8880
```

See the README for the gated `pytest tests/integration` invocation.

## Delivery profile — what the host must decide

`cogno-vox` translates a `DeliveryProfile` into what each engine can be told. **Producing** it is
the host's job, because the inputs are all host-side: the persona's configured traits (after any
per-turn modulation), the contact's emotional state, and the turn's own outcome.

```python
from cogno_vox import DeliveryProfile, sanitize_delivery

profile, dropped = sanitize_delivery({"style": style, "pace": pace, "energy": energy})
if dropped:
    log_once(f"delivery axes ignored: {dropped}")      # once per CONFIG, not per turn
result = await synthesizer.synthesize(reply, emotion=cue, delivery=profile or None)
```

Two rules worth carrying across the boundary:

* **the same source for text and voice.** If the host already suppressed a trait for this turn
  (humour on a somber message, detail on an escalation), the delivery profile must come from the
  *modulated* traits, not the declared ones — otherwise the reply reads sober and sounds cheerful.
* **the cue blocklist governs differently here, and the difference is the point.** For an
  `emotion` it is absolute: a laugh at an upset contact reads as mockery and has no gentler
  version, so PII / frustration / urgency / a handoff must produce no cue at all. A delivery
  DOES have a gentler version — and the upset contact is exactly who should be read to slowly
  and gently. Muting the profile there spends the feature everywhere except where it earns its
  keep.

  So let **the somber ones** — PII, frustration, a handoff — **steer** the shape instead of
  switching it off: down in energy, slower, never warm-and-bright. The rule survives — nothing
  should make a PII turn sound cheerful — but it belongs in the mapping, not in an early return.

  **Urgency is the member that goes the other way**, and carrying it in this list was an
  over-reach a review caught: a pressed contact is not calmed by a slow read, they are kept
  waiting. It steers `pace` to `fast` and leaves warmth alone. (Both halves correct an earlier
  version of this guide, which said "→ plain voice" for the whole list.)

An engine that cannot shape delivery ignores the profile. That is by design and never an error —
do not treat an unshaped reply as a failure.

You **do** have to declare which engines can be shaped, on the tier itself:

```python
TierConfig(provider="openai", model="gpt-4o-mini-tts", delivery_dialect="instructions")
TierConfig(provider="elevenlabs", model="...",         delivery_dialect="voice_settings")
TierConfig(provider="local",  model="kokoro")          # "" — cannot be shaped, and is not asked
```

It is not inferable from the adapter: `OpenAICompatSynthesizer` drives OpenAI, Kokoro, Dia and
Orpheus over one HTTP shape and only the first honours `instructions`. Leaving it unset is the
safe default — the tier simply speaks unshaped.
