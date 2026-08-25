# Changelog

## Unreleased

### Added

- **Delivery profile** — `DeliveryProfile(style, pace, energy)`, an engine-agnostic description
  of HOW an utterance is said, distinct from the existing `emotion` cue (one discrete tag).
  `synthesize(..., delivery=)` carries it; `cogno_vox.delivery` renders it per engine family
  (`instructions` prose for OpenAI-compatible, `voice_settings` numbers for ElevenLabs).
- `DeliveryAwareBackend` — an **optional** second protocol — plus `TierConfig.delivery_dialect`,
  the engine's own declaration. Both are required and both are checked per tier: one adapter
  class drives OpenAI, Kokoro, Dia and Orpheus, so the class alone cannot say who honours a
  profile. An engine that cannot shape delivery speaks the same words and the call succeeds; a
  failover from a shaping tier to a plain one degrades the delivery, never the call.
- `sanitize_delivery(raw) -> (profile, dropped)` — pure and total (a dict with mixed key types,
  a value whose `__str__` raises, a bare string: all absorbed); an unknown axis value is dropped
  rather than guessed, and `dropped` names what to log once per configuration. The renderers
  coerce through it too, so a plain `dict` can never escape as an exception from inside a
  backend call.
- `voxbench.py --delivery "style=warm,pace=slow"` — measures whether shaping costs
  intelligibility, and reports **APPLIED** vs **IGNORED** so an unmoved WER cannot be read as
  a result when nothing was applied.

A caller that passes no profile takes the byte-identical path as before.


## 0.1.0 — 2026-07-25

First public release on PyPI.

Voice/audio I/O edge for the Cogno cognitive pipeline — speech-to-text (STT) in, text-to-speech (TTS) out, with provider-agnostic fallback chains
