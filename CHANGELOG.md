# Changelog

## Unreleased

### Added

- **`PdfTextExtractor`** (`cogno_vox.pdf_text`) — the TEXT LAYER of an untrusted PDF, read in a
  sandboxed worker (F2.4). Satisfies `cogno_engram.documents.TextExtractor` structurally (no
  import either way): bytes + `max_bytes`/`max_pages`/`timeout_s` in, `pages[].number/.text` and
  `outline[].level/.title/.page` out, or `PdfExtractionError.reason` in
  `no_text`/`over_limit`/`encrypted`/`invalid`/`timeout`.
  - byte ceiling and header checked BEFORE a worker exists; page count refused over
    `max_pages` BEFORE any page is read;
  - worker = separate process (`python -E -c`, empty cwd, minimal env) killed at `timeout_s`,
    with its own `RLIMIT_CPU`; inside, every descriptor above stdio is closed and
    `RLIMIT_NOFILE` = 3 (no socket, no file — at the kernel level, measured by `selfcheck()`),
    `RLIMIT_AS`, `RLIMIT_FSIZE` 0, `RLIMIT_NPROC` 0; lazily-loaded paths warmed up on a
    document of our own first (measured: the first `get_text_range` imports a codec, which
    opens a file);
  - text layer only: forms never initialised (no JavaScript), no attachments, no links, no
    rendering.
- **Extra `pdf`** = `pypdfium2` alone (no opencv/numpy). CI installs `.[vision,pdf]`.

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
