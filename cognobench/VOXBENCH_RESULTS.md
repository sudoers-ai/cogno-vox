# Voice round-trip bench — results

`python3 voxbench.py` measures **round-trip Word Error Rate**: synthesize a known
sentence (TTS) → transcribe it back (STT) → score WER. Run `--stub` for the
deterministic doubles (no network) or the default for real OpenAI-compatible
STT/TTS backends (configured via env).

## What this does and does NOT measure

cogno-vox is **transport** — it forwards audio to Whisper/Kokoro/ElevenLabs/etc.
The recognition quality is the **provider's**, not the library's. So, unlike the
cogno-core/engram cognobenches (which benchmark cognition the lib *owns*), this is
**not** an "is the library smart" score. It is a repeatable harness to:

- **regression-guard** the end-to-end audio path through vox's own code (synthesize
  → bytes → transcribe), catching wiring/format/encoding breakage;
- **compare STT/TTS backends** wired through vox (swap env config, re-run, compare
  mean WER) without versioning any audio blobs — the audio is generated at runtime.

## Baselines

| Backends | mean WER | Notes |
|---|---|---|
| stub (lossless text↔bytes) | **0.000** | plumbing guard; a unit smoke asserts it |
| real (local faster-whisper + Kokoro, or cloud) | _host-measured_ | run `python3 voxbench.py` with `VOX_TTS_*` / `VOX_STT_*` set |

The stub round-trip is lossless by construction (encode → decode), so WER is 0 and
proves the harness, not any model. Real numbers depend entirely on the chosen
backends — record yours here when you wire a server.

## Running against real backends

```bash
export VOX_TTS_BASE_URL=http://localhost:8880  VOX_TTS_MODEL=kokoro
export VOX_STT_BASE_URL=http://localhost:8000  VOX_STT_MODEL=Systran/faster-whisper-small
python3 voxbench.py                       # prints per-case + mean WER
python3 voxbench.py --max-wer 0.40        # CI gate
```

The integration test (`tests/integration/test_voxbench_live.py`) runs the same
round-trip and gates mean WER at a lenient **0.40** ceiling (gross-regression /
misconfiguration guard, not a tight quality bar) — auto-skipping when no STT/TTS
servers are configured.
