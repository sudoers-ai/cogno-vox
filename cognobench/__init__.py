"""
cognobench — voice round-trip quality benchmark for cogno-vox.

**Honest framing:** cogno-vox is *transport* (it forwards audio to Whisper/Kokoro/
ElevenLabs/…). The transcription quality is the *provider's*, not the library's —
so unlike cogno-core/engram (which own their cognition), this is not an "is the lib
smart" bench. What it measures, and where vox *is* responsible, is the **end-to-end
audio path through vox's own code**: synthesize a known sentence → transcribe it
back → score Word Error Rate (round-trip WER). That makes it a repeatable tool to
**compare/regress STT+TTS backends** wired through vox, without versioning audio
blobs (the audio is generated at runtime).

Two modes, like the sibling benches:

* ``--stub`` — a lossless text↔bytes round-trip (no network); WER must be 0 (a unit
  smoke asserts the plumbing).
* default — real backends (OpenAI-compatible STT/TTS from env), measuring true
  round-trip WER. Auto-skips when nothing is configured.
"""
