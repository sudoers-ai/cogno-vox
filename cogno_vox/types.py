"""
cogno_vox.types — config + result models for the voice I/O edge.

Pure data, no I/O. ``VoxConfig`` is what the *host* injects (it does NOT read a
``model_ladder.yaml`` off disk — that is a host concern); the ``*Result`` models
carry the telemetry the host folds into its own accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Audio container formats accepted for transcription (STT input).
SUPPORTED_INPUT_FORMATS = frozenset(
    {"ogg", "mp3", "wav", "m4a", "webm", "flac", "opus"}
)


@dataclass(frozen=True)
class TierConfig:
    """One backend tier in a fallback chain.

    ``provider`` selects the adapter (``openai`` covers any OpenAI-compatible
    HTTP server — faster-whisper-server, Kokoro, OpenAI, Groq — distinguished
    only by ``base_url``). ``extra`` carries provider-specific knobs (e.g. a
    Grok ``sample_rate``) without widening the schema.
    """

    provider: str
    model: str
    base_url: str = ""
    api_key: str = ""
    voice: str = ""              # TTS only
    response_format: str = ""    # TTS only; "" → chain default
    timeout: float = 60.0
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VoxConfig:
    """Host-injected configuration: an ordered list of tiers per direction.

    Tiers are tried in order; the first that returns a non-empty result wins.
    """

    transcribe_tiers: tuple[TierConfig, ...] = ()
    synthesize_tiers: tuple[TierConfig, ...] = ()
    # Default container for synthesized audio (Opus is mandatory for Telegram
    # native voice notes; the host overrides per channel).
    default_tts_format: str = "opus"


@dataclass(frozen=True)
class TranscriptionResult:
    """Outcome of a successful STT call."""

    text: str
    tier: str            # "<provider>:<model>" that produced the text
    elapsed_ms: float
    chars: int = 0

    def __post_init__(self) -> None:
        if not self.chars:
            object.__setattr__(self, "chars", len(self.text))


@dataclass(frozen=True)
class SynthesisResult:
    """Outcome of a successful TTS call."""

    audio: bytes
    fmt: str             # container/codec of ``audio`` (e.g. "opus")
    tier: str            # "<provider>:<model>" that produced the audio
    elapsed_ms: float
    nbytes: int = 0
    chars: int = 0       # input-text length — the TTS billable unit (for cogno-meter)

    def __post_init__(self) -> None:
        if not self.nbytes:
            object.__setattr__(self, "nbytes", len(self.audio))
