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

# Image and video formats accepted for vision analysis.
SUPPORTED_VISION_FORMATS = frozenset(
    {"jpg", "jpeg", "png", "webp", "gif", "mp4", "webm", "ogg", "mov"}
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
    # TTS only: the expressive-tag dialect this engine understands ("dia", "orpheus" — see
    # text_prep._EMOTION_DIALECTS). "" → plain engine: an emotion hint is ignored for this
    # tier and any inline tag is stripped before it would be read out loud.
    emotion_dialect: str = ""
    # TTS only: HOW this engine takes a delivery profile — "instructions" (a prose field, as
    # `gpt-4o-mini-tts` does), "voice_settings" (numbers, as ElevenLabs does), or "" (it does
    # not take one). Declared per TIER and NOT inferred from the adapter class, because the
    # class cannot answer it: `OpenAICompatSynthesizer` drives OpenAI, Kokoro, Dia AND Orpheus
    # over the same HTTP shape and only the first honours `instructions`. A review measured
    # this exact hole — the profile was reaching two tag engines whose own docs ignore it.
    delivery_dialect: str = ""
    timeout: float = 60.0
    extra: dict[str, object] = field(default_factory=dict)


# ── Delivery profile (HOW the words are said, not WHICH words) ────────────
#
# Distinct from ``emotion``, and the distinction is the whole design: an emotion hint is a
# discrete NON-VERBAL cue ("chuckle") rendered as an inline tag, present or absent. A delivery
# profile shapes the WHOLE utterance — three independent axes, each a closed vocabulary — and
# every engine renders it differently or not at all (see ``cogno_vox.delivery``).
#
# Kept as three flat axes rather than one "mood" label on purpose: the host derives them from
# separate signals (style from the persona's modulated traits, pace from urgency, energy from
# the turn's outcome), and collapsing them would force it to pick a winner where there is none.

# How the voice relates to the listener.
VALID_DELIVERY_STYLE = frozenset({"warm", "reserved", "empathetic"})
# How fast the words come.
VALID_DELIVERY_PACE = frozenset({"fast", "steady", "slow"})
# How much lift the voice carries.
VALID_DELIVERY_ENERGY = frozenset({"high", "normal", "low"})

# The two ways an engine can be TOLD a delivery; "" means it cannot be told at all.
DELIVERY_INSTRUCTIONS = "instructions"
DELIVERY_VOICE_SETTINGS = "voice_settings"
VALID_DELIVERY_DIALECTS = frozenset({DELIVERY_INSTRUCTIONS, DELIVERY_VOICE_SETTINGS})


@dataclass(frozen=True)
class DeliveryProfile:
    """Engine-agnostic delivery shape. Every axis is optional; ``""`` means "engine default".

    Falsy when no axis is set, so a caller can write ``if delivery:`` and a profile that says
    nothing costs nothing — the synthesizer then takes the byte-identical path it took before
    this type existed.
    """

    style: str = ""
    pace: str = ""
    energy: str = ""

    def __bool__(self) -> bool:
        return bool(self.style or self.pace or self.energy)


@dataclass(frozen=True)
class VoxConfig:
    """Host-injected configuration: an ordered list of tiers per direction.

    Tiers are tried in order; the first that returns a non-empty result wins.
    """

    transcribe_tiers: tuple[TierConfig, ...] = ()
    synthesize_tiers: tuple[TierConfig, ...] = ()
    vision_tiers: tuple[TierConfig, ...] = ()
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


@dataclass(frozen=True)
class VisionAnalysisResult:
    """Outcome of a successful vision analysis call."""

    summary: str
    category: str = "GENERAL_IMAGE"
    extracted_data: dict[str, object] = field(default_factory=dict)
    confidence: float = 1.0
    tier: str = ""        # "<provider>:<model>" that performed analysis
    elapsed_ms: float = 0.0

