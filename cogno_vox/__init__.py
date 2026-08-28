"""cogno-vox — voice/audio I/O edge for the Cogno cognitive pipeline.

STT before the mind, TTS after it:
    gateway -> [vox STT] -> cogno-anima -> [vox TTS] -> gateway

The core imports neither cogno-anima nor cogno-engram — it only converts
audio <-> text behind two capability-scoped protocols with provider-agnostic
fallback chains.
"""

from cogno_vox.chunker import split_text_for_tts
from cogno_vox.delivery import (
    as_instructions,
    as_voice_settings,
    sanitize_delivery,
    shapes_delivery,
)
from cogno_vox.factory import create_synthesizer, create_transcriber, create_vision_analyzer
from cogno_vox.text_prep import apply_emotion, clean_text_for_tts, strip_emotion_tags
from cogno_vox.ports import (
    DeliveryAwareBackend,
    SynthesisError,
    SynthesizerBackend,
    TranscriberBackend,
    TranscriptionError,
    VisionAnalyzerBackend,
    VisionError,
    VoxError,
)
from cogno_vox.synthesizer import (
    ElevenLabsSynthesizer,
    FallbackSynthesizer,
    GeminiSynthesizer,
    GrokSynthesizer,
    OpenAICompatSynthesizer,
)
from cogno_vox.transcriber import (
    BedrockTranscriber,
    FallbackTranscriber,
    GeminiTranscriber,
    OpenAICompatTranscriber,
)
from cogno_vox.types import (
    SUPPORTED_INPUT_FORMATS,
    SUPPORTED_VISION_FORMATS,
    VALID_DELIVERY_ENERGY,
    VALID_DELIVERY_PACE,
    VALID_DELIVERY_STYLE,
    VALID_DELIVERY_DIALECTS,
    DELIVERY_INSTRUCTIONS,
    DELIVERY_VOICE_SETTINGS,
    DeliveryProfile,
    SynthesisResult,
    TierConfig,
    TranscriptionResult,
    VisionAnalysisResult,
    VoxConfig,
)
from cogno_vox.vision import FallbackVisionAnalyzer, OpenAICompatVisionAnalyzer

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # ports + errors
    "TranscriberBackend", "SynthesizerBackend", "DeliveryAwareBackend", "VisionAnalyzerBackend",
    "VoxError", "TranscriptionError", "SynthesisError", "VisionError",
    # types
    "VoxConfig", "TierConfig", "TranscriptionResult", "SynthesisResult", "VisionAnalysisResult",
    "SUPPORTED_INPUT_FORMATS", "SUPPORTED_VISION_FORMATS",
    # delivery profile (HOW the words are said)
    "DeliveryProfile", "VALID_DELIVERY_STYLE", "VALID_DELIVERY_PACE",
    "VALID_DELIVERY_ENERGY", "VALID_DELIVERY_DIALECTS", "DELIVERY_INSTRUCTIONS",
    "DELIVERY_VOICE_SETTINGS", "sanitize_delivery", "shapes_delivery",
    "as_instructions", "as_voice_settings",
    # chains
    "FallbackTranscriber", "FallbackSynthesizer", "FallbackVisionAnalyzer",
    # transcriber backends
    "OpenAICompatTranscriber", "GeminiTranscriber", "BedrockTranscriber",
    # synthesizer backends
    "OpenAICompatSynthesizer", "GrokSynthesizer",
    "ElevenLabsSynthesizer", "GeminiSynthesizer",
    # vision backends
    "OpenAICompatVisionAnalyzer",
    # factory + utils
    "create_transcriber", "create_synthesizer", "create_vision_analyzer",
    "split_text_for_tts",
    "apply_emotion",
    "clean_text_for_tts",
    "strip_emotion_tags",
]


