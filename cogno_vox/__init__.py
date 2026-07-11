"""cogno-vox — voice/audio I/O edge for the Cogno cognitive pipeline.

STT before the mind, TTS after it:
    gateway -> [vox STT] -> cogno-anima -> [vox TTS] -> gateway

The core imports neither cogno-anima nor cogno-engram — it only converts
audio <-> text behind two capability-scoped protocols with provider-agnostic
fallback chains.
"""

from cogno_vox.chunker import split_text_for_tts
from cogno_vox.factory import create_synthesizer, create_transcriber
from cogno_vox.text_prep import clean_text_for_tts
from cogno_vox.ports import (
    SynthesisError,
    SynthesizerBackend,
    TranscriberBackend,
    TranscriptionError,
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
    SynthesisResult,
    TierConfig,
    TranscriptionResult,
    VoxConfig,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # ports + errors
    "TranscriberBackend", "SynthesizerBackend",
    "VoxError", "TranscriptionError", "SynthesisError",
    # types
    "VoxConfig", "TierConfig", "TranscriptionResult", "SynthesisResult",
    "SUPPORTED_INPUT_FORMATS",
    # chains
    "FallbackTranscriber", "FallbackSynthesizer",
    # transcriber backends
    "OpenAICompatTranscriber", "GeminiTranscriber", "BedrockTranscriber",
    # synthesizer backends
    "OpenAICompatSynthesizer", "GrokSynthesizer",
    "ElevenLabsSynthesizer", "GeminiSynthesizer",
    # factory + utils
    "create_transcriber", "create_synthesizer", "split_text_for_tts",
    "clean_text_for_tts",
]
