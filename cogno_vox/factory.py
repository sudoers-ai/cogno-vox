"""
cogno_vox.factory — build fallback chains from a host-injected ``VoxConfig``.

The host owns tier selection (model ladder, per-tenant routing, BYOK). The
factory just maps each ``TierConfig.provider`` to the right backend and assembles
the chain in order. Unlike the parent, it reads NO YAML and NO env contextvar.
"""

from __future__ import annotations

from typing import cast

from cogno_vox.ports import SynthesizerBackend, TranscriberBackend
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
from cogno_vox.types import TierConfig, VoxConfig


def _build_transcriber(cfg: TierConfig) -> TranscriberBackend:
    p = cfg.provider
    if p == "gemini":
        return GeminiTranscriber(api_key=cfg.api_key, model=cfg.model,
                                 timeout=cfg.timeout)
    if p == "bedrock":
        region = str(cfg.extra.get("region", "us-east-1"))
        return BedrockTranscriber(model=cfg.model, region=region,
                                  timeout=cfg.timeout)
    # "local" / "openai" / "groq" → OpenAI-compatible HTTP
    return OpenAICompatTranscriber(
        base_url=cfg.base_url, model=cfg.model, api_key=cfg.api_key,
        timeout=cfg.timeout,
    )


def _build_synthesizer(cfg: TierConfig, default_fmt: str) -> SynthesizerBackend:
    p = cfg.provider
    fmt = cfg.response_format or default_fmt
    voice = cfg.voice
    backend: SynthesizerBackend
    if p == "grok":
        backend = GrokSynthesizer(
            api_key=cfg.api_key, base_url=cfg.base_url or "https://api.x.ai/v1",
            model=cfg.model, voice=voice or "eve", response_format=fmt,
            sample_rate=cast(int, cfg.extra.get("sample_rate", 24000)),
            timeout=cfg.timeout,
        )
    elif p == "elevenlabs":
        backend = ElevenLabsSynthesizer(
            api_key=cfg.api_key,
            base_url=cfg.base_url or "https://api.elevenlabs.io/v1",
            model=cfg.model, voice=voice or "JBFqnCBsd6RMkjVDRZzb",
            response_format=fmt, timeout=cfg.timeout,
        )
    elif p == "gemini":
        backend = GeminiSynthesizer(api_key=cfg.api_key, model=cfg.model,
                                    voice=voice or "Kore", timeout=cfg.timeout)
    else:
        # "local" (Kokoro, a Dia wrapper) / "openai" → OpenAI-compatible HTTP
        backend = OpenAICompatSynthesizer(
            base_url=cfg.base_url, model=cfg.model, api_key=cfg.api_key,
            voice=voice or "alloy", response_format=fmt, timeout=cfg.timeout,
        )
    # The tier's expressive-tag dialect rides on the backend instance so the fallback
    # chain can decorate per tier (a failover to a plain engine must not carry tags).
    setattr(backend, "emotion_dialect", cfg.emotion_dialect)
    return backend


def create_transcriber(config: VoxConfig) -> FallbackTranscriber:
    """Assemble the STT fallback chain from ``config.transcribe_tiers``."""
    return FallbackTranscriber(
        [_build_transcriber(c) for c in config.transcribe_tiers]
    )


def create_synthesizer(config: VoxConfig) -> FallbackSynthesizer:
    """Assemble the TTS fallback chain from ``config.synthesize_tiers``."""
    return FallbackSynthesizer(
        [_build_synthesizer(c, config.default_tts_format)
         for c in config.synthesize_tiers]
    )
