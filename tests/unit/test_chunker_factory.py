"""Pure chunker math + factory wiring (provider -> backend type)."""

from __future__ import annotations

from cogno_vox import (
    ElevenLabsSynthesizer,
    GeminiSynthesizer,
    GeminiTranscriber,
    GrokSynthesizer,
    OpenAICompatSynthesizer,
    OpenAICompatTranscriber,
    TierConfig,
    VoxConfig,
    create_synthesizer,
    create_transcriber,
    split_text_for_tts,
)


# ── chunker ───────────────────────────────────────────────────────────────

def test_chunker_empty():
    assert split_text_for_tts("") == []
    assert split_text_for_tts("   ") == []


def test_chunker_short_single_segment():
    assert split_text_for_tts("Olá, tudo bem?") == ["Olá, tudo bem?"]


def test_chunker_splits_on_sentence_boundary():
    sentence = "Esta é uma frase com várias palavras para somar bastante. "
    text = sentence * 4  # ~40 words -> exceeds a small max
    segs = split_text_for_tts(text, max_words=10)
    assert len(segs) > 1
    # every segment stays within the word budget
    assert all(len(s.split()) <= 10 for s in segs)
    # reassembled word set is preserved
    assert " ".join(segs).split() == text.split()


def test_chunker_caps_segments():
    text = " ".join(f"word{i}." for i in range(200))
    segs = split_text_for_tts(text, max_words=5, max_segments=3)
    assert len(segs) == 3


# ── factory ───────────────────────────────────────────────────────────────

def test_factory_builds_transcriber_chain_by_provider():
    cfg = VoxConfig(transcribe_tiers=(
        TierConfig(provider="local", model="small", base_url="http://localhost:8000"),
        TierConfig(provider="gemini", model="gemini-2.5-flash", api_key="g"),
    ))
    chain = create_transcriber(cfg)
    assert isinstance(chain.backends[0], OpenAICompatTranscriber)
    assert isinstance(chain.backends[1], GeminiTranscriber)


def test_factory_builds_synthesizer_chain_by_provider():
    cfg = VoxConfig(synthesize_tiers=(
        TierConfig(provider="local", model="kokoro", base_url="http://localhost:8880/v1",
                   voice="af_alloy"),
        TierConfig(provider="grok", model="grok-2-tts", api_key="x"),
        TierConfig(provider="elevenlabs", model="eleven_multilingual_v2", api_key="e"),
        TierConfig(provider="gemini", model="gemini-2.5-flash-preview-tts", api_key="g"),
    ))
    chain = create_synthesizer(cfg)
    assert isinstance(chain.backends[0], OpenAICompatSynthesizer)
    assert isinstance(chain.backends[1], GrokSynthesizer)
    assert isinstance(chain.backends[2], ElevenLabsSynthesizer)
    assert isinstance(chain.backends[3], GeminiSynthesizer)


def test_factory_default_format_flows_to_backend():
    cfg = VoxConfig(
        synthesize_tiers=(TierConfig(provider="openai", model="tts-1", api_key="k"),),
        default_tts_format="mp3",
    )
    chain = create_synthesizer(cfg)
    assert chain.backends[0].fmt == "mp3"


def test_factory_tier_response_format_overrides_default():
    cfg = VoxConfig(
        synthesize_tiers=(
            TierConfig(provider="openai", model="tts-1", api_key="k",
                       response_format="opus"),
        ),
        default_tts_format="mp3",
    )
    chain = create_synthesizer(cfg)
    assert chain.backends[0].fmt == "opus"


def test_factory_propagates_emotion_dialect():
    from cogno_vox import TierConfig, VoxConfig, create_synthesizer
    chain = create_synthesizer(VoxConfig(synthesize_tiers=(
        TierConfig(provider="local", model="dia-ptbr", base_url="http://x", emotion_dialect="dia"),
        TierConfig(provider="local", model="kokoro", base_url="http://y"),
    )))
    assert getattr(chain.backends[0], "emotion_dialect", "") == "dia"
    assert getattr(chain.backends[1], "emotion_dialect", "") == ""


def test_factory_local_opus_goes_via_wav_but_cloud_does_not():
    # Kokoro's own opus encoder truncates the tail → the local tier re-encodes via WAV;
    # cloud (OpenAI) and non-opus local tiers (the Dia wrapper) are untouched.
    cfg = VoxConfig(synthesize_tiers=(
        TierConfig(provider="local", model="kokoro", base_url="http://localhost:8880/v1",
                   voice="pf_dora", response_format="opus"),
        TierConfig(provider="openai", model="tts-1", api_key="k", response_format="opus"),
        TierConfig(provider="dia", model="dia-ptbr", base_url="http://localhost:8881/v1",
                   response_format="opus"),
    ))
    chain = create_synthesizer(cfg)
    local, cloud, dia = chain.backends
    assert local._opus_via_wav is True
    assert cloud._opus_via_wav is False
    assert dia._opus_via_wav is False


def test_factory_sanitizes_kokoro_voice_on_openai_tier():
    # The host's language→voice map picks Kokoro names ("pf_dora"); OpenAI 400s on them.
    cfg = VoxConfig(synthesize_tiers=(
        TierConfig(provider="openai", model="tts-1", api_key="k", voice="pf_dora"),
        TierConfig(provider="openai", model="tts-1", api_key="k", voice="nova"),
        TierConfig(provider="local", model="kokoro", base_url="http://localhost:8880/v1",
                   voice="pf_dora"),
    ))
    chain = create_synthesizer(cfg)
    bad, good, local = chain.backends
    assert bad._voice == "alloy"       # invalid OpenAI voice → default
    assert good._voice == "nova"       # a real OpenAI voice is kept
    assert local._voice == "pf_dora"   # the local tier keeps the Kokoro voice
