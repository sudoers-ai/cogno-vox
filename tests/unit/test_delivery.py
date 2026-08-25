"""The delivery profile: what each engine family is told, and who is told nothing.

Three claims are pinned here, and the third is the one that makes the feature safe to ship:

1. ``sanitize_delivery`` is total — anything in, a valid profile out, never an exception;
2. each renderer says only what the caller asked for, in a stable order;
3. **an engine that cannot shape delivery is not a failure.** It speaks the same words and the
   call succeeds; and a profile that says nothing produces the byte-identical request the code
   produced before this feature existed.
"""

from __future__ import annotations

import pytest

from cogno_vox import (
    DeliveryAwareBackend,
    DeliveryProfile,
    ElevenLabsSynthesizer,
    FallbackSynthesizer,
    GeminiSynthesizer,
    GrokSynthesizer,
    OpenAICompatSynthesizer,
    as_instructions,
    as_voice_settings,
    sanitize_delivery,
    shapes_delivery,
)
from cogno_vox import TierConfig, VoxConfig, create_synthesizer
from tests.conftest import StubSynthesizer


# ── sanitize: total function ──────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ({"style": "warm"}, DeliveryProfile(style="warm")),
    ({"style": "WARM", "pace": " Fast "}, DeliveryProfile(style="warm", pace="fast")),
    ({"style": "warm", "pace": "steady", "energy": "low"},
     DeliveryProfile(style="warm", pace="steady", energy="low")),
    ({}, DeliveryProfile()),
    (None, DeliveryProfile()),
    ("warm", DeliveryProfile()),                 # a bare string is not a profile
    (["warm"], DeliveryProfile()),
    ({"style": None, "pace": ""}, DeliveryProfile()),
    ({"style": 3}, DeliveryProfile()),
    (DeliveryProfile(style="reserved"), DeliveryProfile(style="reserved")),
])
def test_sanitize_accepts_anything_and_never_raises(raw, expected):
    profile, _ = sanitize_delivery(raw)
    assert profile == expected


def test_an_unknown_value_is_dropped_not_guessed():
    """Correcting `friendly` to `warm` would put a tone on a voice nobody asked for; the
    failure mode of dropping is the engine default, which is what saying nothing gives."""
    profile, dropped = sanitize_delivery({"style": "friendly", "pace": "slow"})
    assert profile == DeliveryProfile(pace="slow")
    assert dropped == ["style=friendly"]


def test_an_axis_that_does_not_exist_is_reported_by_name():
    """The host wiring its own field name is the realistic bug, and the log line has to name
    it — `dropped` is what tells a host it has been sending nothing for a week."""
    _, dropped = sanitize_delivery({"style": "warm", "mood": "chirpy", "volume": 3})
    assert dropped == ["mood", "volume"]


def test_a_rejected_value_is_labelled_safely():
    """`dropped` goes into a log line, so a hostile or malformed value must not carry control
    characters or unbounded length into it."""
    _, dropped = sanitize_delivery({"style": "x" * 500 + "\n\x00INJECTED"})
    assert len(dropped) == 1 and len(dropped[0]) <= 46         # "style=" + 40
    assert "\n" not in dropped[0] and "\x00" not in dropped[0]


# ── instruction engines ───────────────────────────────────────────────────

def test_instructions_say_only_what_was_asked_in_a_stable_order():
    assert as_instructions(DeliveryProfile(style="warm")) == "Speak warm and unhurried."
    assert as_instructions(DeliveryProfile(style="warm", pace="fast", energy="high")) == (
        "Speak warm and unhurried, at a brisk pace, bright and animated.")
    # order is a property of the AXES, not of the dict the caller built
    assert as_instructions(DeliveryProfile(energy="high", style="warm")) == (
        "Speak warm and unhurried, bright and animated.")


def test_the_default_of_an_axis_renders_nothing():
    """Naming the default spends instruction budget asking for what already happens."""
    assert as_instructions(DeliveryProfile(pace="steady", energy="normal")) == ""
    assert as_instructions(DeliveryProfile(style="warm", pace="steady")) == (
        "Speak warm and unhurried.")


def test_an_empty_profile_renders_nothing():
    assert as_instructions(DeliveryProfile()) == ""


# ── parameter engines ─────────────────────────────────────────────────────

def test_voice_settings_override_only_the_axes_that_were_set():
    assert as_voice_settings(DeliveryProfile()) == {"stability": 0.5, "similarity_boost": 0.75}
    warm = as_voice_settings(DeliveryProfile(style="warm"))
    assert warm["style"] == 0.30
    assert warm["stability"] == 0.5 and warm["similarity_boost"] == 0.75   # untouched


def test_a_slower_read_is_a_more_stable_one():
    """The documented approximation: ElevenLabs has no pace control, and `stability` is the
    closest lever. Pinned so the mapping cannot drift silently."""
    assert as_voice_settings(DeliveryProfile(pace="slow"))["stability"] > 0.5
    assert as_voice_settings(DeliveryProfile(pace="fast"))["stability"] < 0.5


# ── who can be shaped at all ──────────────────────────────────────────────

def test_an_adapter_that_can_carry_a_profile_is_not_asked_to_unless_the_ENGINE_declares_it():
    """The hole a review measured, and the reason capability is DECLARED per tier.

    `OpenAICompatSynthesizer` drives OpenAI, Kokoro, Dia and Orpheus over one HTTP shape — only
    the first honours `instructions`. Asking the class answers "which transport is this", which
    is a different question wearing the right question's clothes: `instructions` was being sent
    to two tag engines whose own docs say they ignore it.
    """
    undeclared = ShapingStub("local:dia", b"AUDIO")
    undeclared.delivery_dialect = ""                    # can carry it; does not understand it
    assert isinstance(undeclared, DeliveryAwareBackend)  # the class-only probe still says yes…
    assert not shapes_delivery(undeclared, DeliveryProfile(style="warm"))   # …and is wrong

    declared = ShapingStub("openai:gpt-4o-mini-tts", b"AUDIO")
    assert shapes_delivery(declared, DeliveryProfile(style="warm"))


async def test_an_undeclared_tier_speaks_unshaped_rather_than_failing():
    tier = ShapingStub("local:dia", b"AUDIO")
    tier.delivery_dialect = ""
    result = await FallbackSynthesizer([tier]).synthesize(
        "Bom dia.", delivery=DeliveryProfile(style="warm"))
    assert result.audio == b"AUDIO" and tier.shaped == [] and tier.plain == 1


def test_only_the_engines_that_can_shape_satisfy_the_protocol():
    """The probe is the whole safety mechanism: a chain mixes shaping and plain engines, and
    the plain ones must be invisible to it."""
    assert issubclass(OpenAICompatSynthesizer, DeliveryAwareBackend)
    assert issubclass(ElevenLabsSynthesizer, DeliveryAwareBackend)
    assert not issubclass(GrokSynthesizer, DeliveryAwareBackend)
    assert not issubclass(GeminiSynthesizer, DeliveryAwareBackend)
    assert not issubclass(StubSynthesizer, DeliveryAwareBackend)


# ── the chain ─────────────────────────────────────────────────────────────

class ShapingStub(StubSynthesizer):
    """A tier that can be shaped, recording what it was told.

    It declares `delivery_dialect` because carrying the method is only half the answer — see
    `test_an_adapter_that_can_carry_a_profile_is_not_asked_to_unless_the_ENGINE_declares_it`.
    """

    delivery_dialect = "instructions"

    def __init__(self, name: str, audio: bytes) -> None:
        super().__init__(name, audio)
        self.shaped: list[DeliveryProfile] = []
        self.plain = 0

    async def synthesize(self, text: str) -> bytes:
        self.plain += 1
        return await super().synthesize(text)

    async def synthesize_shaped(self, text: str, delivery: DeliveryProfile) -> bytes:
        self.shaped.append(delivery)
        return await super().synthesize(text)


async def test_the_profile_reaches_a_tier_that_can_use_it():
    tier = ShapingStub("openai:gpt-4o-mini-tts", b"AUDIO")
    profile = DeliveryProfile(style="warm", pace="slow")
    await FallbackSynthesizer([tier]).synthesize("Bom dia.", delivery=profile)
    assert tier.shaped == [profile] and tier.plain == 0


async def test_a_plain_tier_speaks_the_same_words_and_the_call_SUCCEEDS():
    """The contract that makes this safe: an engine with no lever is not a failure and never
    a failover. Mutating the chain to treat it as one would strand every plain-engine tenant."""
    plain = StubSynthesizer("grok:grok-2-tts", b"AUDIO")
    result = await FallbackSynthesizer([plain]).synthesize(
        "Bom dia.", delivery=DeliveryProfile(style="warm"))
    assert result.audio == b"AUDIO" and plain.calls == 1


async def test_failover_from_a_shaping_tier_degrades_the_DELIVERY_not_the_call():
    """Probed per tier, not once for the chain — so the fallback still speaks."""
    dead = ShapingStub("openai:gpt-4o-mini-tts", b"")        # returns b"" → fails over
    plain = StubSynthesizer("grok:grok-2-tts", b"AUDIO")
    result = await FallbackSynthesizer([dead, plain]).synthesize(
        "Bom dia.", delivery=DeliveryProfile(style="warm"))
    assert result.audio == b"AUDIO" and result.tier == "grok:grok-2-tts"
    assert dead.shaped and plain.calls == 1


async def test_no_profile_is_the_old_behaviour():
    """A caller that says nothing must take the pre-feature path exactly — including on a tier
    that COULD have been shaped."""
    tier = ShapingStub("openai:gpt-4o-mini-tts", b"AUDIO")
    await FallbackSynthesizer([tier]).synthesize("Bom dia.")
    await FallbackSynthesizer([tier]).synthesize("Bom dia.", delivery=DeliveryProfile())
    assert tier.shaped == [] and tier.plain == 2


async def test_delivery_and_emotion_compose():
    """They are different mechanisms — a cue is inline text, a profile is engine config — and
    a turn may carry both."""
    class Both(ShapingStub):
        emotion_dialect = "dia"

    tier = Both("local:dia", b"AUDIO")
    await FallbackSynthesizer([tier]).synthesize(
        "Que boa notícia! Parabéns.", emotion="laugh",
        delivery=DeliveryProfile(energy="high"))
    assert tier.shaped == [DeliveryProfile(energy="high")]


# ── regressions from the review ───────────────────────────────────────────

class _Hostile:
    """Something that arrived across a boundary and does not want to be described."""

    def __str__(self) -> str:
        raise RuntimeError("nope")


@pytest.mark.parametrize("raw", [
    {None: 1, "a": 2},                      # mixed key types: `sorted(keys)` raised TypeError
    {(1, 2): "a", "b": "c"},
    {"style": _Hostile()},                  # a value whose __str__ raises
    {_Hostile(): "warm"},                   # ...and a key
    {"style": {"nested": "dict"}},
    {"style": b"\xff\xfe"},
])
def test_the_barrier_the_host_is_told_to_trust_never_raises(raw):
    """`sanitize_delivery` is the guard a host puts in FRONT of the boundary. A guard that
    raises on the malformed input it exists to absorb is worse than no guard — the host wrote
    `sanitize_delivery(...)` precisely so it would not have to write a try/except."""
    profile, dropped = sanitize_delivery(raw)
    assert isinstance(profile, DeliveryProfile) and isinstance(dropped, list)


async def test_a_raw_DICT_never_escapes_as_an_exception():
    """The HIGH the review found, pinned at the chain.

    A plain dict is the shape `sanitize_delivery` documents and the shape a host serializing
    across a process boundary produces. It used to raise `AttributeError` from inside
    `synthesize_shaped` — outside every `return b""` guard — so:
      * one tier  → the chain raised `AttributeError`, not `SynthesisError`; the host's handler
                    missed it and the voice note was lost;
      * mixed     → `resilient_call` read it as a tier failure and recorded a breaker fault
                    against a HEALTHY engine, degrading later turns that pass no delivery.
    """
    tier = ShapingStub("openai:gpt-4o-mini-tts", b"AUDIO")
    result = await FallbackSynthesizer([tier]).synthesize("Bom dia.", delivery={"style": "warm"})
    assert result.audio == b"AUDIO"
    assert as_instructions({"style": "warm"}) == "Speak warm and unhurried."


async def test_a_healthy_tier_is_never_blamed_for_a_malformed_profile():
    """The second half of the same finding: the FIRST tier must still win. If the malformed
    profile were read as a tier failure the chain would fail over past a working engine."""
    good = ShapingStub("openai:gpt-4o-mini-tts", b"FIRST")
    spare = StubSynthesizer("grok:grok-2-tts", b"SECOND")
    result = await FallbackSynthesizer([good, spare]).synthesize(
        "Bom dia.", delivery={"style": "not-a-real-style"})
    assert result.tier == "openai:gpt-4o-mini-tts" and spare.calls == 0


def test_energy_has_no_lever_on_a_parameter_engine_and_that_is_deliberate():
    """It used to map to `similarity_boost`, which is voice ADHERENCE: `high` → 0.85 holds the
    read *closer* to its reference, the opposite of "bright and animated". Between a wrong
    mapping and none, none — an unshaped axis is a preference unheard, an inverted one fights
    the delivery that was asked for. Pinned so it cannot come back without a measurement."""
    assert as_voice_settings(DeliveryProfile(energy="high")) == \
        as_voice_settings(DeliveryProfile(energy="low")) == \
        as_voice_settings(DeliveryProfile())


def test_the_dialect_travels_from_TierConfig_through_the_factory():
    """Removing the factory's stamp survived every other test in this file — the fix would have
    shipped inert. Everything above builds a stub that sets the attribute by hand; nothing
    exercised the one line that puts it there from the host's config.

    Kokoro and `gpt-4o-mini-tts` are the same adapter class over the same HTTP shape, so this is
    also the clearest statement of why the class cannot be the answer.
    """
    def tier(model: str, dialect: str):
        return create_synthesizer(VoxConfig(synthesize_tiers=(
            TierConfig(provider="local" if model == "kokoro" else "openai", model=model,
                       base_url="http://x", delivery_dialect=dialect),))).backends[0]

    warm = DeliveryProfile(style="warm")
    assert not shapes_delivery(tier("kokoro", ""), warm)          # declares nothing → unshaped
    assert not shapes_delivery(tier("dia", ""), warm)             # a tag engine, same adapter
    assert shapes_delivery(tier("gpt-4o-mini-tts", "instructions"), warm)
    # ...and the attribute really is the config's, not a default that happens to match
    assert tier("kokoro", "instructions").delivery_dialect == "instructions"
