"""
cogno_vox.delivery — a delivery profile, and how each engine family renders it.

Pure functions, no I/O. The host decides WHAT the delivery should be (it owns the persona,
the contact and the turn); this module owns only the translation into what a given engine
can actually be told, which is different in kind for each family:

* **instruction engines** (``gpt-4o-mini-tts`` and the OpenAI-compatible servers that copy
  its API) take a free-text ``instructions`` field — prose describing the delivery;
* **parameter engines** (ElevenLabs) take numbers — ``style``/``stability``/``similarity_boost``;
* **tag engines** (Dia, Orpheus) can voice a discrete cue but have no lever for a sustained
  delivery, and **plain engines** have neither. Both simply ignore a profile.

That last line is the contract: an engine that cannot honour a profile is not an error and
never a failover — it speaks the same words, unshaped. A delivery profile is a preference,
and a preference that breaks a voice note is worse than one that goes unheard.

The instruction prose is **English regardless of the spoken language**. The field is a
directive to the model about HOW to speak, not text to be spoken, and this library
synthesizes any language — a Portuguese instruction would tie the shaping to one locale for
no gain. (Consistent with the repo's English-canonical rule; the words the listener hears
come from the caller, untouched.)
"""

from __future__ import annotations

from typing import Any

from cogno_vox.types import (
    VALID_DELIVERY_ENERGY,
    VALID_DELIVERY_PACE,
    VALID_DELIVERY_STYLE,
    DeliveryProfile,
)

_AXES: tuple[tuple[str, frozenset[str]], ...] = (
    ("style", VALID_DELIVERY_STYLE),
    ("pace", VALID_DELIVERY_PACE),
    ("energy", VALID_DELIVERY_ENERGY),
)


def sanitize_delivery(raw: Any) -> "tuple[DeliveryProfile, list[str]]":
    """``(profile, dropped)`` from anything — a dict, a profile, or garbage. NEVER raises.

    The host builds this from its own signals and hands it across a process boundary, so it
    can arrive as any shape at all. An unknown axis value is DROPPED rather than corrected:
    guessing which of ``warm``/``reserved`` a typo meant would put a tone on a voice nobody
    asked for, and the failure mode of dropping is the engine default — exactly what a
    caller who said nothing would get.

    ``dropped`` is for the caller to log ONCE per configuration, not per turn: a host that
    keeps sending ``"friendly"`` has a wiring bug, and a warning that repeats every message
    is the same as no warning.
    """
    if isinstance(raw, DeliveryProfile):
        raw = {"style": raw.style, "pace": raw.pace, "energy": raw.energy}
    elif hasattr(raw, "style") or hasattr(raw, "pace") or hasattr(raw, "energy"):
        raw = {axis: getattr(raw, axis, "") for axis, _ in _AXES}
    if not isinstance(raw, dict):
        return DeliveryProfile(), ([_label(raw)] if raw else [])

    kept: dict[str, str] = {}
    dropped: list[str] = []
    for axis, vocab in _AXES:
        value = raw.get(axis)
        if value is None or value == "":
            continue
        try:
            text = str(value).strip().lower()
        except Exception:                       # noqa: BLE001 — same reason as `_label`
            dropped.append(f"{axis}=<unprintable>")
            continue
        if text in vocab:
            kept[axis] = text
        else:
            dropped.append(f"{axis}={_label(value)}")
    # `sorted` on the KEYS raises on a dict with mixed key types (`{None: 1, "a": 2}` —
    # measured), which a JSON round-trip or a hand-built dict produces easily. Sorting the
    # LABELS keeps the output stable without asking the keys to be comparable.
    dropped.extend(sorted(_label(k) for k in set(raw) - {axis for axis, _ in _AXES}))
    return DeliveryProfile(**kept), dropped


def _label(value: Any) -> str:
    """A short, printable stand-in for a rejected value — it goes into a log line.

    ``str()`` is inside the guard because a value that reaches here is by definition one the
    caller got wrong, and an object whose ``__str__`` raises is exactly the kind of thing that
    arrives across a boundary. A sanitizer that raises while describing what it rejected is
    worse than no sanitizer: the host put it there so it would not have to write try/except."""
    try:
        text = "".join(ch for ch in str(value) if ch.isprintable())
    except Exception:                   # noqa: BLE001 — a hostile __str__ is data, not a bug
        return "<unprintable>"
    return text[:40] if text else "<empty>"


def shapes_delivery(backend: Any, delivery: Any) -> bool:
    """Will THIS tier be shaped by THIS profile? Two questions, both required.

    The adapter must be able to carry a profile (``DeliveryAwareBackend``) **and** the engine
    behind it must declare a dialect. The second is not optional and not inferable: one adapter
    class drives OpenAI, Kokoro, Dia and Orpheus over the same HTTP shape, and only the first
    honours ``instructions``. Asking the class alone answers "which transport is this" — a
    different question wearing the right question's clothes.
    """
    from cogno_vox.ports import DeliveryAwareBackend      # local: ports imports types, not this

    profile, _ = sanitize_delivery(delivery)
    return (bool(profile) and isinstance(backend, DeliveryAwareBackend)
            and bool(getattr(backend, "delivery_dialect", "")))


# ── instruction engines (gpt-4o-mini-tts and the OpenAI-compatible copies) ──
#
# Fragments, not sentences: they are joined into one directive, and an engine reads a short
# comma-separated phrase more reliably than prose. `normal`/`steady` deliberately render
# NOTHING — naming the default spends instruction budget to ask for what already happens.
_INSTRUCTIONS: dict[str, dict[str, str]] = {
    "style": {"warm": "warm and unhurried",
              "reserved": "courteous and restrained",
              "empathetic": "gentle and understanding"},
    "pace": {"fast": "at a brisk pace", "slow": "slowly, without rushing", "steady": ""},
    "energy": {"high": "bright and animated", "low": "subdued", "normal": ""},
}


def as_instructions(profile: Any) -> str:
    """The ``instructions`` field for an instruction engine, or ``""`` when there is nothing
    to say. Order is fixed (style, pace, energy) so the same profile always renders the same
    string — a caching layer, a diff or a test can rely on it.

    Coerced, not trusted. A review measured the alternative: a plain ``dict`` — the very shape
    ``sanitize_delivery`` accepts, and the shape a host serializing across a process boundary
    produces — raised ``AttributeError`` from here, INSIDE ``synthesize_shaped`` and therefore
    OUTSIDE the backend's ``try``. One tier: the chain raised a bare ``AttributeError`` instead
    of ``SynthesisError``, the host's handler missed it, the voice note was lost. Mixed chain:
    ``resilient_call`` read it as a tier failure and recorded a breaker fault against a HEALTHY
    engine, degrading later turns that pass no delivery at all. A preference that breaks a
    voice note is worse than one that goes unheard — the guard belongs where it is read."""
    profile, _ = sanitize_delivery(profile)
    parts = [_INSTRUCTIONS[axis].get(getattr(profile, axis), "") for axis, _ in _AXES]
    said = [p for p in parts if p]
    return "Speak " + ", ".join(said) + "." if said else ""


# ── parameter engines (ElevenLabs) ─────────────────────────────────────────
#
# ElevenLabs exposes expressiveness, not delivery: `style` exaggerates, `stability` trades
# consistency for variation, `similarity_boost` holds the voice to its reference. None of
# them is a pace control, and pretending otherwise would be the honest failure here — so
# `pace` maps to `stability` as the CLOSEST available lever (a slower, steadier read is a
# more stable one) and that approximation is stated rather than hidden. An axis the caller
# did not set leaves the engine default in place.
#
# `energy` maps to NOTHING here, and that is a decision. The first draft sent it to
# `similarity_boost`, which a review caught as very likely INVERTED: that knob is voice
# ADHERENCE, so `high` → 0.85 holds the read CLOSER to its reference — the opposite of "bright
# and animated" — and the one real expressiveness lever (`style`) is already spent on the
# `style` axis. Between a wrong mapping and none, none: an unshaped axis is a preference
# unheard, an inverted one actively fights the delivery the caller asked for. An engine with a
# real energy control can have this line back, with a measurement behind it.
_EL_DEFAULTS = {"stability": 0.5, "similarity_boost": 0.75}
_EL_STYLE = {"warm": 0.30, "empathetic": 0.35, "reserved": 0.0}
_EL_STABILITY = {"slow": 0.70, "fast": 0.35}


def as_voice_settings(profile: Any) -> "dict[str, float]":
    """The ``voice_settings`` payload for a parameter engine — the engine defaults, with only
    the axes the caller set overridden. Returns the defaults unchanged for an empty profile,
    so a caller can always send the result. Coerced for the reason in ``as_instructions``."""
    profile, _ = sanitize_delivery(profile)
    settings: dict[str, float] = dict(_EL_DEFAULTS)
    if profile.style in _EL_STYLE:
        settings["style"] = _EL_STYLE[profile.style]
    if profile.pace in _EL_STABILITY:
        settings["stability"] = _EL_STABILITY[profile.pace]
    return settings
