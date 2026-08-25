"""
Stub-mode smoke + WER unit tests — guards the bench plumbing in CI without any
audio model (mirrors the sibling cognobench smokes).
"""

from cognobench.cases import CASES
from cognobench.runner import format_report, main, run_bench
from cogno_vox.types import DeliveryProfile
from cognobench.stub_backends import StubSynthesizer, StubTranscriber
from cognobench.wer import normalize, wer


# ── WER metric ───────────────────────────────────────────────────────────
def test_wer_perfect_is_zero():
    assert wer("the cat sat on the mat", "the cat sat on the mat") == 0.0


def test_wer_ignores_case_and_punctuation():
    assert wer("Hello, World!", "hello world") == 0.0


def test_wer_ignores_diacritics():
    assert wer("olá meu amigo", "ola meu amigo") == 0.0


def test_wer_counts_substitution_insertion_deletion():
    assert wer("a b c d", "a x c d") == 0.25          # 1 sub / 4
    assert wer("a b c d", "a b c d e") == 0.25        # 1 ins / 4
    assert wer("a b c d", "a b c") == 0.25            # 1 del / 4


def test_wer_empty_edges():
    assert wer("", "") == 0.0
    assert wer("", "noise") == 1.0
    assert wer("hello world", "") == 1.0              # 2 del / 2


def test_normalize_tokenizes():
    assert normalize("  The  Quick, brown FOX! ") == ["the", "quick", "brown", "fox"]
    assert normalize("") == []


# ── stub round-trip ──────────────────────────────────────────────────────
async def test_stub_round_trip_is_lossless():
    report = await run_bench(StubSynthesizer(), StubTranscriber())
    assert len(report.results) == len(CASES)
    assert report.mean_wer == 0.0
    assert report.empty_outputs == 0
    assert all(r.nbytes > 0 for r in report.results)


async def test_format_report_renders():
    report = await run_bench(StubSynthesizer(), StubTranscriber())
    out = format_report(report)
    assert "mean WER: 0.000" in out
    assert "empty outputs: 0/8" in out


def test_main_stub_gate_passes():
    assert main(["--stub", "--max-wer", "0.0"]) == 0


def test_main_stub_gate_fails_when_impossible():
    # An impossible negative bar → mean WER (0.0) > -0.1 → non-zero exit.
    assert main(["--stub", "--max-wer", "-0.1"]) == 1


def test_main_stub_limit():
    assert main(["--stub", "--limit", "3", "--max-wer", "0.0"]) == 0


# ── delivery dimension: the instrument declares what it MEASURED ──────────

class _ShapingStub(StubSynthesizer):
    """A stub that can be shaped — the only kind whose WER says anything about delivery."""

    delivery_dialect = "instructions"

    async def synthesize_shaped(self, text, delivery):
        self.last_delivery = delivery
        return await self.synthesize(text)


async def test_a_profile_an_engine_cannot_honour_is_reported_as_IGNORED():
    """The trap this guards is specific: a delivery run against a plain engine returns the same
    WER as a run without one, and "the number did not move" reads as "shaping is free" when it
    actually means "nothing was applied". The report has to say which."""
    report = await run_bench(StubSynthesizer(), StubTranscriber(), cases=CASES[:2],
                             delivery=DeliveryProfile(style="warm"))
    assert report.shaped is False
    assert "IGNORED" in format_report(report)
    assert "measures nothing about it" in format_report(report)


async def test_a_profile_the_engine_APPLIES_says_so_and_reaches_it():
    shaping = _ShapingStub()
    profile = DeliveryProfile(style="warm", pace="slow")
    report = await run_bench(shaping, StubTranscriber(), cases=CASES[:2], delivery=profile)
    assert report.shaped is True and shaping.last_delivery == profile
    assert "APPLIED by this engine" in format_report(report)


async def test_no_profile_leaves_the_report_unchanged():
    """A bench that grew a dimension must not change what it printed for everyone else."""
    plain = await run_bench(StubSynthesizer(), StubTranscriber(), cases=CASES[:2])
    assert plain.delivery is None and plain.shaped is False
    assert "delivery:" not in format_report(plain)


def test_the_cli_drops_an_unknown_axis_instead_of_failing_the_run():
    """A typo in `--delivery` must not cost the operator the whole run — the bench is often the
    only thing standing between a voice change and production."""
    assert main(["--stub", "--limit", "1", "--delivery", "style=warm,mood=chirpy"]) == 0


async def test_a_profile_that_renders_to_NOTHING_is_not_reported_as_applied():
    """`--delivery "pace=steady,energy=normal"` is truthy but renders to `""` — the payload is
    byte-identical to a plain run. Reporting APPLIED there is the exact misreading the flag
    exists to prevent: an unmoved WER would read as "shaping is free"."""
    report = await run_bench(_ShapingStub(), StubTranscriber(), cases=CASES[:1],
                             delivery=DeliveryProfile(pace="steady", energy="normal"))
    assert report.shaped is False
    assert "IGNORED" in format_report(report)


def test_the_cli_tolerates_a_space_after_the_comma(capsys):
    """The natural way to type it. It used to yield the key `" pace"`, dropped as unknown and
    logged with a leading space (invisible in a log line) — half the profile silently missing
    while the report claimed APPLIED.

    Driven through `main()` and asserted on the RENDERED report: the first version of this test
    re-implemented the parsing inline, so removing the `.strip()` from the real CLI left it
    green. Testing the button instead of whoever presses it."""
    assert main(["--stub", "--limit", "1", "--delivery", "style=warm, pace=slow"]) == 0
    out = capsys.readouterr().out
    assert "style=warm" in out and "pace=slow" in out, out
