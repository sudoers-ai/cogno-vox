"""
Stub-mode smoke + WER unit tests — guards the bench plumbing in CI without any
audio model (mirrors the sibling cognobench smokes).
"""

from cognobench.cases import CASES
from cognobench.runner import format_report, main, run_bench
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
