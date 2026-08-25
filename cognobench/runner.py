"""
The round-trip bench harness: synthesize → transcribe → score WER.

Drives the real vox backends via DI (any ``SynthesizerBackend`` + ``TranscriberBackend``).
``--stub`` uses the lossless text↔bytes doubles (CI smoke, WER 0); the default
builds OpenAI-compatible STT/TTS backends from env (a local faster-whisper-server +
Kokoro, or the OpenAI/Groq clouds with keys).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

from cogno_vox.delivery import (
    as_instructions,
    as_voice_settings,
    sanitize_delivery,
    shapes_delivery,
)
from cogno_vox.types import DELIVERY_INSTRUCTIONS, DeliveryProfile

from cognobench.cases import CASES, VoiceCase
from cognobench.stub_backends import StubSynthesizer, StubTranscriber
from cognobench.wer import wer


def _plain_settings() -> "dict[str, float]":
    """What a parameter engine gets with NO profile — the baseline a shaped run must differ
    from to have measured anything."""
    return as_voice_settings(DeliveryProfile())


@dataclass(frozen=True)
class CaseResult:
    text: str
    lang: str
    hypothesis: str
    wer: float
    nbytes: int


@dataclass
class BenchReport:
    results: List[CaseResult]
    # The delivery profile the run asked for, and whether anything was actually SENT. Both are
    # reported because a WER that did not move means two different things: "shaping is free"
    # when the engine shaped, "nothing was measured" when it did not. A review caught this flag
    # lying in the two paths that matter most — the README's own example command (Kokoro
    # satisfies the adapter protocol but ignores `instructions`) and a profile of pure defaults
    # (`pace=steady,energy=normal` renders to "") — both printing APPLIED over an unchanged
    # payload. It now answers "did this run differ from a plain one", which is the only
    # question a reader of the WER is actually asking.
    delivery: "Optional[DeliveryProfile]" = None
    shaped: bool = False

    @property
    def mean_wer(self) -> float:
        return sum(r.wer for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def empty_outputs(self) -> int:
        return sum(1 for r in self.results if not r.hypothesis or r.nbytes == 0)


async def run_bench(
    synthesizer,
    transcriber,
    *,
    cases: Sequence[VoiceCase] = CASES,
    delivery: "Optional[DeliveryProfile]" = None,
) -> BenchReport:
    """Round-trip WER, optionally with a delivery profile applied.

    The question a delivery run answers is NOT "does it sound better" — WER cannot see that.
    It is the one thing that can go wrong and would go unnoticed: **shaping the voice must not
    cost intelligibility.** A profile that pushes an engine into a whisper or a rush is a
    regression the listener pays for, and this is the cheapest place to catch it."""
    shaped = shapes_delivery(synthesizer, delivery) and bool(
        as_instructions(delivery)
        if getattr(synthesizer, "delivery_dialect", "") == DELIVERY_INSTRUCTIONS
        else as_voice_settings(delivery) != _plain_settings())
    results: List[CaseResult] = []
    for case in cases:
        audio = (await synthesizer.synthesize_shaped(case.text, delivery)  # type: ignore[union-attr]
                 if shaped else await synthesizer.synthesize(case.text))
        hyp = await transcriber.transcribe(audio, filename="bench.wav") if audio else ""
        results.append(CaseResult(
            text=case.text, lang=case.lang, hypothesis=hyp,
            wer=round(wer(case.text, hyp), 3), nbytes=len(audio),
        ))
    return BenchReport(results, delivery=delivery, shaped=shaped)


def format_report(report: BenchReport) -> str:
    lines = ["", "  lang  wer    bytes  reference → hypothesis", "  " + "─" * 72]
    for r in report.results:
        lines.append(f"  {r.lang:<4}  {r.wer:>4.2f}  {r.nbytes:>6}  "
                     f"{r.text[:28]!r} → {r.hypothesis[:28]!r}")
    lines.append("  " + "─" * 72)
    lines.append(f"  mean WER: {report.mean_wer:.3f}   empty outputs: {report.empty_outputs}"
                 f"/{len(report.results)}")
    if report.delivery:
        axes = ", ".join(f"{k}={v}" for k, v in
                         (("style", report.delivery.style), ("pace", report.delivery.pace),
                          ("energy", report.delivery.energy)) if v)
        # Said out loud, because "WER did not move" is not a result when nothing was applied.
        lines.append(f"  delivery: {axes}   "
                     + ("APPLIED by this engine" if report.shaped
                        else "IGNORED — this engine cannot shape delivery, so this run"
                             " measures nothing about it"))
    return "\n".join(lines)


def _build_backends(args: argparse.Namespace):
    if args.stub:
        return StubSynthesizer(), StubTranscriber()
    from cogno_vox.synthesizer import OpenAICompatSynthesizer
    from cogno_vox.transcriber import OpenAICompatTranscriber
    synth = OpenAICompatSynthesizer(
        base_url=os.getenv("VOX_TTS_BASE_URL", "http://localhost:8880"),
        model=os.getenv("VOX_TTS_MODEL", "kokoro"),
        api_key=os.getenv("VOX_TTS_API_KEY", ""),
        voice=os.getenv("VOX_TTS_VOICE", "alloy"),
        response_format="wav",
    )
    stt = OpenAICompatTranscriber(
        base_url=os.getenv("VOX_STT_BASE_URL", "http://localhost:8000"),
        model=os.getenv("VOX_STT_MODEL", "Systran/faster-whisper-small"),
        api_key=os.getenv("VOX_STT_API_KEY", ""),
    )
    return synth, stt


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="cogno-vox round-trip WER benchmark")
    parser.add_argument("--stub", action="store_true",
                        help="use the lossless text<->bytes doubles (no network)")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    parser.add_argument("--max-wer", type=float, default=None,
                        help="exit non-zero if mean WER exceeds this (CI gate)")
    parser.add_argument("--delivery", default="",
                        help="apply a delivery profile, e.g. 'style=warm,pace=slow,energy=low'"
                             " — measures whether shaping costs intelligibility")
    args = parser.parse_args(argv)

    cases = CASES[: args.limit] if args.limit else CASES
    delivery, dropped = sanitize_delivery(
        {k.strip(): v.strip()
         for k, v in (pair.split("=", 1) for pair in args.delivery.split(",") if "=" in pair)}
        if args.delivery else {})
    if dropped:
        print(f"  ignored in --delivery: {', '.join(dropped)}", file=sys.stderr)
    synth, stt = _build_backends(args)
    report = asyncio.run(run_bench(synth, stt, cases=cases, delivery=delivery or None))
    print(format_report(report))

    if args.max_wer is not None and report.mean_wer > args.max_wer:
        print(f"  FAIL: mean WER {report.mean_wer:.3f} > max {args.max_wer:.3f}", file=sys.stderr)
        return 1
    return 0
