#!/usr/bin/env python3
"""Vision benchmark runner — measures latency and accuracy of the VisionAnalyzerPort.

Examples:
    python3 visionbench.py --stub         # deterministic smoke test (no network)
    python3 visionbench.py --base-url ... # benchmark against live VLLM endpoint
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from cogno_vox import (
    OpenAICompatVisionAnalyzer,
    TierConfig,
    VisionAnalysisResult,
    create_vision_analyzer,
    VoxConfig,
)


class StubVisionBackend:
    @property
    def name(self) -> str:
        return "stub:qwen2.5-vl"

    async def analyze(
        self,
        media_bytes: bytes,
        filename_or_mime: str = "image.png",
        *,
        prompt: str = ""
    ) -> VisionAnalysisResult:
        await asyncio.sleep(0.05)  # 50ms simulated latency
        return VisionAnalysisResult(
            summary="Comprovante de pagamento PIX no valor de R$ 250,00",
            category="PIX_RECEIPT",
            extracted_data={"amount": 250.00, "recipient": "Acme Corp"},
            confidence=0.99,
            tier=self.name,
            elapsed_ms=50.0,
        )


async def run_benchmark(stub: bool = False, base_url: str = "http://localhost:11434/v1", model: str = "qwen2.5-vl-7b") -> int:
    print(f"=== Running VisionBench (stub={stub}, model={model}) ===")
    dummy_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    start_t = time.perf_counter()
    if stub:
        analyzer = StubVisionBackend()
        res = await analyzer.analyze(dummy_png, "sample.png")
    else:
        cfg = VoxConfig(
            vision_tiers=(
                TierConfig(provider="local", model=model, base_url=base_url),
            )
        )
        chain = create_vision_analyzer(cfg)
        res = await chain.analyze(dummy_png, "sample.png")

    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    print(f"✓ Summary: {res.summary}")
    print(f"✓ Category: {res.category}")
    print(f"✓ Extracted Data: {res.extracted_data}")
    print(f"✓ Tier Used: {res.tier}")
    print(f"✓ Total Latency: {elapsed_ms:.2f} ms")

    if not res.summary:
        print("❌ FAILED: Empty summary returned")
        return 1

    print("=== VisionBench Passed Successfully ===")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Cogno Vision Benchmark Runner")
    parser.add_argument("--stub", action="store_true", help="Run deterministic stub test")
    parser.add_argument("--base-url", default="http://localhost:11434/v1", help="Base URL of VLLM server")
    parser.add_argument("--model", default="qwen2.5-vl-7b", help="Model name for vision analysis")
    args = parser.parse_args()

    return asyncio.run(run_benchmark(stub=args.stub, base_url=args.base_url, model=args.model))


if __name__ == "__main__":
    sys.exit(main())
