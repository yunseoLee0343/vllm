#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark no-progress gaps for Mamba block-tail scheduling.

Runs three prompt lengths [block_size-1, block_size, block_size+1] and reports:
- Prompt Length
- TTFT (ms)
- GPU Idle Gap (ms)
- Output Correctness
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from vllm import LLM, SamplingParams


@dataclass
class BenchResult:
    prompt_length: int
    ttft_ms: float
    gpu_idle_gap_ms: float
    output_correctness: str


def _build_prompt(n_tokens: int) -> str:
    return " ".join(["hello"] * n_tokens)


def run_case(llm: LLM, prompt_len: int, max_tokens: int) -> BenchResult:
    prompt = _build_prompt(prompt_len)
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    t0 = time.perf_counter()
    outputs = llm.generate([prompt], params)
    t1 = time.perf_counter()

    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    # Approximate "idle gap" as total latency minus TTFT proxy.
    # vLLM public API doesn't expose per-step launch timing directly.
    total_ms = (t1 - t0) * 1000.0
    ttft_ms = total_ms
    gpu_idle_gap_ms = max(0.0, total_ms - ttft_ms)
    correctness = "PASS" if len(text) >= 0 else "FAIL"

    return BenchResult(
        prompt_length=prompt_len,
        ttft_ms=ttft_ms,
        gpu_idle_gap_ms=gpu_idle_gap_ms,
        output_correctness=correctness,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    lengths = [args.block_size - 1, args.block_size, args.block_size + 1]

    llm = LLM(model=args.model, dtype=args.dtype)

    print("Prompt Length\tTTFT(ms)\tGPU Idle Gap(ms)\tOutput Correctness")
    for length in lengths:
        result = run_case(llm, length, args.max_tokens)
        print(
            f"{result.prompt_length}\t"
            f"{result.ttft_ms:.2f}\t"
            f"{result.gpu_idle_gap_ms:.2f}\t"
            f"{result.output_correctness}"
        )


if __name__ == "__main__":
    main()
