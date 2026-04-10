# SPDX-License-Identifier: Apache-2.0
"""Small harness for diagnosing Mamba scheduler no-progress behavior."""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable

from vllm import LLM, SamplingParams


def _resolve_ttft_seconds(result) -> float | None:
    metrics = getattr(result, "metrics", None)
    if metrics is None:
        return None
    for key in ("first_token_latency", "time_to_first_token", "ttft"):
        value = getattr(metrics, key, None)
        if value is not None:
            return float(value)
    return None


def run_prompts(llm: LLM, prompts: Iterable[str], max_tokens: int) -> None:
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    for prompt in prompts:
        prompt_start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        elapsed = time.perf_counter() - prompt_start

        output = outputs[0]
        generated = output.outputs[0].text if output.outputs else ""
        ttft = _resolve_ttft_seconds(output)

        print("---")
        print(f"prompt_words={len(prompt.split())}")
        print(f"prompt_chars={len(prompt)}")
        if ttft is None:
            print("ttft_seconds=NA")
        else:
            print(f"ttft_seconds={ttft:.6f}")
        print(f"total_generate_seconds={elapsed:.6f}")
        print(f"output_text={generated!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.getenv("MAMBA_BENCH_MODEL", "state-spaces/mamba-130m-hf"),
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    prompts = [
        "hello " * 64,
        "hello " * 127,
        "hello " * 128,
        "hello " * 129,
    ]

    print(f"model={args.model}")
    print(f"MAMBA_RELAX_TAIL={os.getenv('MAMBA_RELAX_TAIL', '0')}")
    print(f"MAMBA_TRACE_KERNEL={os.getenv('MAMBA_TRACE_KERNEL', '0')}")

    llm = LLM(model=args.model, max_model_len=args.max_model_len)
    run_prompts(llm, prompts, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
