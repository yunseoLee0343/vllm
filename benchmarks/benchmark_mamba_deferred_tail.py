# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Micro-benchmark for deferred Mamba tail execution.

Measures:
- TTFT proxy (time to first successful allocation)
- token continuity (monotonic progress across steps)
- KV allocation count
"""

from __future__ import annotations

import argparse
import time

import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec


def build_manager(block_size: int, num_blocks: int) -> MambaManager:
    spec = MambaSpec(
        block_size=block_size,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    pool = BlockPool(
        num_gpu_blocks=num_blocks,
        enable_caching=True,
        hash_block_size=block_size,
    )
    return MambaManager(
        spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
    )


def run_workload(prompt_len: int, block_size: int, num_blocks: int, steps: int) -> None:
    manager = build_manager(block_size=block_size, num_blocks=num_blocks)
    request_id = "bench_req"

    # seed one running block to simulate resumed/hybrid behavior
    manager.req_to_blocks[request_id] = manager.block_pool.get_new_blocks(1)
    manager._allocated_block_reqs.add(request_id)

    start = time.perf_counter()
    first_allocation_time = None
    kv_allocation_count = 0
    prev_tokens = 0
    token_continuity_ok = True

    for step in range(steps):
        # long prompt + tail-heavy progression
        scheduled_tokens = min(prompt_len, prev_tokens + block_size - 1)
        new_blocks = manager.allocate_new_blocks(
            request_id=request_id,
            num_tokens=scheduled_tokens,
            num_tokens_main_model=scheduled_tokens,
        )
        if new_blocks:
            kv_allocation_count += len(new_blocks)
            if first_allocation_time is None:
                first_allocation_time = time.perf_counter() - start

        # continuity proxy: progress should be monotonic
        if scheduled_tokens < prev_tokens:
            token_continuity_ok = False
        prev_tokens = scheduled_tokens

    elapsed = time.perf_counter() - start
    ttft = first_allocation_time if first_allocation_time is not None else elapsed

    print("Mamba deferred-tail benchmark")
    print(f"prompt_len={prompt_len} block_size={block_size} num_blocks={num_blocks} steps={steps}")
    print(f"ttft_sec={ttft:.6f}")
    print(f"token_continuity={token_continuity_ok}")
    print(f"kv_allocation_count={kv_allocation_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-len", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()
    run_workload(args.prompt_len, args.block_size, args.num_blocks, args.steps)
