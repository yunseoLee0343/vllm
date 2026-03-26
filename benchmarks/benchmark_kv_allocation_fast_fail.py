# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark allocator fast-fail behavior under high KV pressure.

Measures:
- allocation attempts per request
- scheduler loop iterations (simulated)
"""

from __future__ import annotations

import argparse
import time

import torch

from vllm.utils.hashing import init_none_hash, sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import Request
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.sampling_params import SamplingParams


def make_kv_cache_config(block_size: int, num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )


def make_request(request_id: str, num_tokens: int, block_size: int) -> Request:
    sampling_params = SamplingParams(max_tokens=1)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        lora_request=None,
        cache_salt=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def run_benchmark(num_requests: int, token_len: int, block_size: int, num_blocks: int) -> None:
    init_none_hash(sha256)
    manager = KVCacheManager(
        kv_cache_config=make_kv_cache_config(block_size, num_blocks),
        max_model_len=8192,
        hash_block_size=block_size,
        enable_caching=True,
        use_eagle=False,
        log_stats=False,
    )

    requests = [make_request(f"r{i}", token_len, block_size) for i in range(num_requests)]

    allocation_attempts = 0
    simulated_scheduler_iterations = 0
    allocated = 0

    t0 = time.perf_counter()
    for req in requests:
        simulated_scheduler_iterations += 1
        allocation_attempts += 1
        blocks = manager.allocate_slots(req, num_new_tokens=token_len)
        if blocks is None:
            continue
        allocated += 1
    dt = time.perf_counter() - t0

    attempts_per_request = allocation_attempts / max(num_requests, 1)
    print("KV fast-fail benchmark")
    print(f"num_requests={num_requests} token_len={token_len} block_size={block_size} num_blocks={num_blocks}")
    print(f"allocated_requests={allocated}")
    print(f"allocation_attempts_per_request={attempts_per_request:.4f}")
    print(f"scheduler_loop_iterations={simulated_scheduler_iterations}")
    print(f"elapsed_sec={dt:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--token-len", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=64)
    args = parser.parse_args()
    run_benchmark(args.num_requests, args.token_len, args.block_size, args.num_blocks)
