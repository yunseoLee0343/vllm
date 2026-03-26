# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Step-wise Mamba selective_scan benchmark with scheduler-style chunking.

Simulates vLLM-like scheduling over multiple steps for one request (or small batch),
including:
- token-budget scheduling,
- mamba block-aligned split behavior,
- APC chunk metadata construction,
- selective_scan_fn launches only when aligned tokens are schedulable,
- retry/stall accounting when aligned split returns 0.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_scan_fn
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


@dataclass
class StepStats:
    kernel_calls: int = 0
    total_chunks: int = 0
    total_chunk_tokens: int = 0
    retries: int = 0
    stalled_steps: int = 0
    total_steps: int = 0
    scheduled_tokens: int = 0
    masked_lanes: float = 0.0
    mask_lanes_total: float = 0.0
    state_writes: int = 0
    zero_fill_time: float = 0.0


def mamba_block_aligned_split(
    num_computed_tokens: int,
    num_prompt_tokens: int,
    request_num_tokens: int,
    num_new_tokens: int,
    block_size: int,
    use_eagle: bool,
) -> int:
    """Reimplementation of scheduler._mamba_block_aligned_split behavior."""
    if num_computed_tokens < max(num_prompt_tokens, request_num_tokens - 1):
        last_cache_position = request_num_tokens - request_num_tokens % block_size
        if use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)
        num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
        if num_computed_tokens_after_sched < last_cache_position:
            num_new_tokens = num_new_tokens // block_size * block_size
        elif num_computed_tokens < last_cache_position < num_computed_tokens_after_sched:
            num_new_tokens = last_cache_position - num_computed_tokens
    return num_new_tokens


def build_step_chunks(start: int, n_tokens: int, block_size: int) -> list[int]:
    """Return per-chunk token counts respecting block boundaries."""
    if n_tokens <= 0:
        return []
    chunks: list[int] = []
    pos = start
    remaining = n_tokens
    while remaining > 0:
        boundary = ((pos // block_size) + 1) * block_size
        take = min(boundary - pos, remaining)
        chunks.append(take)
        pos += take
        remaining -= take
    return chunks


def run(args: argparse.Namespace) -> None:
    device = args.device
    dtype = torch.float16
    wtype = torch.float32

    total_tokens = args.seqlen
    batch = args.batch
    dim = args.dim
    dstate = args.dstate

    u = torch.randn(batch, dim, total_tokens, device=device, dtype=dtype)
    delta = torch.rand(batch, dim, total_tokens, device=device, dtype=dtype)
    A = -0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)
    B = torch.randn(batch, dstate, total_tokens, device=device, dtype=dtype)
    C = torch.randn(batch, dstate, total_tokens, device=device, dtype=dtype)
    D = torch.randn(dim, device=device, dtype=torch.float32)
    z = torch.randn(batch, dim, total_tokens, device=device, dtype=dtype)
    delta_bias = torch.rand(dim, device=device, dtype=torch.float32)

    num_blocks = math.ceil(total_tokens / args.block_size) + 2
    cache_indices = (
        torch.arange(num_blocks, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .repeat(batch, 1)
    )

    ssm_states = torch.randn(batch, dim, dstate, device=device, dtype=dtype)
    num_computed = torch.zeros(batch, dtype=torch.int32, device=device)

    stats = StepStats()
    start_time = time.perf_counter()

    while True:
        unfinished = (num_computed < total_tokens).nonzero(as_tuple=False).flatten()
        if len(unfinished) == 0:
            break

        stats.total_steps += 1

        # single common schedule for simplicity (small-batch approximation)
        min_computed = int(num_computed[unfinished].min().item())
        remaining = total_tokens - min_computed
        proposed = min(args.token_budget, remaining)

        aligned = mamba_block_aligned_split(
            num_computed_tokens=min_computed,
            num_prompt_tokens=total_tokens,
            request_num_tokens=total_tokens,
            num_new_tokens=proposed,
            block_size=args.block_size,
            use_eagle=args.use_eagle,
        )

        if aligned == 0:
            stats.retries += 1
            stats.stalled_steps += 1
            continue

        step_start = min_computed
        step_tokens = aligned
        step_end = step_start + step_tokens

        chunks = build_step_chunks(step_start, step_tokens, args.block_size)
        n_chunks = len(chunks)
        cu_chunk = [0]
        for c in chunks:
            cu_chunk.append(cu_chunk[-1] + c)

        block_first = step_start // args.block_size
        block_last = (step_end - 1) // args.block_size
        init_block = max(block_first - 1, 0)

        block_idx_first = torch.full((batch,), block_first, dtype=torch.int32, device=device)
        block_idx_last = torch.full((batch,), block_last, dtype=torch.int32, device=device)
        initial_state_idx = torch.full((batch,), init_block, dtype=torch.int32, device=device)
        cu_chunk_seqlen = torch.tensor(cu_chunk, dtype=torch.int32, device=device)
        last_chunk_indices = torch.full((batch,), n_chunks - 1, dtype=torch.int32, device=device)
        has_initial_state = torch.ones(batch, dtype=torch.bool, device=device)
        if step_start == 0:
            has_initial_state = torch.zeros(batch, dtype=torch.bool, device=device)

        # zero-fill overhead proxy
        zf_start = time.perf_counter()
        _ = torch.zeros(batch, dim, step_tokens, dtype=dtype, device=device)
        torch.cuda.synchronize(device)
        stats.zero_fill_time += time.perf_counter() - zf_start

        torch.cuda.synchronize(device)
        k_start = time.perf_counter()
        _ = selective_scan_fn(
            u[:, :, step_start:step_end],
            ssm_states,
            delta[:, :, step_start:step_end],
            A,
            B[:, :, step_start:step_end],
            C[:, :, step_start:step_end],
            D,
            z=z[:, :, step_start:step_end],
            delta_bias=delta_bias,
            delta_softplus=True,
            query_start_loc=None,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            pad_slot_id=PAD_SLOT_ID,
            block_size=args.block_size,
            block_idx_first_scheduled_token=block_idx_first,
            block_idx_last_scheduled_token=block_idx_last,
            initial_state_idx=initial_state_idx,
            cu_chunk_seqlen=cu_chunk_seqlen,
            last_chunk_indices=last_chunk_indices,
        )
        torch.cuda.synchronize(device)
        _ = time.perf_counter() - k_start

        stats.kernel_calls += 1
        stats.total_chunks += n_chunks
        stats.total_chunk_tokens += sum(chunks)
        stats.scheduled_tokens += step_tokens
        stats.state_writes += n_chunks * dstate * 1  # rows=1

        # masked-lane ratio estimate based on tile size.
        for c in chunks:
            tile = args.tile_size
            masked = max(tile - c, 0)
            stats.masked_lanes += masked
            stats.mask_lanes_total += tile

        num_computed[unfinished] = torch.clamp(
            num_computed[unfinished] + step_tokens,
            max=total_tokens,
        )

    total_runtime = time.perf_counter() - start_time
    avg_chunk_tokens = (
        stats.total_chunk_tokens / stats.total_chunks if stats.total_chunks > 0 else 0.0
    )
    eff_tokens_per_sec = stats.scheduled_tokens / max(total_runtime, 1e-9)
    stall_ratio = stats.stalled_steps / max(stats.total_steps, 1)
    avg_tokens_per_step = stats.scheduled_tokens / max(stats.total_steps, 1)
    useful_compute_ratio = 1.0
    if stats.mask_lanes_total > 0:
        useful_compute_ratio = 1.0 - (stats.masked_lanes / stats.mask_lanes_total)
    cache_writes_per_token = stats.state_writes / max(stats.scheduled_tokens, 1)
    zero_fill_fraction = stats.zero_fill_time / max(total_runtime, 1e-9)

    print("Mamba selective_scan step-wise benchmark")
    print(
        f"seqlen={args.seqlen} dim={args.dim} dstate={args.dstate} "
        f"block_size={args.block_size} token_budget={args.token_budget}"
    )
    print(f"total_runtime_sec={total_runtime:.6f}")
    print(f"effective_tokens_per_sec={eff_tokens_per_sec:.3f}")
    print(f"scheduler_retry_count={stats.retries}")
    print(f"stall_ratio={stall_ratio:.6f}")
    print(f"avg_tokens_per_step={avg_tokens_per_step:.3f}")
    print(f"kernel_calls={stats.kernel_calls}")
    print(f"total_chunks={stats.total_chunks}")
    print(f"avg_chunk_tokens={avg_chunk_tokens:.3f}")
    print(f"estimated_masked_lane_ratio={1.0 - useful_compute_ratio:.6f}")
    print(f"estimated_useful_compute_ratio={useful_compute_ratio:.6f}")
    print(f"cache_writes_per_token={cache_writes_per_token:.6f}")
    print(f"zero_fill_time_fraction={zero_fill_fraction:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dstate", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--token-budget", type=int, default=1536)
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--use-eagle", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run(args)
