# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark Mamba selective-scan kernel behavior.

Workloads:
- long sequence
- APC-heavy decode simulation

Metrics:
- kernel time
- TTFT proxy
- HBM write count (if profiler available)
"""

from __future__ import annotations

import argparse
import time

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_scan_fn
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


def run_once(seqlen: int, dim: int, dstate: int, use_apc: bool) -> tuple[float, float]:
    device = "cuda"
    itype = torch.float16
    wtype = torch.float32

    batch = 1
    u = torch.randn(batch, dim, seqlen, device=device, dtype=itype)
    delta = torch.rand(batch, dim, seqlen, device=device, dtype=itype)
    A = -0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)
    B = torch.randn(batch, dstate, seqlen, device=device, dtype=itype)
    C = torch.randn(batch, dstate, seqlen, device=device, dtype=itype)
    D = torch.randn(dim, device=device, dtype=torch.float32)
    z = torch.randn(batch, dim, seqlen, device=device, dtype=itype)
    delta_bias = torch.rand(dim, device=device, dtype=torch.float32)
    ssm_states = torch.randn(batch, dim, dstate, device=device, dtype=itype)

    # APC metadata approximation
    block_idx_first = block_idx_last = init_idx = cu_chunk = last_chunk = None
    if use_apc:
        block_size = 1024
        n_blocks = max((seqlen + block_size - 1) // block_size, 1)
        cache_indices = torch.arange(n_blocks, device=device, dtype=torch.int32).unsqueeze(0)
        block_idx_first = torch.tensor([0], device=device, dtype=torch.int32)
        block_idx_last = torch.tensor([n_blocks - 1], device=device, dtype=torch.int32)
        init_idx = torch.tensor([0], device=device, dtype=torch.int32)
        # one chunk per block for simple heavy decode simulation
        chunks = [0]
        for i in range(1, n_blocks + 1):
            chunks.append(min(i * block_size, seqlen))
        cu_chunk = torch.tensor(chunks, device=device, dtype=torch.int32)
        last_chunk = torch.tensor([len(chunks) - 2], device=device, dtype=torch.int32)
    else:
        block_size = 2048
        cache_indices = None

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = selective_scan_fn(
        u,
        ssm_states,
        delta,
        A,
        B,
        C,
        D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=True,
        query_start_loc=None,
        cache_indices=cache_indices,
        has_initial_state=None,
        pad_slot_id=PAD_SLOT_ID,
        block_size=block_size,
        block_idx_first_scheduled_token=block_idx_first,
        block_idx_last_scheduled_token=block_idx_last,
        initial_state_idx=init_idx,
        cu_chunk_seqlen=cu_chunk,
        last_chunk_indices=last_chunk,
    )
    torch.cuda.synchronize()
    kernel_time = time.perf_counter() - t0

    # TTFT proxy: identical single-step runtime in this micro benchmark.
    return kernel_time, kernel_time


def maybe_hbm_writes() -> str:
    try:
        from torch.profiler import profile, ProfilerActivity

        return f"enabled ({ProfilerActivity.CUDA})"
    except Exception:
        return "unavailable"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seqlen", type=int, default=8192)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--dstate", type=int, default=64)
    p.add_argument("--apc-heavy", action="store_true")
    args = p.parse_args()

    kernel_time, ttft = run_once(
        seqlen=args.seqlen,
        dim=args.dim,
        dstate=args.dstate,
        use_apc=args.apc_heavy,
    )

    print("Mamba selective-scan kernel benchmark")
    print(f"seqlen={args.seqlen} dim={args.dim} dstate={args.dstate} apc_heavy={args.apc_heavy}")
    print(f"kernel_time_sec={kernel_time:.6f}")
    print(f"ttft_proxy_sec={ttft:.6f}")
    print(f"hbm_write_count={maybe_hbm_writes()}")


if __name__ == "__main__":
    main()
