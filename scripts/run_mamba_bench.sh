#!/usr/bin/env bash
set -euo pipefail

export MAMBA_BENCH_MODEL="${MAMBA_BENCH_MODEL:-Qwen/Qwen3.5-0.5B-Instruct}"

echo "=== BASELINE ==="
export MAMBA_RELAX_TAIL=0
export MAMBA_TRACE_KERNEL=1
python3 benchmarks/mamba_no_progress.py > baseline.log

echo "=== RELAXED ==="
export MAMBA_RELAX_TAIL=1
python3 benchmarks/mamba_no_progress.py > relaxed.log
