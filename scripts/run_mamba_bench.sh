#!/usr/bin/env bash
set -euo pipefail

echo "=== BASELINE ==="
export MAMBA_RELAX_TAIL=0
export MAMBA_TRACE_KERNEL=1
.venv/bin/python benchmarks/mamba_no_progress.py > baseline.log

echo "=== RELAXED ==="
export MAMBA_RELAX_TAIL=1
.venv/bin/python benchmarks/mamba_no_progress.py > relaxed.log
