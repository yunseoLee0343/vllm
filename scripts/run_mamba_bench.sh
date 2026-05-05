#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-state-spaces/mamba-130m}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"

echo "Running baseline (MAMBA_RELAX_TAIL=0)"
MAMBA_RELAX_TAIL=0 \
python benchmarks/mamba_no_progress.py \
  --model "$MODEL" \
  --block-size "$BLOCK_SIZE" | tee baseline.log

echo "Running relaxed tail (MAMBA_RELAX_TAIL=1)"
MAMBA_RELAX_TAIL=1 \
python benchmarks/mamba_no_progress.py \
  --model "$MODEL" \
  --block-size "$BLOCK_SIZE" | tee relaxed.log

echo "Logs written to baseline.log and relaxed.log"
