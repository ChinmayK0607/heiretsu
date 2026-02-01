#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/root/heiretsu/heiretsu/data/fineweb10B}"
B=${B:-4}
T=${T:-32}
SEED=${SEED:-1234}
PP=${PP:-2}
BACKEND=${BACKEND:-nccl}

printf "\n=== PP forward (pp=%s, backend=%s) ===\n" "$PP" "$BACKEND"
DATA_DIR="$DATA_DIR" B="$B" T="$T" SEED="$SEED" OUT="loss_pp.npy" PP="$PP" BACKEND="$BACKEND" \
  torchrun --standalone --nproc_per_node=$PP tests/forward_pp.py

printf "\nDone. Output: loss_pp.npy\n"
