#!/usr/bin/env bash
# ============================================================================
# Heiretsu Quick Training Test (5 mins)
# Verifies the full training pipeline works before a long run
# ============================================================================
set -euo pipefail

# Quick test settings - small model, few iterations
export N_LAYER=6
export N_HEAD=6
export N_EMBED=384
export BATCH_SIZE=8
export BLOCK_SIZE=256
export GRAD_ACCUM=2
export MAX_ITERS=100
export EVAL_INTERVAL=50
export SAVE_INTERVAL=0
export DROPOUT=0.1
export AMP=bf16
export RUN_NAME="heiretsu-quicktest-$(date +%H%M%S)"
export WANDB_PROJECT="heiretsu-test"

# Use data parallel only for simplicity
export DP=4
export TP=1
export PP=1

echo "Running quick training test (100 iters, ~5 mins)..."
bash "$(dirname "$0")/run_train.sh"
