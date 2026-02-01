#!/usr/bin/env bash
set -euo pipefail

BACKEND=${BACKEND:-nccl}

configs=(
  "dp=4,tp=1,pp=1"
  "dp=1,tp=4,pp=1"
  "dp=1,tp=1,pp=4"
  "dp=2,tp=2,pp=1"
  "dp=2,tp=1,pp=2"
  "dp=1,tp=2,pp=2"
)

printf "\n=== Running gradient parity tests ===\n"

for cfg in "${configs[@]}"; do
  IFS=',' read -r dp_kv tp_kv pp_kv <<< "$cfg"
  dp=${dp_kv#dp=}
  tp=${tp_kv#tp=}
  pp=${pp_kv#pp=}
  nproc=$((dp*tp*pp))

  echo "\n--- cfg: dp=$dp tp=$tp pp=$pp (nproc=$nproc) ---"
  PYTHONPATH="$(pwd)" torchrun --standalone --nproc_per_node=$nproc tests/grad_parity.py \
    --dp $dp --tp $tp --pp $pp --backend $BACKEND

done
