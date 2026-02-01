#!/usr/bin/env bash
set -euo pipefail

THRESH=${THRESH:-1e-4}
BACKEND=${BACKEND:-nccl}

configs=(
  "dp=4,tp=1,pp=1"
  "dp=1,tp=4,pp=1"
  "dp=1,tp=1,pp=4"
  "dp=2,tp=2,pp=1"
  "dp=2,tp=1,pp=2"
  "dp=1,tp=2,pp=2"
)

printf "\n=== Running DP/TP/PP boundary tests ===\n"

pass=()
fail=()

for cfg in "${configs[@]}"; do
  IFS=',' read -r dp_kv tp_kv pp_kv <<< "$cfg"
  dp=${dp_kv#dp=}
  tp=${tp_kv#tp=}
  pp=${pp_kv#pp=}
  nproc=$((dp*tp*pp))

  echo "\n--- cfg: dp=$dp tp=$tp pp=$pp (nproc=$nproc) ---"
  out=$(PYTHONPATH="$(pwd)" torchrun --standalone --nproc_per_node=$nproc tests/parallel_sanity.py \
    --dp $dp --tp $tp --pp $pp --backend $BACKEND 2>&1 | tee /dev/stderr)

  diff_max=$(echo "$out" | grep -m1 "LOGIT_DIFF_MAX" | cut -d'=' -f2)
  bwd_ok=$(echo "$out" | grep -m1 "BACKWARD_OK" | cut -d'=' -f2)

  if [[ -z "$diff_max" || -z "$bwd_ok" ]]; then
    fail+=("dp=$dp tp=$tp pp=$pp (missing output)")
    continue
  fi

  cmp=$(python - <<PY
import math
val=float("$diff_max")
print(1 if val <= float("$THRESH") else 0)
PY
)

  if [[ "$cmp" == "1" && "$bwd_ok" == "1" ]]; then
    pass+=("dp=$dp tp=$tp pp=$pp (diff=$diff_max)")
  else
    fail+=("dp=$dp tp=$tp pp=$pp (diff=$diff_max, bwd=$bwd_ok)")
  fi

done

printf "\n=== Summary ===\n"
printf "PASS (%d):\n" "${#pass[@]}"
for p in "${pass[@]}"; do echo "  $p"; done
printf "FAIL (%d):\n" "${#fail[@]}"
for f in "${fail[@]}"; do echo "  $f"; done
