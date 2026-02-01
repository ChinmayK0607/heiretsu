#!/usr/bin/env bash
set -euo pipefail

BACKEND=${BACKEND:-nccl}
THRESH=${THRESH:-1e-4}
TRAIN_STEPS=${TRAIN_STEPS:-10}
MICRO_BATCHES=${MICRO_BATCHES:-8}
LOG_DIR=${LOG_DIR:-tests/logs/full_suite_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$LOG_DIR"

# Base configs for 4 GPUs
configs=(
  "dp=4,tp=1,pp=1"
  "dp=1,tp=4,pp=1"
  "dp=1,tp=1,pp=4"
  "dp=2,tp=2,pp=1"
  "dp=2,tp=1,pp=2"
  "dp=1,tp=2,pp=2"
)

# Add full 3D config if 8+ GPUs available
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi --list-gpus | wc -l)
  if [[ $num_gpus -ge 8 ]]; then
    configs+=("dp=2,tp=2,pp=2")
  fi
fi

printf "\n=== FULL SUITE: forward+backward+grad parity+training sanity ===\n"

pass=()
fail=()

for cfg in "${configs[@]}"; do
  IFS=',' read -r dp_kv tp_kv pp_kv <<< "$cfg"
  dp=${dp_kv#dp=}
  tp=${tp_kv#tp=}
  pp=${pp_kv#pp=}
  nproc=$((dp*tp*pp))
  logfile="$LOG_DIR/dp${dp}_tp${tp}_pp${pp}.log"

  printf "\n--- cfg: dp=%s tp=%s pp=%s (nproc=%s) --- [%s]\n" "$dp" "$tp" "$pp" "$nproc" "$(date +%H:%M:%S)" | tee -a "$logfile"

  # Forward + backward sanity (logit diff + backward OK)
  echo "[parallel_sanity] start $(date +%H:%M:%S)" | tee -a "$logfile"
  out=$(PYTHONPATH="$(pwd)" torchrun --standalone --nproc_per_node=$nproc tests/parallel_sanity.py \
    --dp $dp --tp $tp --pp $pp --backend $BACKEND 2>&1 | tee -a "$logfile" /dev/stderr)

  diff_max=$(echo "$out" | grep -m1 "LOGIT_DIFF_MAX" | cut -d'=' -f2)
  bwd_ok=$(echo "$out" | grep -m1 "BACKWARD_OK" | cut -d'=' -f2)

  # Grad parity (prints diffs; no strict threshold here)
  echo "[grad_parity] start $(date +%H:%M:%S)" | tee -a "$logfile"
  PYTHONPATH="$(pwd)" torchrun --standalone --nproc_per_node=$nproc tests/grad_parity.py \
    --dp $dp --tp $tp --pp $pp --backend $BACKEND | tee -a "$logfile"

  # Training sanity (boundary push)
  echo "[boundary_push] start $(date +%H:%M:%S)" | tee -a "$logfile"
  PYTHONPATH="$(pwd)" torchrun --standalone --nproc_per_node=$nproc tests/boundary_push.py \
    --dp $dp --tp $tp --pp $pp --steps $TRAIN_STEPS --micro_batches $MICRO_BATCHES --backend $BACKEND | tee -a "$logfile"

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
printf "\nLogs saved to: %s\n" "$LOG_DIR"
