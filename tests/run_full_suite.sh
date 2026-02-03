#!/usr/bin/env bash
set -euo pipefail

# Change to the heiretsu project root (parent of tests/)
cd "$(dirname "$0")/.."

# Suppress NCCL P2P serialization warning (informational, not an error)
export TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING=0

BACKEND=${BACKEND:-nccl}
THRESH=${THRESH:-1e-4}
TRAIN_STEPS=${TRAIN_STEPS:-10}
MICRO_BATCHES=${MICRO_BATCHES:-8}
NUM_EXPERTS=${NUM_EXPERTS:-8}
TOP_K=${TOP_K:-2}
LOG_DIR=${LOG_DIR:-tests/logs/full_suite_$(date +%Y%m%d_%H%M%S)}
TEST_MOE=${TEST_MOE:-auto}  # auto, 0, or 1

mkdir -p "$LOG_DIR"

# Detect GPU count
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi --list-gpus | wc -l)
else
  num_gpus=0
fi

echo "Detected $num_gpus GPUs"

# Configs: "dp,ep,tp,pp,moe" where moe=0 (dense) or 1 (MoE)
configs=()

# 4-GPU dense configs
if [[ $num_gpus -ge 4 ]]; then
  configs+=(
    "4,1,1,1,0"   # DP only
    "1,1,4,1,0"   # TP only
    "1,1,1,4,0"   # PP only
    "2,1,2,1,0"   # DP + TP
    "2,1,1,2,0"   # DP + PP
    "1,1,2,2,0"   # TP + PP
  )
fi

# 8-GPU configs
if [[ $num_gpus -ge 8 ]]; then
  # Dense 3D
  configs+=(
    "2,1,2,2,0"   # Full 3D (DP + TP + PP)
    "4,1,2,1,0"   # DP(4) + TP(2)
    "2,1,4,1,0"   # DP(2) + TP(4)
  )
  
  # MoE + EP configs (if enabled)
  if [[ "$TEST_MOE" == "auto" || "$TEST_MOE" == "1" ]]; then
    configs+=(
      "2,2,2,1,1"   # DP(2) + EP(2) + TP(2) - 4D MoE
      "4,2,1,1,1"   # DP(4) + EP(2) - MoE with EP
      "1,2,2,2,1"   # EP(2) + TP(2) + PP(2) - MoE 3D
      "2,4,1,1,1"   # DP(2) + EP(4) - heavy EP
    )
  fi
fi

printf "\n=== FULL SUITE: 4D Parallelism (DP+EP+TP+PP) + MoE ===\n"
printf "GPUs: %d  Thresh: %s  MoE Experts: %d  Top-K: %d\n" "$num_gpus" "$THRESH" "$NUM_EXPERTS" "$TOP_K"

pass=()
fail=()

for cfg in "${configs[@]}"; do
  IFS=',' read -r dp ep tp pp moe <<< "$cfg"
  nproc=$((dp*ep*tp*pp))
  
  # Skip if not enough GPUs
  if [[ $nproc -gt $num_gpus ]]; then
    echo "[SKIP] dp=$dp ep=$ep tp=$tp pp=$pp moe=$moe (needs $nproc GPUs)"
    continue
  fi
  
  tag="dp=${dp}_ep=${ep}_tp=${tp}_pp=${pp}"
  [[ $moe -eq 1 ]] && tag="${tag}_moe"
  logfile="$LOG_DIR/${tag}.log"
  
  moe_args=""
  if [[ $moe -eq 1 ]]; then
    moe_args="--num_experts $NUM_EXPERTS --top_k $TOP_K"
  fi

  printf "\n--- cfg: dp=%s ep=%s tp=%s pp=%s moe=%s (nproc=%s) --- [%s]\n" \
    "$dp" "$ep" "$tp" "$pp" "$moe" "$nproc" "$(date +%H:%M:%S)" | tee -a "$logfile"

  # Forward + backward sanity (logit diff + backward OK)
  echo "[parallel_sanity] start $(date +%H:%M:%S)" | tee -a "$logfile"
  out=$(torchrun --standalone --nproc_per_node=$nproc tests/parallel_sanity.py \
    --dp $dp --ep $ep --tp $tp --pp $pp --backend $BACKEND $moe_args 2>&1 | tee -a "$logfile" /dev/stderr)

  diff_max=$(echo "$out" | grep -m1 "LOGIT_DIFF_MAX" | cut -d'=' -f2 || echo "")
  bwd_ok=$(echo "$out" | grep -m1 "BACKWARD_OK" | cut -d'=' -f2 || echo "")

  # Grad parity (prints diffs; no strict threshold here)
  echo "[grad_parity] start $(date +%H:%M:%S)" | tee -a "$logfile"
  torchrun --standalone --nproc_per_node=$nproc tests/grad_parity.py \
    --dp $dp --ep $ep --tp $tp --pp $pp --backend $BACKEND $moe_args 2>&1 | tee -a "$logfile"

  # Training sanity (boundary push)
  echo "[boundary_push] start $(date +%H:%M:%S)" | tee -a "$logfile"
  torchrun --standalone --nproc_per_node=$nproc tests/boundary_push.py \
    --dp $dp --ep $ep --tp $tp --pp $pp --steps $TRAIN_STEPS --micro_batches $MICRO_BATCHES \
    --backend $BACKEND $moe_args 2>&1 | tee -a "$logfile"

  if [[ -z "$diff_max" || -z "$bwd_ok" ]]; then
    fail+=("$tag (missing output)")
    continue
  fi

  cmp=$(python3 -c "print(1 if float('$diff_max') <= float('$THRESH') else 0)" 2>/dev/null || echo "0")

  if [[ "$cmp" == "1" && "$bwd_ok" == "1" ]]; then
    pass+=("$tag (diff=$diff_max)")
  else
    fail+=("$tag (diff=$diff_max, bwd=$bwd_ok)")
  fi

done

printf "\n=== Summary ===\n"
printf "PASS (%d):\n" "${#pass[@]}"
for p in "${pass[@]}"; do echo "  $p"; done
printf "FAIL (%d):\n" "${#fail[@]}"
for f in "${fail[@]}"; do echo "  $f"; done
printf "\nLogs saved to: %s\n" "$LOG_DIR"

# Exit with error if any failures
if [[ ${#fail[@]} -gt 0 ]]; then
  exit 1
fi
