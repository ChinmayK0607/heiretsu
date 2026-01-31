#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/root/heiretsu/heiretsu/data/fineweb10B}"
B=${B:-4}
T=${T:-32}
SEED=${SEED:-1234}
TP=${TP:-2}
BACKEND=${BACKEND:-nccl}

printf "\n=== Single GPU forward ===\n"
DATA_DIR="$DATA_DIR" B="$B" T="$T" SEED="$SEED" OUT="logits_single.npy" \
  python tests/forward_single.py

printf "\n=== TP forward (tp=%s, backend=%s) ===\n" "$TP" "$BACKEND"
DATA_DIR="$DATA_DIR" B="$B" T="$T" SEED="$SEED" OUT="logits_tp.npy" TP="$TP" BACKEND="$BACKEND" \
  torchrun --standalone --nproc_per_node=$TP tests/forward_tp.py

printf "\n=== Comparing logits ===\n"
python - <<PY
import numpy as np
logits_single = np.load("logits_single.npy")
logits_tp = np.load("logits_tp.npy")
diff = np.abs(logits_single - logits_tp)
print("logits diff: max", diff.max(), "mean", diff.mean())
PY

printf "\nDone. Outputs: logits_single.npy, logits_tp.npy\n"
