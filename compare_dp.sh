#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/root/heiretsu/heiretsu/data/fineweb10B}"
STEPS=${STEPS:-5}
GLOBAL_BS=${GLOBAL_BS:-64}
BLOCK_SIZE=${BLOCK_SIZE:-512}
NPROC=${NPROC:-2}
BACKEND=${BACKEND:-nccl}

single_ckpt="single_ckpt.pt"
dp_ckpt="dp_ckpt.pt"

printf "\n=== Single GPU run ===\n"
python train.py \
  --device auto \
  --data_dir "$DATA_DIR" \
  --batch_size $GLOBAL_BS \
  --block_size $BLOCK_SIZE \
  --max_iters $STEPS \
  --save_interval $STEPS \
  --wandb_mode disabled \
  --run_name single_test

mv "$(printf "ckpt_%07d.pt" "$STEPS")" "$single_ckpt"

printf "\n=== DP run (nproc=%s, backend=%s) ===\n" "$NPROC" "$BACKEND"
torchrun --standalone --nproc_per_node=$NPROC train.py \
  --device auto \
  --dp $NPROC --tp 1 --pp 1 \
  --dist_backend $BACKEND \
  --data_dir "$DATA_DIR" \
  --batch_size $((GLOBAL_BS / NPROC)) \
  --block_size $BLOCK_SIZE \
  --max_iters $STEPS \
  --save_interval $STEPS \
  --wandb_mode disabled \
  --run_name dp_test

mv "$(printf "ckpt_%07d.pt" "$STEPS")" "$dp_ckpt"

printf "\n=== Comparing logits ===\n"
python - <<PY
import torch, numpy as np
from gpt_model import GPT, GPTConfig
from train import RandomBatchLoader

DATA_DIR = "$DATA_DIR"
B = $GLOBAL_BS
T = $BLOCK_SIZE
SEED = 1234

single_ckpt = "$single_ckpt"
dp_ckpt = "$dp_ckpt"

def load_logits(path):
  state = torch.load(path, map_location="cpu")
  cfg = GPTConfig(**state["cfg"])
  model = GPT(cfg)
  model.load_state_dict(state["model"])
  model.eval()
  loader = RandomBatchLoader(DATA_DIR, "val", B, T, num_chunks=1, seed=SEED, rank=0, dp_rank=0, dp_size=1)
  x, _ = loader.next_batch(torch.device("cpu"))
  with torch.no_grad():
    logits, _ = model(x)
  return logits.float().cpu().numpy()

logits_single = load_logits(single_ckpt)
logits_dp = load_logits(dp_ckpt)
np.save("logits_single.npy", logits_single)
np.save("logits_dp.npy", logits_dp)
diff = np.abs(logits_single - logits_dp)
print("logits diff: max", diff.max(), "mean", diff.mean())
PY

printf "\nDone. Outputs: %s, %s, logits_single.npy, logits_dp.npy\n" "$single_ckpt" "$dp_ckpt"
