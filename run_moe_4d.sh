#!/bin/bash
# MoE 4D-parallel training: dp=2 ep=2 tp=2 pp=1 (8 GPUs)
# 8 experts, top-2 routing, MoE every 2 layers
# 12L/768D, bf16, 5000 steps

set -euo pipefail
cd "$(dirname "$0")"
source /root/pp/.venv/bin/activate

torchrun --standalone --nproc_per_node=8 train.py \
  --data_dir data/fineweb10B \
  --dp 2 --ep 2 --tp 2 --pp 1 \
  --n_layer 12 --n_head 12 --n_embed 768 \
  --num_experts 8 --top_k 2 --moe_freq 2 \
  --batch_size 8 --grad_accum_steps 4 \
  --max_iters 5000 \
  --learning_rate 3e-4 \
  --eval_interval 500 --eval_iters 50 \
  --save_interval 1000 \
  --block_size 1024 \
  --wandb --wandb_project heiretsu-moe \
  --run_name "moe-8e-top2-4D-dp2ep2tp2-12L768D-5k" \
  --amp bf16
