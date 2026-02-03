#!/usr/bin/env bash
# ============================================================================
# Heiretsu Large Training Run
# 4-GPU training with FineWeb10B data and wandb logging
# ============================================================================
set -euo pipefail

# ------------------------------------------------------------------------
# WandB Configuration
# ------------------------------------------------------------------------
WANDB_API_KEY="2ed1829df7a84ddc3ff3ce6dc6e711807ef0051e"
SAVE_PATH="${SAVE_PATH:-./checkpoints}"
mkdir -p "$SAVE_PATH"

if [ "$WANDB_API_KEY" != "None" ]; then
    wandb login --relogin $WANDB_API_KEY --host=https://microsoft-research.wandb.io
    export WANDB_DIR=${SAVE_PATH}
fi

# ------------------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------------------

# Parallelism: 4 GPUs
# Option 1: DP=4 (data parallel only - simplest, good for 4 GPUs)
# Option 2: DP=2, TP=2 (2-way data parallel, 2-way tensor parallel)
# Option 3: DP=2, PP=2 (2-way data parallel, 2-way pipeline parallel)
DP=${DP:-4}
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-1}

# MoE Configuration (set NUM_EXPERTS=0 for dense model)
NUM_EXPERTS=${NUM_EXPERTS:-0}
TOP_K=${TOP_K:-2}
MOE_FREQ=${MOE_FREQ:-2}
AUX_LOSS_COEF=${AUX_LOSS_COEF:-0.01}

# Model size: GPT-2 Medium (~355M params)
N_LAYER=${N_LAYER:-24}
N_HEAD=${N_HEAD:-16}
N_EMBED=${N_EMBED:-1024}
DROPOUT=${DROPOUT:-0.1}

# Training hyperparameters
BATCH_SIZE=${BATCH_SIZE:-8}              # Per-GPU micro-batch size
BLOCK_SIZE=${BLOCK_SIZE:-1024}           # Sequence length
GRAD_ACCUM=${GRAD_ACCUM:-4}              # Gradient accumulation steps
MAX_ITERS=${MAX_ITERS:-50000}            # Total training iterations
LEARNING_RATE=${LEARNING_RATE:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
GRAD_CLIP=${GRAD_CLIP:-1.0}

# AMP (mixed precision) - use bf16 on A100s
AMP=${AMP:-bf16}

# Evaluation
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_ITERS=${EVAL_ITERS:-100}

# Checkpointing
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}

# Data
DATA_DIR=${DATA_DIR:-data/fineweb10B}

# Logging
RUN_NAME=${RUN_NAME:-"heiretsu-gpt2m-4gpu-$(date +%Y%m%d_%H%M%S)"}
WANDB_PROJECT=${WANDB_PROJECT:-"heiretsu-fineweb"}

# Compute effective batch size
WORLD_SIZE=$((DP * TP * PP * EP))
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM * DP))

echo "============================================================================"
echo "Heiretsu Training Run: ${RUN_NAME}"
echo "============================================================================"
echo ""
echo "Parallelism: DP=${DP} TP=${TP} PP=${PP} EP=${EP} (world_size=${WORLD_SIZE})"
echo "Model: ${N_LAYER}L ${N_HEAD}H ${N_EMBED}D (dropout=${DROPOUT})"
if [[ $NUM_EXPERTS -gt 0 ]]; then
    echo "MoE: ${NUM_EXPERTS} experts, top-${TOP_K}, every ${MOE_FREQ} layers"
fi
echo "Batch: ${BATCH_SIZE} x ${GRAD_ACCUM} accum x ${DP} DP = ${EFFECTIVE_BATCH} effective"
echo "Sequence length: ${BLOCK_SIZE}"
echo "Training: ${MAX_ITERS} iters, lr=${LEARNING_RATE}, wd=${WEIGHT_DECAY}"
echo "AMP: ${AMP}"
echo "Data: ${DATA_DIR}"
echo "Checkpoints: ${SAVE_PATH} (every ${SAVE_INTERVAL} iters)"
echo "WandB: ${WANDB_PROJECT} / ${RUN_NAME}"
echo "============================================================================"
echo ""

# Build MoE args if enabled
MOE_ARGS=""
if [[ $NUM_EXPERTS -gt 0 ]]; then
    MOE_ARGS="--num_experts ${NUM_EXPERTS} --top_k ${TOP_K} --moe_freq ${MOE_FREQ} --aux_loss_coef ${AUX_LOSS_COEF}"
fi

# Suppress NCCL P2P warning
export TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING=0

# Run training
cd "$(dirname "$0")"

torchrun --standalone --nproc_per_node=${WORLD_SIZE} train.py \
    --data_dir "${DATA_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --block_size ${BLOCK_SIZE} \
    --n_layer ${N_LAYER} \
    --n_head ${N_HEAD} \
    --n_embed ${N_EMBED} \
    --dropout ${DROPOUT} \
    --max_iters ${MAX_ITERS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --grad_clip ${GRAD_CLIP} \
    --grad_accum_steps ${GRAD_ACCUM} \
    --amp ${AMP} \
    --dp ${DP} \
    --tp ${TP} \
    --pp ${PP} \
    --ep ${EP} \
    ${MOE_ARGS} \
    --eval_interval ${EVAL_INTERVAL} \
    --eval_iters ${EVAL_ITERS} \
    --save_interval ${SAVE_INTERVAL} \
    --wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_mode online \
    --run_name "${RUN_NAME}" \
    --seed 1337

echo ""
echo "============================================================================"
echo "Training complete!"
echo "============================================================================"
