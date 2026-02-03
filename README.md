
wip
=======
Minimal 3D parallelism (DP + TP + PP) in pure PyTorch.

What’s here
- Manual DP gradient averaging (no DDP/FSDP).
- Megatron-style TP (column/row parallel linears) for attention + MLP.
- GPipe-style PP with microbatching (fill/drain schedule).
- Optional AMP (fp16/bf16).

Quick start
- Single GPU:
  python train.py --device auto
- Data parallel:
  torchrun --standalone --nproc_per_node=2 train.py --dp 2
- Tensor parallel:
  torchrun --standalone --nproc_per_node=2 train.py --tp 2
- Pipeline parallel:
  torchrun --standalone --nproc_per_node=2 train.py --pp 2 --grad_accum_steps 4

Notes
- PP uses `--grad_accum_steps` as the microbatch count.
- Embedding/LM-head weight tying is only enabled when pp=1.

TP parity test
- torchrun --standalone --nproc_per_node=2 tests_equiv.py --tp 2

Minimal 3D parallelism (DP + TP + PP) in pure PyTorch.


### Full Test Suite (4 GPUs)

Runs forward parity, backward parity, gradient parity, and training smoke tests across 6 configurations:

```bash
cd heiretsu
source ../.venv/bin/activate
bash tests/run_full_suite.sh
```

Configurations tested:
- `dp=4 tp=1 pp=1` (DP only)
- `dp=1 tp=4 pp=1` (TP only)
- `dp=1 tp=1 pp=4` (PP only)
- `dp=2 tp=2 pp=1` (DP + TP)
- `dp=2 tp=1 pp=2` (DP + PP)
- `dp=1 tp=2 pp=2` (TP + PP)

Optional environment variables:
```bash
THRESH=1e-4          # Logit diff threshold (default: 1e-4)
TRAIN_STEPS=10       # Training steps for boundary_push (default: 10)
MICRO_BATCHES=8      # Microbatches per step (default: 8)
```

### Full 3D Parallelism (8 GPUs)

If 8+ GPUs are available, the test suite automatically includes `dp=2 tp=2 pp=2`:

```bash
# On 8-GPU machine, runs 7 configs including full 3D
bash tests/run_full_suite.sh
```

Or run individual tests manually:

```bash
# Forward + backward sanity
torchrun --standalone --nproc_per_node=8 tests/parallel_sanity.py --dp 2 --tp 2 --pp 2

# Gradient parity
torchrun --standalone --nproc_per_node=8 tests/grad_parity.py --dp 2 --tp 2 --pp 2

# Training smoke test
torchrun --standalone --nproc_per_node=8 tests/boundary_push.py --dp 2 --tp 2 --pp 2 --steps 10 --micro_batches 8
```

### Individual Test Scripts

```bash
# TP parity test (2 GPUs)
torchrun --standalone --nproc_per_node=2 tests/tests_equiv.py --tp 2

# Parallel sanity (any config)
torchrun --standalone --nproc_per_node=4 tests/parallel_sanity.py --dp 2 --tp 2 --pp 1

# Gradient parity vs single-GPU baseline
torchrun --standalone --nproc_per_node=4 tests/grad_parity.py --dp 1 --tp 2 --pp 2

# Training smoke test with loss printing
torchrun --standalone --nproc_per_node=4 tests/boundary_push.py --dp 1 --tp 1 --pp 4 --steps 20
```

### Expected Output

```
=== Summary ===
PASS (6):
  dp=4 tp=1 pp=1 (diff=0.0)
  dp=1 tp=4 pp=1 (diff=1.27e-07)
  dp=1 tp=1 pp=4 (diff=0.0)
  dp=2 tp=2 pp=1 (diff=1.19e-07)
  dp=2 tp=1 pp=2 (diff=0.0)
  dp=1 tp=2 pp=2 (diff=1.19e-07)
FAIL (0):
```

---

## Modal Cloud Training

Run distributed training on Modal with 4x A100 GPUs.

### Prerequisites

1. Install Modal CLI:
```bash
pip install modal
modal setup
```

2. Ensure you have these Modal resources configured:
   - **Volume**: `fineweb-data` (with FineWeb10B data)
   - **Secret**: `wandb-secret` (with `WANDB_API_KEY`)

### Run Training

```bash
cd heiretsu
modal run modal_train.py
```

### Default Configuration

| Setting | Value |
|---------|-------|
| GPUs | 4x A100-40GB |
| Model | GPT-2 Medium (24L/16H/1024D, ~355M params) |
| MoE | 8 experts, top-2, every 2 layers |
| Parallelism | DP=2, TP=2 |
| Batch | 8 per GPU, grad_accum=4 |
| Steps | 2000 |
| Precision | bf16 |

Results are logged to WandB project `heiretsu-moe-training`.

### Customization

Edit `TRAINING_CONFIG` in `modal_train.py` to change:
- Model size (`n_layer`, `n_head`, `n_embed`)
- Parallelism dimensions (`dp`, `tp`, `pp`, `ep`)
- MoE settings (`num_experts`, `top_k`, `moe_freq`)
- Training hyperparameters

### Estimated Cost

~$3-5 for 2000 steps (~25-30 minutes on 4x A100-40GB)
