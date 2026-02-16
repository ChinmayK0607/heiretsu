
# heiretsu

Minimal parallel-GPT training playground in pure PyTorch.

`heiretsu` implements composable parallelism for transformer training:

- `DP` (data parallel): manual gradient averaging.
- `TP` (tensor parallel): column/row-sharded linears for attention + MLP.
- `PP` (pipeline parallel): GPipe-style stage split + microbatching.
- `EP` (expert parallel): Mixture-of-Experts expert sharding/routing.
- Mixed precision (`--amp fp16|bf16`) and optional `torch.compile`.

The topology is 4D and uses:

`world_size = dp * ep * tp * pp`

## Project layout

- `train.py` — main trainer and CLI.
- `gpt_model.py` — GPT blocks + optional MoE blocks.
- `topo.py` — 4D process-group topology helpers.
- `tp_linear.py` — tensor-parallel linears.
- `pipeline.py` — stage wrapper + GPipe engine.
- `moe.py`, `ep_comm.py` — MoE routing + expert comms.
- `tests/` — forward/grad parity and smoke tests.

## Setup

```bash
cd heiretsu
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Data

Download FineWeb GPT-2 token bins:

```bash
python data/data_bins.py 1
```

This downloads validation and 1 train shard into `data/fineweb10B`.
Increase the argument for more training shards.

## Basic usage examples

### 1) Single-device sanity run

```bash
python train.py \
  --device auto \
  --data_dir data/fineweb10B \
  --max_iters 50 \
  --batch_size 8 \
  --block_size 256
```

### 2) 4-GPU data parallel (DP)

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data_dir data/fineweb10B \
  --dp 4 --ep 1 --tp 1 --pp 1 \
  --batch_size 8 --grad_accum_steps 2 --amp bf16
```

### 3) 4-GPU mixed parallel (DP + TP)

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data_dir data/fineweb10B \
  --dp 2 --ep 1 --tp 2 --pp 1 \
  --batch_size 8 --grad_accum_steps 2 --amp bf16
```

### 4) 8-GPU 4D with MoE (DP + EP + TP)

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --data_dir data/fineweb10B \
  --dp 2 --ep 2 --tp 2 --pp 1 \
  --num_experts 8 --top_k 2 --moe_freq 2 --aux_loss_coef 0.01 \
  --batch_size 8 --grad_accum_steps 4 --amp bf16
```

### 5) Quick smoke run script

```bash
bash run_train_quick.sh
```

## Useful flags

- `--dp`, `--ep`, `--tp`, `--pp`: parallelism degrees.
- `--grad_accum_steps`: microbatch count (especially important for PP).
- `--num_experts`, `--top_k`, `--moe_freq`: enable/configure MoE.
- `--wandb`: optional experiment logging.
- `--compile`: use `torch.compile`.
- `--dist_backend gloo`: useful for CPU-only debugging.

## Testing

Run the full parallel test suite:

```bash
bash tests/run_full_suite.sh
```

Run a single TP parity check:

```bash
torchrun --standalone --nproc_per_node=2 tests/tests_equiv.py --tp 2
```

## Current features

- Composable 4D process topology (`DP/EP/TP/PP`).
- GPT training loop with accumulation + AMP.
- MoE expert routing, load-balancing aux loss, and EP comm path.
- Manual distributed control path for learning/debugging.
- Parity and smoke tests for major parallel configurations.

## TODO / future extensions

- Activation checkpointing for deeper models.
- ZeRO-style optimizer/state sharding.
- Better checkpoint format for resuming across topology changes.
- `PP > 1` schedules beyond simple GPipe fill/drain (e.g., 1F1B).
- CUDA graph capture and fused kernels for throughput.
- Config files (YAML/TOML) + launch presets.
- Multi-node launcher support and networking docs.
- Richer monitoring dashboards (per-rank throughput, comm overlap).

## Notes

- For best performance, use CUDA + NCCL.
- Keep `nproc_per_node == dp*ep*tp*pp`.
- `wandb` is optional; training runs without it.
