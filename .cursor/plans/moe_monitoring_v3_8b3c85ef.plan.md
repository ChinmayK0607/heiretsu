---
name: MoE Monitoring v3
overview: Add MoE performance monitoring, PyTorch profiler traces to ddp-traces volume, and checkpoint saving to HuggingFace Hub.
todos:
  - id: moe-return-counts-v3
    content: Update load_balancing_loss to return (aux_loss, counts) tuple
    status: pending
  - id: moe-last-counts
    content: Store last_counts tensor on MoELayer (no .item() calls)
    status: pending
  - id: stats-accumulator
    content: Add MoEStatsAccumulator class for microbatch aggregation
    status: pending
  - id: collect-counts
    content: Add collect_moe_counts helper function
    status: pending
  - id: reduce-dp-counts
    content: Add reduce_counts_dp for DP aggregation
    status: pending
  - id: train-loop-moe
    content: Integrate accumulator into training loop with wandb logging
    status: pending
  - id: profiler-traces
    content: Add PyTorch profiler with JSON trace export to ddp-traces volume
    status: pending
  - id: hf-checkpoint
    content: Add HuggingFace Hub checkpoint upload to TheFireHacker/moe-first-try
    status: pending
  - id: modal-secrets
    content: Add hf-secret to modal_train.py and mount ddp-traces volume
    status: pending
isProject: false
---

# MoE Performance Monitoring v3

Addresses all Codex feedback and adds checkpoint/trace saving.

---

## Issues Fixed from v2


| Issue                                   | Fix                                                              |
| --------------------------------------- | ---------------------------------------------------------------- |
| GPU sync per layer (.item() in forward) | Store only `counts` tensor, compute stats in `collect_moe_stats` |
| Missing Union import                    | Use `Optional` instead of `Union`                                |
| Last microbatch only                    | Accumulate counts across microbatches                            |
| PP stats missing                        | Document limitation (local stage only)                           |
| Traces not saved                        | Mount ddp-traces volume, enable checkpointing                    |


---

## Part 1: MoE Stats (Optimized)

**File:** [moe.py](moe.py)

### 1a. Update load_balancing_loss

```python
from typing import Optional, Tuple

def load_balancing_loss(
    gate_probs: torch.Tensor,
    top_indices: torch.Tensor,
    num_experts: int,
    top_k: int,
    aux_loss_coef: float = 0.01,
    return_counts: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Returns (aux_loss, counts) if return_counts=True, else (aux_loss, None)."""
    # ... existing count computation ...
    
    if return_counts:
        return aux_loss, counts.detach()  # Detach to avoid graph retention
    return aux_loss, None
```

### 1b. Store only counts on MoELayer (no .item() calls)

```python
class MoELayer(nn.Module):
    def __init__(self, ...):
        # ... existing ...
        self.last_counts: Optional[torch.Tensor] = None  # [E] tensor
    
    def forward(self, x):
        # ... existing routing ...
        aux_loss, counts = load_balancing_loss(..., return_counts=True)
        self.last_counts = counts  # Store tensor only, no GPU sync
        # ... rest unchanged ...
        return y, aux_loss
```

---

## Part 2: Collect and Aggregate Stats

**File:** [train.py](train.py)

### 2a. Helper to collect counts from model

```python
def collect_moe_counts(model: nn.Module) -> Optional[torch.Tensor]:
    """Collect and sum counts from all MoE layers. Returns [E] tensor or None."""
    from moe import MoELayer
    
    all_counts = []
    for m in model.modules():
        if isinstance(m, MoELayer) and m.last_counts is not None:
            all_counts.append(m.last_counts)
    
    if not all_counts:
        return None
    
    # Sum across MoE layers -> [E]
    return torch.stack(all_counts).sum(dim=0)
```

### 2b. Accumulator for microbatches

```python
class MoEStatsAccumulator:
    """Accumulates MoE counts across microbatches within a step."""
    
    def __init__(self, num_experts: int, device: torch.device):
        self.counts = torch.zeros(num_experts, device=device)
    
    def add(self, counts: Optional[torch.Tensor]):
        if counts is not None:
            self.counts += counts
    
    def reset(self):
        self.counts.zero_()
    
    def get_stats(self) -> dict:
        c = self.counts.float()
        return {
            "counts": self.counts,
            "load_std": c.std().item(),
            "load_min": c.min().item(),
            "load_max": c.max().item(),
            "load_imbalance": (c.max() / c.min().clamp(min=1)).item(),
        }
```

### 2c. DP reduction

```python
def reduce_counts_dp(counts: torch.Tensor, dp_group) -> torch.Tensor:
    """Sum counts across DP group."""
    if dp_group is None or not dist.is_initialized():
        return counts
    result = counts.clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM, group=dp_group)
    return result
```

### 2d. Usage in training loop

```python
# Before training loop:
moe_accum = MoEStatsAccumulator(args.num_experts, device) if args.num_experts > 0 else None

# Inside training loop, after each microbatch forward:
if moe_accum:
    moe_accum.add(collect_moe_counts(local_module))

# After all microbatches (before optimizer.step):
if moe_accum and is_main and run:
    counts = reduce_counts_dp(moe_accum.counts, topo.dp_group)
    stats = compute_stats_from_counts(counts)
    wandb.log({
        "moe/load_std": stats["load_std"],
        "moe/load_min": stats["load_min"],
        "moe/load_max": stats["load_max"],
        "moe/load_imbalance": stats["load_imbalance"],
    }, step=step)
    
    if step % 100 == 0:
        wandb.log({"moe/expert_load_distribution": wandb.Histogram(counts.cpu().numpy())}, step=step)
    
    moe_accum.reset()
```

---

## Part 3: Checkpoint Saving to Modal Volume

**File:** [modal_train.py](modal_train.py)

### 3a. Mount ddp-traces volume

```python
fineweb_volume = modal.Volume.from_name("fineweb-data")
traces_volume = modal.Volume.from_name("ddp-traces")  # NEW
wandb_secret = modal.Secret.from_name("wandb-secret")

@app.function(
    image=image,
    gpu="a100-40gb:4",
    timeout=3600,
    secrets=[wandb_secret],
    volumes={
        "/data": fineweb_volume,
        "/traces": traces_volume,  # NEW
    },
)
def train():
    # ...
```

### 3b. Add checkpoint args to command

```python
cmd = [
    # ... existing args ...
    "--save_interval=500",  # Save every 500 steps
    "--save_dir=/traces",   # Save to mounted volume
]
```

**File:** [train.py](train.py)

### 3c. Add save_dir argument and use it

```python
p.add_argument("--save_dir", type=str, default=".", help="Directory for checkpoints")

# In checkpoint saving:
if args.save_interval and (step % args.save_interval == 0):
    ckpt_path = os.path.join(args.save_dir, f"ckpt_{step:07d}.pt")
    torch.save({...}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
```

---

## Part 4: Pipeline Parallelism Limitation

**Documented behavior:** With PP > 1, MoE stats only reflect layers on the logging rank (last PP stage). Earlier stage stats are not aggregated.

To get full stats with PP, each rank would need to log separately with prefix `moe/pp{rank}/...`. This is out of scope for v3.

---

## Files Changed


| File           | Changes                                                      |
| -------------- | ------------------------------------------------------------ |
| moe.py         | Return counts from loss fn, store `last_counts` (no .item()) |
| train.py       | Add accumulator, collector, DP reduction, save_dir arg       |
| modal_train.py | Mount ddp-traces volume, add save_interval and save_dir      |


---

## What You'll See

**WandB Dashboard:**

- `moe/load_std`, `moe/load_min`, `moe/load_max`, `moe/load_imbalance`
- `moe/expert_load_distribution` histogram (every 100 steps)

**ddp-traces Volume:**

- `ckpt_0000500.pt`, `ckpt_0001000.pt`, etc.

