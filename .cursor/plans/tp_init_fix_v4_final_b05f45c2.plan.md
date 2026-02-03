---
name: TP Init Fix v4 Final
overview: "Final comprehensive fix: DP+MoE grad sync, per-TP per-layer per-module unique seeds, device-aware RNG, args.seed wiring, pipeline global layer_idx. Addresses all reviewer feedback from v1-v3."
todos:
  - id: fix-dp-grad-sync
    content: "Fix dp.py: skip frozen params, init zero grad for unused before all_reduce"
    status: completed
  - id: revert-global-seed
    content: Revert train.py lines 261-266 to original global seeding
    status: completed
  - id: add-init-seed-config
    content: Add init_seed to GPTConfig dataclass
    status: completed
  - id: wire-args-seed
    content: Wire init_seed=args.seed when constructing GPTConfig in train.py
    status: completed
  - id: update-tp-constructors
    content: Add layer_idx and module_id params to all 3 TP linear classes
    status: completed
  - id: rewrite-reset-params
    content: Rewrite reset_parameters with device-aware unique seed formula
    status: completed
  - id: update-attention
    content: Update AttentionTP to accept layer_idx, pass module_id 0,1
    status: completed
  - id: update-mlp
    content: Update MLPTP to accept layer_idx, pass module_id 2,3
    status: completed
  - id: update-blocks
    content: Update BlockTP, BlockMoE, make_block to pass layer_idx
    status: completed
  - id: update-gpt-init
    content: Update GPT.__init__ to pass layer_idx=i
    status: completed
  - id: update-init-weights
    content: Update _init_weights to pass cfg.init_seed
    status: completed
  - id: update-pipeline
    content: Update pipeline.py with global layer_idx and init_seed
    status: completed
  - id: run-modal-test
    content: Run modal training to verify fixes
    status: completed
isProject: false
---

# TP Initialization Fix v4 (Final)

## All Issues Addressed


| #   | Issue                                          | Solution                                     |
| --- | ---------------------------------------------- | -------------------------------------------- |
| 1   | DP+MoE NCCL timeout                            | Zero grad for unused params, skip frozen     |
| 2   | Global per-TP seeding breaks replicated params | Revert, scope to TP layers only              |
| 3   | Same seed for all TP layers on a rank          | Pass `layer_idx`                             |
| 4   | Same seed for modules within a layer           | Pass `module_id` (0-3)                       |
| 5   | Generator device mismatch                      | `torch.Generator(device=self.weight.device)` |
| 6   | Hardcoded seed ignores CLI                     | Wire `init_seed=args.seed` to GPTConfig      |
| 7   | Pipeline layer collision                       | Use global `block_start + local_idx`         |


---

## Part 1: Fix DP + MoE Grad Sync

**File:** [dp.py](dp.py)

```python
def average_gradients(model: torch.nn.Module, dp_group: Optional[dist.ProcessGroup]) -> None:
    """All-reduce grads across DP group (SUM then /world_size).
    
    For MoE: unused experts get zero grad and receive weight decay.
    This encourages balanced expert utilization.
    """
    if dp_group is None or not dist.is_initialized() or dist.get_world_size(group=dp_group) == 1:
        return
    world = dist.get_world_size(group=dp_group)
    for p in model.parameters():
        if not p.requires_grad:
            continue  # Skip frozen params
        if p.grad is None:
            # MoE experts may be unused on some ranks - init zero grad
            p.grad = torch.zeros_like(p.data)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_group)
        p.grad.div_(world)
```

---

## Part 2: Revert Global Per-TP Seeding

**File:** [train.py](train.py) lines 261-266

Revert to original (same seed for all ranks):

```python
# model init must be identical across DP ranks -> seed with base seed only
torch.manual_seed(args.seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(args.seed)
```

---

## Part 3: Add init_seed to GPTConfig and Wire from args

**File:** [gpt_model.py](gpt_model.py)

Add to GPTConfig:

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    num_experts: int = 0
    top_k: int = 2
    moe_freq: int = 2
    aux_loss_coef: float = 0.01
    init_seed: int = 1337  # NEW: Base seed for TP weight init
```

**File:** [train.py](train.py) - wire args.seed when constructing GPTConfig:

```python
cfg = GPTConfig(
    block_size=args.block_size,
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_embed=args.n_embed,
    dropout=args.dropout,
    num_experts=args.num_experts,
    top_k=args.top_k,
    moe_freq=args.moe_freq,
    aux_loss_coef=args.aux_loss_coef,
    init_seed=args.seed,  # NEW: Wire CLI seed
)
```

---

## Part 4: TP Linear Layers with Unique Seeds

**File:** [tp_linear.py](tp_linear.py)

### 4a. Update constructors to accept layer_idx and module_id

```python
class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_group: Optional[dist.ProcessGroup] = None,
        tp_rank: Optional[int] = None,
        tp_world_size: Optional[int] = None,
        layer_idx: int = 0,    # NEW
        module_id: int = 0,    # NEW
    ) -> None:
        super().__init__()
        # ... existing code ...
        self.layer_idx = layer_idx
        self.module_id = module_id
```

Same for `ColumnParallelLinearQKV` and `RowParallelLinear`.

### 4b. Rewrite reset_parameters with device-aware unique seed

```python
def reset_parameters(self, std: float = 0.02, base_seed: int = 1337) -> None:
    """Initialize local shard with unique seed per TP-rank, layer, and module.
    
    Seed = base_seed + tp_rank * 1000000 + layer_idx * 100 + module_id
    """
    seed = base_seed + self.tp_rank * 1000000 + self.layer_idx * 100 + self.module_id
    
    # Device-aware generator
    device = self.weight.device
    if device.type == "cuda":
        gen = torch.Generator(device=device)
    else:
        gen = torch.Generator()
    gen.manual_seed(seed)
    
    # Initialize local shard directly
    with torch.no_grad():
        self.weight.normal_(mean=0.0, std=std, generator=gen)
    
    if self.bias is not None:
        nn.init.zeros_(self.bias)
```

---

## Part 5: Update Call Sites with module_id

**File:** [gpt_model.py](gpt_model.py)

### Module ID Assignment:


| Call Site          | module_id |
| ------------------ | --------- |
| AttentionTP.c_attn | 0         |
| AttentionTP.c_proj | 1         |
| MLPTP.c_fc         | 2         |
| MLPTP.c_proj       | 3         |


### AttentionTP.**init**:

```python
def __init__(self, cfg: GPTConfig, layer_idx: int, tp_group=None, tp_rank=0, tp_world_size=1):
    super().__init__()
    # ... existing setup ...
    self.c_attn = ColumnParallelLinearQKV(
        cfg.n_embed, cfg.n_embed,
        tp_group=tp_group, tp_rank=tp_rank, tp_world_size=tp_world_size,
        layer_idx=layer_idx, module_id=0,
    )
    self.c_proj = RowParallelLinear(
        cfg.n_embed, cfg.n_embed,
        tp_group=tp_group, tp_rank=tp_rank, tp_world_size=tp_world_size,
        layer_idx=layer_idx, module_id=1,
    )
    # ...
```

### MLPTP.**init**:

```python
def __init__(self, cfg: GPTConfig, layer_idx: int, tp_group=None, tp_rank=0, tp_world_size=1):
    super().__init__()
    self.c_fc = ColumnParallelLinear(
        cfg.n_embed, 4 * cfg.n_embed,
        tp_group=tp_group, tp_rank=tp_rank, tp_world_size=tp_world_size,
        layer_idx=layer_idx, module_id=2,
    )
    self.c_proj = RowParallelLinear(
        4 * cfg.n_embed, cfg.n_embed,
        tp_group=tp_group, tp_rank=tp_rank, tp_world_size=tp_world_size,
        layer_idx=layer_idx, module_id=3,
    )
    # ...
```

### BlockTP.**init** and BlockMoE.**init**:

Accept `layer_idx` parameter, pass to AttentionTP and MLPTP.

### make_block:

```python
def make_block(cfg, layer_idx, tp_group=None, tp_rank=0, tp_world_size=1, ...):
    # Pass layer_idx to BlockTP or BlockMoE
```

### GPT.**init**:

```python
h=nn.ModuleList([
    make_block(cfg, layer_idx=i, tp_group=tp_group, ...)
    for i in range(cfg.n_layer)
]),
```

### _init_weights:

```python
def _init_weights(self, m: nn.Module) -> None:
    if isinstance(m, (ColumnParallelLinear, ColumnParallelLinearQKV, RowParallelLinear)):
        std = 0.02
        if hasattr(m, "NANOGPT_SCALE_INIT"):
            std *= (2 * self.cfg.n_layer) ** -0.5
        m.reset_parameters(std=std, base_seed=self.cfg.init_seed)
    # ... rest unchanged
```

---

## Part 6: Pipeline with Global Layer Index

**File:** [pipeline.py](pipeline.py)

When constructing blocks in StageModule, use global layer index:

```python
# In StageModule.__init__:
for local_idx, global_layer_idx in enumerate(range(block_start, block_end)):
    block = make_block(cfg, layer_idx=global_layer_idx, ...)  # Use global index
    self.blocks.append(block)
```

Also update `_init_weights` to pass `cfg.init_seed`.

---

## Seed Formula Verification

For `base_seed=1337`:


| TP Rank | Layer | Module       | Seed                             |
| ------- | ----- | ------------ | -------------------------------- |
| 0       | 0     | c_attn (0)   | 1337 + 0 + 0 + 0 = 1337          |
| 0       | 0     | c_proj (1)   | 1337 + 0 + 0 + 1 = 1338          |
| 0       | 0     | c_fc (2)     | 1337 + 0 + 0 + 2 = 1339          |
| 0       | 0     | mlp_proj (3) | 1337 + 0 + 0 + 3 = 1340          |
| 0       | 1     | c_attn (0)   | 1337 + 0 + 100 + 0 = 1437        |
| 1       | 0     | c_attn (0)   | 1337 + 1000000 + 0 + 0 = 1001337 |
| 1       | 0     | c_proj (1)   | 1337 + 1000000 + 0 + 1 = 1001338 |


All unique. Different TP ranks, layers, and modules all get different seeds.

---

## Files Changed Summary


| File         | Changes                                                                        |
| ------------ | ------------------------------------------------------------------------------ |
| dp.py        | Skip frozen params, init zero grad for unused                                  |
| train.py     | Revert global seeding, wire init_seed to GPTConfig                             |
| gpt_model.py | Add init_seed to GPTConfig, pass layer_idx and module_id, update _init_weights |
| tp_linear.py | Add layer_idx/module_id to constructors, rewrite reset_parameters              |
| pipeline.py  | Use global layer_idx (block_start + local_idx), pass init_seed                 |


