"""Data parallel helpers (manual grad averaging).

Notes for future TP/PP integration: batch dim is sharded across DP ranks.
Einops hint: `b -> (dp b_local)` to remember how the global batch is split.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from typing import Optional


def average_gradients(model: torch.nn.Module, dp_group: Optional[dist.ProcessGroup]) -> None:
    """All-reduce grads across DP group (SUM then /world_size).

    Assumes gradients are already unscaled if using AMP. No-op on single rank.
    
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
            # MoE experts may be unused on some ranks - init zero grad to sync all_reduce
            p.grad = torch.zeros_like(p.data)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_group)
        p.grad.div_(world)  # batch dim shard: B -> (dp B_local)
