"""Tensor-parallel linear layers (column / row parallel) using torch.distributed.

Minimal Megatron-style sharding:
- ColumnParallelLinear shards output features (rows of weight).
- RowParallelLinear shards input features (columns of weight) and all-reduces outputs.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import einops


# ========== Autograd-aware collective operations ==========

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Identity in forward, all-reduce in backward.
    
    Used before ColumnParallelLinear to ensure gradients from all TP ranks
    are summed when flowing back through the input.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, tp_group: Optional[dist.ProcessGroup]) -> torch.Tensor:
        ctx.tp_group = tp_group
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        if ctx.tp_group is not None and dist.is_initialized():
            dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        return grad_output, None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce in forward, identity in backward.
    
    Used after RowParallelLinear's local matmul to sum partial results.
    In backward, the gradient flows unchanged (no collective needed).
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, tp_group: Optional[dist.ProcessGroup]) -> torch.Tensor:
        if tp_group is not None and dist.is_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=tp_group)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output, None


def copy_to_model_parallel_region(x: torch.Tensor, tp_group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    """Wrapper for _CopyToModelParallelRegion."""
    return _CopyToModelParallelRegion.apply(x, tp_group)


def reduce_from_model_parallel_region(x: torch.Tensor, tp_group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    """Wrapper for _ReduceFromModelParallelRegion."""
    return _ReduceFromModelParallelRegion.apply(x, tp_group)


# ========== Helper functions ==========

def _tp_world_size(tp_group: Optional[dist.ProcessGroup]) -> int:
    if tp_group is None or not dist.is_initialized():
        return 1
    return dist.get_world_size(group=tp_group)


def _tp_rank(tp_group: Optional[dist.ProcessGroup]) -> int:
    if tp_group is None or not dist.is_initialized():
        return 0
    return dist.get_rank(group=tp_group)


def _split_last_dim(x: torch.Tensor, tp_size: int) -> Tuple[torch.Tensor, ...]:
    # TODO: implement with einops-style split
    # x: (..., in) -> split along last dim -> tp shards each (..., in/TP)
    # Example: x (N, in) -> x_shards (tp, N, in/TP)
    y = einops.rearrange(x,'... (tp d) -> tp ... d', tp = tp_size)
    return y 
    # raise NotImplementedError("Implement _split_last_dim with einops-style split


class ColumnParallelLinear(nn.Module):
    """Linear layer with column-parallel (output feature) sharding.

    Weight is sharded along out_features (rows):
      W_full [out, in] -> W_shard [out/TP, in]
    Bias is sharded the same way (if present).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_group: Optional[dist.ProcessGroup] = None,
        tp_rank: Optional[int] = None,
        tp_world_size: Optional[int] = None,
        layer_idx: int = 0,
        module_id: int = 0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group
        self.tp_world_size = tp_world_size if tp_world_size is not None else _tp_world_size(tp_group)
        self.tp_rank = tp_rank if tp_rank is not None else _tp_rank(tp_group)
        self.layer_idx = layer_idx
        self.module_id = module_id
        if out_features % self.tp_world_size != 0:
            raise ValueError(f"out_features {out_features} not divisible by tp {self.tp_world_size}")
        self.out_per_partition = out_features // self.tp_world_size
        self.weight = nn.Parameter(torch.empty(self.out_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_per_partition))
        else:
            self.bias = None

    def reset_parameters(self, std: float = 0.02, base_seed: int = 1337) -> None:
        """Initialize local shard with unique seed per TP-rank, layer, and module.
        
        Seed = base_seed + tp_rank * 1000000 + layer_idx * 100 + module_id
        This ensures diversity across TP shards (SmolLM3 fix) while maintaining
        uniqueness across layers and modules.
        """
        seed = base_seed + self.tp_rank * 1000000 + self.layer_idx * 100 + self.module_id
        
        # Device-aware generator
        device = self.weight.device
        if device.type == "cuda":
            gen = torch.Generator(device=device)
        else:
            gen = torch.Generator()
        gen.manual_seed(seed)
        
        # Initialize local shard directly (no full matrix allocation)
        with torch.no_grad():
            self.weight.normal_(mean=0.0, std=std, generator=gen)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Column-parallel forward: input is replicated, output is sharded.
        # In backward, we need to all-reduce the input gradient.
        # Use copy_to_model_parallel_region for autograd-aware all-reduce in backward.
        x = copy_to_model_parallel_region(x, self.tp_group)
        
        # x: (B,S,in_features)
        # W_shard = self.weight: (out_features/TP, in_features)
        B, S, in_features = x.shape
        x2d = einops.rearrange(x, 'b s d -> (b s) d')
        W_shard = self.weight
        b_shard = self.bias
        # local matmul: y_shard_2d = x2d @ W_shard.T  # (B*S, out/TP)
        y_shard_2d = einops.einsum(x2d, W_shard, 'bs d, op d -> bs op')
        # add bias shard
        if b_shard is not None: 
            y_shard_2d = y_shard_2d + b_shard
        # unflatten: y_shard_2d (B*S, out/TP) -> y_shard (B,S,out/TP)
        y_shard = einops.rearrange(y_shard_2d, '(b s) op -> b s op', b=B, s=S)
        # Output: y_shard (B,S,out_features/TP)
        return y_shard


class ColumnParallelLinearQKV(nn.Module):
    """Column-parallel linear that shards Q/K/V outputs separately.

    Full weight layout is [Q; K; V] stacked: W_full (3*out, in).
    Each TP rank gets rows:
      Q_shard: rows [0*H + tp*H/TP : 0*H + (tp+1)*H/TP]
      K_shard: rows [1*H + tp*H/TP : 1*H + (tp+1)*H/TP]
      V_shard: rows [2*H + tp*H/TP : 2*H + (tp+1)*H/TP]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_group: Optional[dist.ProcessGroup] = None,
        tp_rank: Optional[int] = None,
        tp_world_size: Optional[int] = None,
        layer_idx: int = 0,
        module_id: int = 0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group
        self.tp_world_size = tp_world_size if tp_world_size is not None else _tp_world_size(tp_group)
        self.tp_rank = tp_rank if tp_rank is not None else _tp_rank(tp_group)
        self.layer_idx = layer_idx
        self.module_id = module_id
        if out_features % self.tp_world_size != 0:
            raise ValueError(f"out_features {out_features} not divisible by tp {self.tp_world_size}")
        self.out_per_partition = out_features // self.tp_world_size
        self.weight = nn.Parameter(torch.empty(3 * self.out_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(3 * self.out_per_partition))
        else:
            self.bias = None

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
        
        # Initialize local shard directly (no full matrix allocation)
        with torch.no_grad():
            self.weight.normal_(mean=0.0, std=std, generator=gen)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Column-parallel QKV forward: input is replicated, output is sharded.
        # Use copy_to_model_parallel_region for autograd-aware all-reduce in backward.
        x = copy_to_model_parallel_region(x, self.tp_group)
        
        # x: (B,S,in_features) -> y_shard (B,S,3*out_features/TP)
        B, S, _ = x.shape
        x2d = einops.rearrange(x, "b s d -> (b s) d")
        y_shard_2d = einops.einsum(x2d, self.weight, "bs d, op d -> bs op")
        if self.bias is not None:
            y_shard_2d = y_shard_2d + self.bias
        y_shard = einops.rearrange(y_shard_2d, "(b s) op -> b s op", b=B, s=S)
        return y_shard


class RowParallelLinear(nn.Module):
    """Linear layer with row-parallel (input feature) sharding.

    Weight is sharded along in_features (columns):
      W_full [out, in] -> W_shard [out, in/TP]
    Input is sharded along last dim, and outputs are all-reduced across TP.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_group: Optional[dist.ProcessGroup] = None,
        tp_rank: Optional[int] = None,
        tp_world_size: Optional[int] = None,
        layer_idx: int = 0,
        module_id: int = 0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group
        self.tp_world_size = tp_world_size if tp_world_size is not None else _tp_world_size(tp_group)
        self.tp_rank = tp_rank if tp_rank is not None else _tp_rank(tp_group)
        self.layer_idx = layer_idx
        self.module_id = module_id
        if in_features % self.tp_world_size != 0:
            raise ValueError(f"in_features {in_features} not divisible by tp {self.tp_world_size}")
        self.in_per_partition = in_features // self.tp_world_size
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

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
        
        # Initialize local shard directly (no full matrix allocation)
        with torch.no_grad():
            self.weight.normal_(mean=0.0, std=std, generator=gen)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, input_is_parallel: Optional[bool] = None) -> torch.Tensor:
        # Row-parallel forward: input is sharded, output is replicated (all-reduced).
        # x_full: (B,S,in_features) or x_shard: (B,S,in_features/TP)
        # W_shard: (out_features, in_features/TP)
        # b_full (optional, replicated): (out_features)
        
        # Detect if input is already sharded
        if input_is_parallel is None:
            input_is_parallel = (x.size(-1) == self.in_per_partition)
        
        if input_is_parallel:
            x_shard = x
        else:
            # Split full input along last dim and take this rank's shard
            x_shards = _split_last_dim(x, self.tp_world_size)  # (tp, B, S, in/TP)
            x_shard = x_shards[self.tp_rank]  # (B, S, in/TP)
        
        B, S, _ = x_shard.shape
        W_shard = self.weight
        
        # flatten: x_shard (B,S,in/TP) -> x2d (B*S, in/TP)
        x_2d = einops.rearrange(x_shard, 'b s d -> (b s) d')
        # local matmul: y_partial_2d = x2d @ W_shard.T  # (B*S, out)
        y_partial_2d = einops.einsum(x_2d, W_shard, 'bs d, op d -> bs op')
        # unflatten: y_partial_2d (B*S, out) -> y_partial (B,S,out)
        y_partial = einops.rearrange(y_partial_2d, '(b s) op -> b s op', b=B, s=S)
        
        # Use autograd-aware all-reduce: forward does sum, backward is identity
        y_full = reduce_from_model_parallel_region(y_partial, self.tp_group)
        
        # add bias (replicated)
        if self.bias is not None:
            y_full = y_full + self.bias
        
        return y_full
