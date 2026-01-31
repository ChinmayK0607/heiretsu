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
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group
        self.tp_world_size = tp_world_size if tp_world_size is not None else _tp_world_size(tp_group)
        self.tp_rank = tp_rank if tp_rank is not None else _tp_rank(tp_group)
        if out_features % self.tp_world_size != 0:
            raise ValueError(f"out_features {out_features} not divisible by tp {self.tp_world_size}")
        self.out_per_partition = out_features // self.tp_world_size
        self.weight = nn.Parameter(torch.empty(self.out_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_per_partition))
        else:
            self.bias = None

    def reset_parameters(self, std: float = 0.02) -> None:
        # Initialize from full weight for parity with non-TP initialization.
        full_w = torch.empty(self.out_features, self.in_features, device="cpu")
        nn.init.normal_(full_w, mean=0.0, std=std)
        start = self.tp_rank * self.out_per_partition
        end = start + self.out_per_partition
        shard = full_w[start:end].to(self.weight.device)
        with torch.no_grad():
            self.weight.copy_(shard)
        if self.bias is not None:
            full_b = torch.empty(self.out_features, device="cpu")
            nn.init.zeros_(full_b)
            b_shard = full_b[start:end].to(self.weight.device)
            with torch.no_grad():
                self.bias.copy_(b_shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement column-parallel forward with einops-style reshape
        # x: (B,S,in_features)
        # W_full: (out_features, in_features)
        # W_shard = self.weight: (out_features/TP, in_features)  # already sharded along out_features
        # b_shard (optional): (out_features/TP)
        # 1) flatten: x (B,S,in) -> x2d (B*S, in)
        B,S,in_features = x.shape
        x2d = einops.rearrange(x, 'b s in -> (b s) in')
        W_shard = self.weight
        b_shard = self.bias
        # 2) local matmul: y_shard_2d = x2d @ W_shard.T  # (B*S, out/TP)
        y_shard_2d = einops.einsum(x2d, W_shard, 'bs in, op in -> bs op')
        # 3) add bias shard: y_shard_2d + b_shard  # (B*S, out/TP)
        if b_shard is not None: 
            y_shard_2d += b_shard
        # 4) unflatten: y_shard_2d (B*S, out/TP) -> y_shard (B,S,out/TP)
        y_shard = einops.rearrange(y_shard_2d,'(b s) op -> b s op', b = B, s=S)
        # Expected output: y_shard (B,S,out_features/TP)
        return y_shard
        # raise NotImplementedError("Implement ColumnParallelLinear.forward with einops-style reshape.")


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
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group
        self.tp_world_size = tp_world_size if tp_world_size is not None else _tp_world_size(tp_group)
        self.tp_rank = tp_rank if tp_rank is not None else _tp_rank(tp_group)
        if out_features % self.tp_world_size != 0:
            raise ValueError(f"out_features {out_features} not divisible by tp {self.tp_world_size}")
        self.out_per_partition = out_features // self.tp_world_size
        self.weight = nn.Parameter(torch.empty(3 * self.out_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(3 * self.out_per_partition))
        else:
            self.bias = None

    def reset_parameters(self, std: float = 0.02) -> None:
        # Initialize from full QKV weight for parity with non-TP initialization.
        full_w = torch.empty(3 * self.out_features, self.in_features, device="cpu")
        nn.init.normal_(full_w, mean=0.0, std=std)
        shards = []
        for i in range(3):
            start = i * self.out_features + self.tp_rank * self.out_per_partition
            end = start + self.out_per_partition
            shards.append(full_w[start:end])
        shard = torch.cat(shards, dim=0).to(self.weight.device)
        with torch.no_grad():
            self.weight.copy_(shard)
        if self.bias is not None:
            full_b = torch.empty(3 * self.out_features, device="cpu")
            nn.init.zeros_(full_b)
            b_shards = []
            for i in range(3):
                start = i * self.out_features + self.tp_rank * self.out_per_partition
                end = start + self.out_per_partition
                b_shards.append(full_b[start:end])
            b_shard = torch.cat(b_shards, dim=0).to(self.bias.device)
            with torch.no_grad():
                self.bias.copy_(b_shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same as ColumnParallelLinear forward, but weight/bias already packed as QKV shards.
        # x: (B,S,in_features) -> y_shard (B,S,3*out_features/TP)
        B, S, _ = x.shape
        x2d = einops.rearrange(x, "b s in -> (b s) in")
        y_shard_2d = einops.einsum(x2d, self.weight, "bs in, op in -> bs op")
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
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group
        self.tp_world_size = tp_world_size if tp_world_size is not None else _tp_world_size(tp_group)
        self.tp_rank = tp_rank if tp_rank is not None else _tp_rank(tp_group)
        if in_features % self.tp_world_size != 0:
            raise ValueError(f"in_features {in_features} not divisible by tp {self.tp_world_size}")
        self.in_per_partition = in_features // self.tp_world_size
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

    def reset_parameters(self, std: float = 0.02) -> None:
        # Initialize from full weight for parity with non-TP initialization.
        full_w = torch.empty(self.out_features, self.in_features, device="cpu")
        nn.init.normal_(full_w, mean=0.0, std=std)
        start = self.tp_rank * self.in_per_partition
        end = start + self.in_per_partition
        shard = full_w[:, start:end].to(self.weight.device)
        with torch.no_grad():
            self.weight.copy_(shard)
        if self.bias is not None:
            full_b = torch.empty(self.out_features, device="cpu")
            nn.init.zeros_(full_b)
            with torch.no_grad():
                self.bias.copy_(full_b.to(self.bias.device))

    def forward(self, x: torch.Tensor, input_is_parallel: Optional[bool] = None) -> torch.Tensor:
        # TODO: implement row-parallel forward with einops-style reshape
        # x_full: (B,S,in_features) or x_shard: (B,S,in_features/TP)
        # W_full: (out_features, in_features)
        # W_shard: (out_features, in_features/TP)  # slice along in_features
        # b_full (optional, replicated): (out_features)
        # 1) detect if input is already sharded (in/TP)
        if input_is_parallel is None:
            # Auto-detect: if last dim == in_per_partition, assume sharded
            input_is_parallel = (x.size(-1) == self.in_per_partition)
        
        if input_is_parallel:
            x_shard = x
        else:
            # Split full input along last dim and take this rank's shard
            x_shards = _split_last_dim(x, self.tp_world_size)  # (tp, B, S, in/TP)
            x_shard = x_shards[self.tp_rank]  # (B, S, in/TP)
        B,S,_ = x_shard.shape
        W_shard = self.weight
        # 2) if full, split last dim: x_full (B,S,in) -> x_shards (tp,B,S,in/TP) -> x_shard for this tp_rank
        # 3) flatten: x_shard (B,S,in/TP) -> x2d (B*S, in/TP)
        x_2d = einops.rearrange(x_shard,'b s in -> (b s) in')
        # 4) local matmul: y_partial_2d = x2d @ W_shard.T  # (B*S, out)
        y_partial_2d = einops.einsum(x_2d, W_shard , 'bs in, op in -> bs op')
        # 5) unflatten: y_partial_2d (B*S, out) -> y_partial (B,S,out)
        y_partial = einops.rearrange(y_partial_2d,'(b s) op -> b s op', b = B , s = S)
        # 6) all_reduce(sum,tp): y_partial -> y_full (B,S,out)
        if self.tp_group is not None and dist.is_initialized():
            dist.all_reduce(y_partial, op=dist.ReduceOp.SUM, group=self.tp_group)
        y_full = y_partial
        
        # 7) add bias (replicated): y_full + b_full
        if self.bias is not None:
            y_full = y_full + self.bias
        
        # Expected output: y_full (B,S,out_features)
        return y_full
        # raise NotImplementedError("Implement RowParallelLinear.forward with einops-style reshape.")
