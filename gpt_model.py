#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from tp_linear import ColumnParallelLinear, ColumnParallelLinearQKV, RowParallelLinear
from moe import MoELayer, MoEConfig, make_moe_config

from einops import rearrange


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # GPT-2 vocab
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    # MoE config
    num_experts: int = 0       # 0 = dense model (no MoE)
    top_k: int = 2             # experts per token
    moe_freq: int = 2          # MoE every N layers (0 = all dense)
    aux_loss_coef: float = 0.01  # load balancing loss coefficient


class AttentionTP(nn.Module):
    def __init__(self, cfg: GPTConfig, tp_group=None, tp_rank: int = 0, tp_world_size: int = 1):
        super().__init__()
        assert cfg.n_embed % cfg.n_head == 0
        if cfg.n_embed % tp_world_size != 0:
            raise ValueError(f"n_embed {cfg.n_embed} not divisible by tp {tp_world_size}")
        if cfg.n_head % tp_world_size != 0:
            raise ValueError(f"n_head {cfg.n_head} not divisible by tp {tp_world_size}")
        self.n_embed = cfg.n_embed
        self.n_head = cfg.n_head
        self.tp_world_size = tp_world_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        self.nh_tp = cfg.n_head // tp_world_size
        self.dh = cfg.n_embed // cfg.n_head
        self.h_tp = cfg.n_embed // tp_world_size

        self.c_attn = ColumnParallelLinearQKV(cfg.n_embed, cfg.n_embed, tp_group=tp_group, tp_rank=tp_rank, tp_world_size=tp_world_size)
        self.c_proj = RowParallelLinear(cfg.n_embed, cfg.n_embed, tp_group=tp_group, tp_rank=tp_rank, tp_world_size=tp_world_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.size()
        # qkv_shard = W_qkv(x)  # x (B,S,H) -> qkv_shard (B,S,3*H_tp)
        qkv_shard = self.c_attn(x)
        # qkv_shard: (B,S,3*H_tp) -> (B,S,3,H_tp) -> q,k,v each (B,S,H_tp)
        # Then split heads: (B,S,H_tp) -> (B,nh_tp,S,dh) where nh_tp = H_tp/dh
        # qkv_shard: (B,S,3*H_tp) -> (B,S,3,H_tp) -> q,k,v each (B,S,H_tp)
        # q: (B,S,H_tp) -> (B,nh_tp,S,dh)
        # k: (B,S,H_tp) -> (B,nh_tp,S,dh)
        # v: (B,S,H_tp) -> (B,nh_tp,S,dh)
        d_head = self.dh
        qkv_shard = rearrange(qkv_shard,'b s (t h) -> b s t h',t=3)
        q,k,v = qkv_shard.unbind(dim=2)
        _,_,H_tp = q.size()

        q = rearrange(q,'b s (nh_tp dh) -> b nh_tp s dh',nh_tp = H_tp//d_head, dh = d_head)
        k = rearrange(k,'b s (nh_tp dh) -> b nh_tp s dh',nh_tp = H_tp//d_head, dh = d_head)
        v = rearrange(v,'b s (nh_tp dh) -> b nh_tp s dh',nh_tp = H_tp//d_head, dh = d_head)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,
        )
        y = rearrange (y,'b h s dh -> b s (h dh)')

        # output projection (row-parallel): y_partial (B,S,H) -> all_reduce(sum,tp) -> y_full (B,S,H)
        y = self.c_proj(y, input_is_parallel=True)
        y = self.resid_drop(y)
        return y

class MLPTP(nn.Module):
    def __init__(self, cfg: GPTConfig, tp_group=None, tp_rank: int = 0, tp_world_size: int = 1):
        super().__init__()
        self.c_fc = ColumnParallelLinear(cfg.n_embed, 4 * cfg.n_embed, tp_group=tp_group, tp_rank=tp_rank, tp_world_size=tp_world_size)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = RowParallelLinear(4 * cfg.n_embed, cfg.n_embed, tp_group=tp_group, tp_rank=tp_rank, tp_world_size=tp_world_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fc1 col-parallel: x (B,S,H) -> h (B,S,I_tp)
        x = self.c_fc(x)
        x = self.gelu(x)
        # fc2 row-parallel: h (B,S,I_tp) -> y_partial (B,S,H) -> all_reduce -> y (B,S,H)
        x = self.c_proj(x, input_is_parallel=True)
        x = self.drop(x)
        return x


class BlockTP(nn.Module):
    def __init__(self, cfg: GPTConfig, tp_group=None, tp_rank: int = 0, tp_world_size: int = 1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embed)
        self.attn = AttentionTP(cfg, tp_group=tp_group, tp_rank=tp_rank, tp_world_size=tp_world_size)
        self.ln_2 = nn.LayerNorm(cfg.n_embed)
        self.mlp = MLPTP(cfg, tp_group=tp_group, tp_rank=tp_rank, tp_world_size=tp_world_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (hidden, aux_loss) for consistency with BlockMoE. aux_loss=0 for dense blocks."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        # Dense blocks return zero aux_loss
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x, aux_loss


class BlockMoE(nn.Module):
    """Transformer block with MoE instead of dense MLP."""
    
    def __init__(
        self, 
        cfg: GPTConfig, 
        tp_group=None, 
        tp_rank: int = 0, 
        tp_world_size: int = 1,
        ep_group=None,
        ep_rank: int = 0,
        ep_world_size: int = 1,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embed)
        self.attn = AttentionTP(cfg, tp_group=tp_group, tp_rank=tp_rank, tp_world_size=tp_world_size)
        self.ln_2 = nn.LayerNorm(cfg.n_embed)
        
        # MoE layer replaces MLPTP
        moe_config = make_moe_config(
            hidden_size=cfg.n_embed,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            intermediate_mult=4,
            aux_loss_coef=cfg.aux_loss_coef,
        )
        self.moe = MoELayer(
            config=moe_config,
            ep_group=ep_group,
            ep_rank=ep_rank,
            ep_world_size=ep_world_size,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (hidden, aux_loss)."""
        x = x + self.attn(self.ln_1(x))
        moe_out, aux_loss = self.moe(self.ln_2(x))
        x = x + moe_out
        return x, aux_loss


def make_block(
    cfg: GPTConfig,
    layer_idx: int,
    tp_group=None,
    tp_rank: int = 0,
    tp_world_size: int = 1,
    ep_group=None,
    ep_rank: int = 0,
    ep_world_size: int = 1,
) -> nn.Module:
    """Factory to create either BlockTP (dense) or BlockMoE based on config.
    
    MoE blocks are created for layers where (layer_idx + 1) % moe_freq == 0,
    if num_experts > 0 and moe_freq > 0.
    
    Returns:
        Block that returns (hidden, aux_loss) tuple.
    """
    use_moe = (
        cfg.num_experts > 0 
        and cfg.moe_freq > 0 
        and (layer_idx + 1) % cfg.moe_freq == 0
    )
    
    if use_moe:
        return BlockMoE(
            cfg, 
            tp_group=tp_group, 
            tp_rank=tp_rank, 
            tp_world_size=tp_world_size,
            ep_group=ep_group,
            ep_rank=ep_rank,
            ep_world_size=ep_world_size,
        )
    else:
        return BlockTP(
            cfg, 
            tp_group=tp_group, 
            tp_rank=tp_rank, 
            tp_world_size=tp_world_size,
        )


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig, topo: Optional[object] = None):
        super().__init__()
        self.cfg = cfg
        tp = getattr(topo, "tp", 1) if topo is not None else 1
        tp_rank = getattr(topo, "tp_rank", 0) if topo is not None else 0
        tp_group = getattr(topo, "tp_group", None) if topo is not None else None
        ep = getattr(topo, "ep", 1) if topo is not None else 1
        ep_rank = getattr(topo, "ep_rank", 0) if topo is not None else 0
        ep_group = getattr(topo, "ep_group", None) if topo is not None else None

        self.tr = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embed),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embed),
                h=nn.ModuleList([
                    make_block(
                        cfg, 
                        layer_idx=i,
                        tp_group=tp_group, 
                        tp_rank=tp_rank, 
                        tp_world_size=tp,
                        ep_group=ep_group,
                        ep_rank=ep_rank,
                        ep_world_size=ep,
                    ) 
                    for i in range(cfg.n_layer)
                ]),
                ln_f=nn.LayerNorm(cfg.n_embed),
            )
        )
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)
        # weight tying
        self.tr.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        from moe import Expert
        if isinstance(m, (nn.Linear, ColumnParallelLinear, ColumnParallelLinearQKV, RowParallelLinear)):
            std = 0.02
            if hasattr(m, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.cfg.n_layer) ** -0.5
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                m.reset_parameters(std=std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            logits: [B, T, V]
            loss: Optional cross-entropy loss
            aux_loss: Accumulated MoE auxiliary loss (0 if no MoE blocks)
        """
        B, T = idx.size()
        if T > self.cfg.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.cfg.block_size}")
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tr.wte(idx) + self.tr.wpe(pos)[None, :, :]
        
        # Accumulate aux_loss from all blocks
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for blk in self.tr.h:
            x, aux_loss = blk(x)
            total_aux_loss = total_aux_loss + aux_loss
        
        x = self.tr.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, total_aux_loss
