"""Pipeline parallelism (GPipe-style) with microbatching.

Implements a simple fill/drain schedule with explicit send/recv across PP stages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

from gpt_model import GPTConfig, BlockTP
from topo import rank_from_coords


def _partition_layers(n_layer: int, pp: int) -> List[Tuple[int, int]]:
    if pp <= 0:
        raise ValueError("pp must be >= 1")
    if pp > n_layer:
        raise ValueError(f"pp {pp} > n_layer {n_layer} (would create empty stages)")
    base = n_layer // pp
    rem = n_layer % pp
    sizes = [base + (1 if i < rem else 0) for i in range(pp)]
    ranges = []
    start = 0
    for sz in sizes:
        end = start + sz
        ranges.append((start, end))
        start = end
    return ranges


class StageModule(nn.Module):
    """Local slice of GPT blocks for one PP stage."""

    def __init__(self, cfg: GPTConfig, topo, stage_idx: int, num_stages: int):
        super().__init__()
        self.cfg = cfg
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.is_first = stage_idx == 0
        self.is_last = stage_idx == (num_stages - 1)
        self.tp = topo.tp
        self.tp_rank = topo.tp_rank
        self.tp_group = topo.tp_group

        ranges = _partition_layers(cfg.n_layer, num_stages)
        self.block_start, self.block_end = ranges[stage_idx]
        self.blocks = nn.ModuleList(
            [BlockTP(cfg, tp_group=self.tp_group, tp_rank=self.tp_rank, tp_world_size=self.tp) for _ in range(self.block_start, self.block_end)]
        )

        if self.is_first:
            self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embed)
            self.wpe = nn.Embedding(cfg.block_size, cfg.n_embed)
        else:
            self.wte = None
            self.wpe = None

        if self.is_last:
            self.ln_f = nn.LayerNorm(cfg.n_embed)
            self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)
        else:
            self.ln_f = None
            self.lm_head = None

        # Tie weights only if embeddings and head are on the same stage.
        if self.is_first and self.is_last:
            assert self.wte is not None and self.lm_head is not None
            self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        from tp_linear import ColumnParallelLinear, RowParallelLinear
        if isinstance(m, (nn.Linear, ColumnParallelLinear, RowParallelLinear)):
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

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        if T > self.cfg.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.cfg.block_size}")
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # tok = Wte[idx]  # (B,S,H)
        tok = self.wte(idx)
        # pos = Wpe[arange(S)]  # (S,H)
        pos_emb = self.wpe(pos)[None, :, :]
        # x0 = tok + pos  # (B,S,H)
        x = tok + pos_emb
        return x

    def forward_blocks(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward_head(self, x: torch.Tensor, targets: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


@dataclass
class PipelineState:
    boundaries: List[torch.Tensor]
    recv_inputs: List[torch.Tensor]
    losses: List[torch.Tensor]


class GPipeEngine:
    def __init__(self, stage: StageModule, topo):
        self.stage = stage
        self.topo = topo
        self.pp_rank = topo.pp_rank
        self.pp_size = topo.pp
        self.dp_rank = topo.dp_rank
        self.tp_rank = topo.tp_rank
        self.tp = topo.tp
        self.is_first = stage.is_first
        self.is_last = stage.is_last
        self.prev_rank = None
        self.next_rank = None
        if self.pp_size > 1:
            if self.pp_rank > 0:
                self.prev_rank = rank_from_coords(self.dp_rank, self.pp_rank - 1, self.tp_rank, topo.dp, topo.tp, topo.pp)
            if self.pp_rank < self.pp_size - 1:
                self.next_rank = rank_from_coords(self.dp_rank, self.pp_rank + 1, self.tp_rank, topo.dp, topo.tp, topo.pp)

    def _p2p(self, send_tensor: Optional[torch.Tensor], recv_tensor: Optional[torch.Tensor], peer: Optional[int], tag: int) -> None:
        if peer is None or self.pp_size == 1:
            return
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed not initialized for PP send/recv")
        ops = []
        if send_tensor is not None:
            ops.append(dist.P2POp(dist.isend, send_tensor, peer, tag=tag))
        if recv_tensor is not None:
            ops.append(dist.P2POp(dist.irecv, recv_tensor, peer, tag=tag))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()

    def _fwd_tag(self, m: int) -> int:
        return 1000 + m

    def _bwd_tag(self, m: int) -> int:
        return 2000 + m

    def forward_backward(
        self,
        loader,
        micro_batches: int,
        device: torch.device,
        use_amp: bool,
        amp_dtype: torch.dtype,
        scaler: Optional[torch.cuda.amp.GradScaler],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        state = PipelineState(boundaries=[], recv_inputs=[], losses=[])
        B = batch_size
        S = self.stage.cfg.block_size
        H = self.stage.cfg.n_embed
        act_dtype = amp_dtype if use_amp else torch.float32

        # forward
        for m in range(micro_batches):
            idx, targets = loader.next_batch(device)
            if not self.is_last:
                targets = None
            if not self.is_first:
                # recv_x (B,S,H) from prev stage
                recv_x = torch.empty((B, S, H), device=device, dtype=act_dtype)
                self._p2p(None, recv_x, self.prev_rank, self._fwd_tag(m))
                recv_x.requires_grad_(True)  # recv_x (B,S,H)
            else:
                recv_x = None

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if self.is_first:
                    x = self.stage.embed(idx)
                else:
                    x = recv_x
                x = self.stage.forward_blocks(x)
                if self.is_last:
                    logits, loss = self.stage.forward_head(x, targets)
                else:
                    logits, loss = None, None

            if not self.is_last:
                boundary = x
                # boundary  # (B_micro,S,H) sent to next PP
                self._p2p(boundary.detach(), None, self.next_rank, self._fwd_tag(m))
                state.boundaries.append(boundary)
            if not self.is_first:
                state.recv_inputs.append(recv_x)
            if self.is_last and loss is not None:
                state.losses.append(loss)

        # backward
        for m in reversed(range(micro_batches)):
            if self.is_last:
                loss = state.losses[m]
                # scale loss for microbatching
                loss_mb = loss / micro_batches
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_mb).backward()
                else:
                    loss_mb.backward()
                if not self.is_first:
                    grad_boundary = state.recv_inputs[m].grad
                    # grad_boundary  # (B_micro,S,H) -> send back to prev PP
                    self._p2p(grad_boundary, None, self.prev_rank, self._bwd_tag(m))
            else:
                boundary = state.boundaries[m]
                grad_boundary = torch.empty_like(boundary)
                # grad_boundary  # (B_micro,S,H) recv from next PP
                self._p2p(None, grad_boundary, self.next_rank, self._bwd_tag(m))
                boundary.backward(grad_boundary)
                if not self.is_first:
                    grad_in = state.recv_inputs[m].grad
                    self._p2p(grad_in, None, self.prev_rank, self._bwd_tag(m))

        if self.is_last and state.losses:
            with torch.no_grad():
                loss_total = torch.stack([l.detach() for l in state.losses]).mean()
            return loss_total
        return None

    @torch.no_grad()
    def forward_only(
        self,
        loader,
        micro_batches: int,
        device: torch.device,
        use_amp: bool,
        amp_dtype: torch.dtype,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        B = batch_size
        S = self.stage.cfg.block_size
        H = self.stage.cfg.n_embed
        act_dtype = amp_dtype if use_amp else torch.float32
        losses = []

        for m in range(micro_batches):
            idx, targets = loader.next_batch(device)
            if not self.is_last:
                targets = None
            if not self.is_first:
                recv_x = torch.empty((B, S, H), device=device, dtype=act_dtype)
                self._p2p(None, recv_x, self.prev_rank, self._fwd_tag(m))
            else:
                recv_x = None

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if self.is_first:
                    x = self.stage.embed(idx)
                else:
                    x = recv_x
                x = self.stage.forward_blocks(x)
                if self.is_last:
                    _, loss = self.stage.forward_head(x, targets)
                else:
                    loss = None

            if not self.is_last:
                boundary = x
                self._p2p(boundary.detach(), None, self.next_rank, self._fwd_tag(m))
            if self.is_last and loss is not None:
                losses.append(loss.detach())

        if self.is_last and losses:
            return torch.stack(losses).mean()
        return None
