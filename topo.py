"""Topology helpers for DP/TP/PP groups (single-node).

Derived from plan.md: world_size = DP * TP * PP. Rank layout keeps TP contiguous,
then PP, then DP.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch.distributed as dist


@dataclass
class Topology:
    dp: int
    tp: int
    pp: int
    rank: int
    world_size: int
    tp_rank: int
    pp_rank: int
    dp_rank: int
    tp_group: Optional[dist.ProcessGroup]
    pp_group: Optional[dist.ProcessGroup]
    dp_group: Optional[dist.ProcessGroup]

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1 and dist.is_initialized()


def coords_from_rank(rank: int, dp: int, tp: int, pp: int) -> Tuple[int, int, int]:
    tp_rank = rank % tp
    pp_rank = (rank // tp) % pp
    dp_rank = rank // (tp * pp)
    return dp_rank, pp_rank, tp_rank


def rank_from_coords(dp_rank: int, pp_rank: int, tp_rank: int, dp: int, tp: int, pp: int) -> int:
    return dp_rank * (pp * tp) + pp_rank * tp + tp_rank


def build_groups(dp: int, tp: int, pp: int) -> Tuple[Optional[dist.ProcessGroup], Optional[dist.ProcessGroup], Optional[dist.ProcessGroup]]:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None, None, None

    world_size = dist.get_world_size()
    assert world_size == dp * tp * pp, f"world_size={world_size} must equal dp*tp*pp={dp*tp*pp}"
    rank = dist.get_rank()
    dp_rank, pp_rank, tp_rank = coords_from_rank(rank, dp, tp, pp)

    # Build TP groups: same dp, pp; varying tp
    tp_groups: List[Optional[dist.ProcessGroup]] = []
    for d in range(dp):
        for p in range(pp):
            ranks = [rank_from_coords(d, p, t, dp, tp, pp) for t in range(tp)]
            tp_groups.append(dist.new_group(ranks=ranks))
    tp_group = tp_groups[dp_rank * pp + pp_rank]

    # Build PP groups: same dp, tp; varying pp
    pp_groups: List[Optional[dist.ProcessGroup]] = []
    for d in range(dp):
        for t in range(tp):
            ranks = [rank_from_coords(d, p, t, dp, tp, pp) for p in range(pp)]
            pp_groups.append(dist.new_group(ranks=ranks))
    pp_group = pp_groups[dp_rank * tp + tp_rank]

    # Build DP groups: same pp, tp; varying dp
    dp_groups: List[Optional[dist.ProcessGroup]] = []
    for p in range(pp):
        for t in range(tp):
            ranks = [rank_from_coords(d, p, t, dp, tp, pp) for d in range(dp)]
            dp_groups.append(dist.new_group(ranks=ranks))
    dp_group = dp_groups[pp_rank * tp + tp_rank]

    return tp_group, pp_group, dp_group


def init_topology(dp: int, tp: int, pp: int) -> Topology:
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if world_size != dp * tp * pp:
        if world_size == 1:
            # allow single-rank dry run even if requested larger topology
            dp = tp = pp = 1
        else:
            raise ValueError(f"world_size {world_size} != dp*tp*pp {dp*tp*pp}")
    dp_rank, pp_rank, tp_rank = coords_from_rank(rank, dp, tp, pp)
    tp_group, pp_group, dp_group = build_groups(dp, tp, pp)
    return Topology(
        dp=dp,
        tp=tp,
        pp=pp,
        rank=rank,
        world_size=world_size,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        dp_rank=dp_rank,
        tp_group=tp_group,
        pp_group=pp_group,
        dp_group=dp_group,
    )


def cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
