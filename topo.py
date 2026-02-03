"""Topology helpers for DP/EP/TP/PP groups (single-node).

Rank layout keeps TP contiguous, then PP, then EP, then DP:
global_rank = dp*(EP*PP*TP) + ep*(PP*TP) + pp*TP + tp
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch.distributed as dist


@dataclass
class Topology:
    dp: int
    ep: int
    tp: int
    pp: int
    rank: int
    world_size: int
    dp_rank: int
    ep_rank: int
    pp_rank: int
    tp_rank: int
    tp_group: Optional[dist.ProcessGroup]
    pp_group: Optional[dist.ProcessGroup]
    ep_group: Optional[dist.ProcessGroup]
    dp_group: Optional[dist.ProcessGroup]

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1 and dist.is_initialized()


def coords_from_rank(rank: int, dp: int, ep: int, tp: int, pp: int) -> Tuple[int, int, int, int]:
    tp_rank = rank % tp
    pp_rank = (rank // tp) % pp
    ep_rank = (rank // (tp * pp)) % ep
    dp_rank = rank // (tp * pp * ep)
    return dp_rank, ep_rank, pp_rank, tp_rank


def rank_from_coords(dp_rank: int, ep_rank: int, pp_rank: int, tp_rank: int, dp: int, ep: int, tp: int, pp: int) -> int:
    return dp_rank * (ep * pp * tp) + ep_rank * (pp * tp) + pp_rank * tp + tp_rank


def build_groups(dp: int, ep: int, tp: int, pp: int) -> Tuple[Optional[dist.ProcessGroup], Optional[dist.ProcessGroup], Optional[dist.ProcessGroup], Optional[dist.ProcessGroup]]:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None, None, None, None

    world_size = dist.get_world_size()
    assert world_size == dp * ep * tp * pp, f"world_size={world_size} must equal dp*ep*tp*pp={dp*ep*tp*pp}"
    rank = dist.get_rank()
    dp_rank, ep_rank, pp_rank, tp_rank = coords_from_rank(rank, dp, ep, tp, pp)

    # TP groups: same dp, ep, pp; varying tp
    tp_groups: List[Optional[dist.ProcessGroup]] = []
    for d in range(dp):
        for e in range(ep):
            for p in range(pp):
                ranks = [rank_from_coords(d, e, p, t, dp, ep, tp, pp) for t in range(tp)]
                tp_groups.append(dist.new_group(ranks=ranks))
    tp_group = tp_groups[(dp_rank * ep * pp) + (ep_rank * pp) + pp_rank]

    # EP groups: same dp, pp, tp; varying ep
    ep_groups: List[Optional[dist.ProcessGroup]] = []
    for d in range(dp):
        for p in range(pp):
            for t in range(tp):
                ranks = [rank_from_coords(d, e, p, t, dp, ep, tp, pp) for e in range(ep)]
                ep_groups.append(dist.new_group(ranks=ranks))
    ep_group = ep_groups[(dp_rank * pp * tp) + (pp_rank * tp) + tp_rank]

    # PP groups: same dp, ep, tp; varying pp
    pp_groups: List[Optional[dist.ProcessGroup]] = []
    for d in range(dp):
        for e in range(ep):
            for t in range(tp):
                ranks = [rank_from_coords(d, e, p, t, dp, ep, tp, pp) for p in range(pp)]
                pp_groups.append(dist.new_group(ranks=ranks))
    pp_group = pp_groups[(dp_rank * ep * tp) + (ep_rank * tp) + tp_rank]

    # DP groups: same ep, pp, tp; varying dp
    dp_groups: List[Optional[dist.ProcessGroup]] = []
    for e in range(ep):
        for p in range(pp):
            for t in range(tp):
                ranks = [rank_from_coords(d, e, p, t, dp, ep, tp, pp) for d in range(dp)]
                dp_groups.append(dist.new_group(ranks=ranks))
    dp_group = dp_groups[(ep_rank * pp * tp) + (pp_rank * tp) + tp_rank]

    return tp_group, pp_group, ep_group, dp_group


def init_topology(dp: int, ep: int = 1, tp: int = 1, pp: int = 1) -> Topology:
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if world_size != dp * ep * tp * pp:
        if world_size == 1:
            # allow single-rank dry run even if requested larger topology
            dp = ep = tp = pp = 1
        else:
            raise ValueError(f"world_size {world_size} != dp*ep*tp*pp {dp*ep*tp*pp}")
    dp_rank, ep_rank, pp_rank, tp_rank = coords_from_rank(rank, dp, ep, tp, pp)
    tp_group, pp_group, ep_group, dp_group = build_groups(dp, ep, tp, pp)
    return Topology(
        dp=dp,
        ep=ep,
        tp=tp,
        pp=pp,
        rank=rank,
        world_size=world_size,
        dp_rank=dp_rank,
        ep_rank=ep_rank,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        tp_group=tp_group,
        pp_group=pp_group,
        ep_group=ep_group,
        dp_group=dp_group,
    )


def cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
