"""Expert Parallelism communication helpers (All-to-All dispatch/combine).

Handles token routing across EP ranks for MoE layers.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist


def all_to_all_dispatch(
    x: torch.Tensor,
    send_counts: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup],
    ep_world_size: int,
    payload: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Send tokens to EP ranks owning target experts.
    
    Args:
        x: [N_local, H] tokens sorted by target EP rank
        send_counts: [EP] how many tokens to send to each EP rank
        ep_group: ProcessGroup for EP communication
        ep_world_size: number of EP ranks
        payload: optional 1D tensor (e.g., expert ids) to be sent with identical splits
        
    Returns:
        x_recv: [N_recv, H] tokens received from all EP ranks
        recv_counts: [EP] how many tokens received from each EP rank
        payload_recv: payload tensor received (None if payload is None)
    
    Shape flow:
        x_local: (N, H) -> all_to_all(send_counts, ep_group) -> x_recv: (N', H)
    """
    if ep_group is None or ep_world_size == 1:
        return x, send_counts, payload
    
    device = x.device
    H = x.size(-1)
    
    # Exchange counts first so each rank knows how much to receive
    # recv_counts: [EP] <- all_to_all of send_counts
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=ep_group)
    
    # Prepare send/recv splits
    send_splits = send_counts.tolist()
    recv_splits = recv_counts.tolist()
    
    # Total tokens to receive
    N_recv = int(recv_counts.sum().item())
    
    # Allocate receive buffer
    x_recv = torch.empty((N_recv, H), dtype=x.dtype, device=device)
    
    # All-to-All for the actual tokens
    # x is already sorted by destination EP rank, split into send_splits chunks
    dist.all_to_all_single(
        x_recv,
        x,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
        group=ep_group,
    )

    payload_recv = None
    if payload is not None:
        payload_recv = torch.empty((N_recv,) + payload.shape[1:], dtype=payload.dtype, device=device)
        dist.all_to_all_single(
            payload_recv,
            payload,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=ep_group,
        )
    
    return x_recv, recv_counts, payload_recv


def all_to_all_combine(
    y: torch.Tensor,
    send_counts: torch.Tensor,
    recv_counts: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup],
    ep_world_size: int,
) -> torch.Tensor:
    """
    Receive expert outputs back from EP ranks.
    
    This is the reverse of dispatch: we send back the processed tokens
    to their original EP ranks.
    
    Args:
        y: [N_recv, H] processed tokens (output from local experts)
        send_counts: [EP] original send_counts from dispatch (now recv counts)
        recv_counts: [EP] original recv_counts from dispatch (now send counts)
        ep_group: ProcessGroup for EP communication
        ep_world_size: number of EP ranks
        
    Returns:
        y_combined: [N_local, H] tokens back in original order
        
    Shape flow:
        y_local: (N', H) -> all_to_all(recv_counts, ep_group) -> y: (N, H)
    """
    if ep_group is None or ep_world_size == 1:
        return y
    
    device = y.device
    H = y.size(-1)
    
    # Now we reverse: what we received before, we send back
    # What we sent before, we now receive
    send_splits = recv_counts.tolist()  # we send back what we received
    recv_splits = send_counts.tolist()  # we receive what we originally sent
    
    # Total tokens to receive (should match original N_local)
    N_local = int(send_counts.sum().item())
    
    # Allocate receive buffer
    y_combined = torch.empty((N_local, H), dtype=y.dtype, device=device)
    
    # All-to-All to send results back
    dist.all_to_all_single(
        y_combined,
        y,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
        group=ep_group
    )
    
    return y_combined


def compute_ep_send_counts(
    expert_indices: torch.Tensor,
    num_experts: int,
    ep_world_size: int,
    ep_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute how many tokens to send to each EP rank based on expert assignments.
    
    Args:
        expert_indices: [N*K] flattened expert indices (sorted)
        num_experts: total number of experts
        ep_world_size: number of EP ranks
        ep_rank: this rank's EP index
        
    Returns:
        send_counts: [EP] number of tokens to send to each EP rank
        ep_indices: [N*K] which EP rank owns each expert
        
    Shape flow:
        expert_indices: (N*K,) -> ep_indices: (N*K,) mapping expert_id -> ep_rank
    """
    experts_per_rank = num_experts // ep_world_size
    
    # Map each expert to its EP rank
    # ep_indices: (N*K,) where ep_indices[i] = expert_indices[i] // experts_per_rank
    ep_indices = expert_indices // experts_per_rank
    
    # Count tokens going to each EP rank
    # send_counts: [EP] 
    send_counts = torch.zeros(ep_world_size, dtype=torch.long, device=expert_indices.device)
    send_counts.scatter_add_(0, ep_indices, torch.ones_like(ep_indices))
    
    return send_counts, ep_indices
