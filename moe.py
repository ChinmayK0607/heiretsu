"""Mixture of Experts (MoE) layer with Expert Parallelism support.

Implements:
- Top-K token routing with softmax gating
- Load balancing auxiliary loss
- Expert Parallelism via All-to-All dispatch/combine
- Permutation-based dispatch for clean autograd

Shape annotation convention follows einops style (see extension_plan.md Section 10).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from ep_comm import all_to_all_dispatch, all_to_all_combine
from einops import rearrange, repeat

@dataclass
class MoEConfig:
    """Configuration for MoE layer."""
    num_experts: int = 8          # E: total number of experts
    top_k: int = 2                # K: experts per token
    hidden_size: int = 768        # H: model hidden dimension
    intermediate_size: int = 3072 # I: expert intermediate dimension (default 4*H)
    aux_loss_coef: float = 0.01   # α: load balancing loss coefficient


class Router(nn.Module):
    """Top-K gating network for token-to-expert routing.
    
    Routes each token to top_k experts based on learned gating weights.
    """
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # W_gate: [H, E] - projects hidden states to expert logits
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            x: [N, H] flattened token representations
            
        Returns:
            top_indices: [N, K] which experts each token is routed to
            top_weights: [N, K] normalized routing weights
            gate_probs: [N, E] full routing probabilities (for aux loss)
            
        Shape flow:
            # x: (N, H) @ W_gate: (H, E) -> gate_logits: (N, E)
            # gate_probs = softmax(gate_logits, dim=-1)  # (N, E)
            # top_weights, top_indices = topk(gate_probs, k=K, dim=-1)  # each (N, K)
        """
        # x: (N, H) @ W_gate: (H, E) -> gate_logits: (N, E)
        gate_logits = self.gate(x)
        
        # gate_probs = softmax(gate_logits, dim=-1)  # (N, E)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # top_weights, top_indices = topk(gate_probs, k=K, dim=-1)  # each (N, K)
        top_weights, top_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Renormalize weights to sum to 1
        # top_weights: (N, K) -> normalized (N, K)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        return top_indices, top_weights, gate_probs


class Expert(nn.Module):
    """Single expert MLP (FFN).
    
    Standard 2-layer MLP: hidden -> intermediate -> hidden
    with GELU activation.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # W1: [H, I] - up projection
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        # W2: [I, H] - down projection  
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.GELU(approximate="tanh")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expert forward pass.
        
        Args:
            x: [tokens, H] tokens routed to this expert
            
        Returns:
            y: [tokens, H] processed tokens
            
        Shape flow:
            # h = x @ W1  # (tokens, H) -> (tokens, I)
            # h = GELU(h)
            # y = h @ W2  # (tokens, I) -> (tokens, H)
        """
        # h = x @ W1  # (tokens, H) -> (tokens, I)
        h = self.w1(x)
        h = self.act(h)
        # y = h @ W2  # (tokens, I) -> (tokens, H)
        y = self.w2(h)
        return y


class ExpertGroup(nn.Module):
    """Collection of experts on this EP rank.
    
    Manages E_ep = E / EP experts locally.
    """
    
    def __init__(
        self, 
        num_local_experts: int,
        hidden_size: int, 
        intermediate_size: int,
        ep_rank: int = 0,
        num_experts: int = 8,
    ):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.ep_rank = ep_rank
        self.num_experts = num_experts
        
        # Calculate which global expert IDs this rank owns
        # EP rank 0: experts [0, 1, ..., E_ep-1]
        # EP rank 1: experts [E_ep, ..., 2*E_ep-1]
        # etc.
        self.expert_start_idx = ep_rank * num_local_experts
        
        # Create local experts
        self.experts = nn.ModuleList([
            Expert(hidden_size, intermediate_size) 
            for _ in range(num_local_experts)
        ])
    
    def forward(
        self, 
        x_sorted: torch.Tensor, 
        expert_counts: torch.Tensor,
        local_expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process tokens through local experts.
        
        Args:
            x_sorted: [N_local, H] tokens sorted by local expert index
            expert_counts: [E_local] how many tokens per local expert
            local_expert_indices: [E_local] which local expert each token goes to
            
        Returns:
            y_sorted: [N_local, H] processed tokens in same order
            
        Shape flow:
            # for e in 0..E_local-1:
            #   x_e = x_sorted[offsets[e]:offsets[e+1]]  # (tokens_e, H)
            #   y_e = expert_e(x_e)  # (tokens_e, H)
        """
        if x_sorted.size(0) == 0:
            return x_sorted
        
        # Compute offsets from counts
        # offsets: [E_local + 1] cumulative sum for slicing
        offsets = torch.zeros(self.num_local_experts + 1, dtype=torch.long, device=x_sorted.device)
        offsets[1:] = torch.cumsum(expert_counts, dim=0)
        
        # Allocate output buffer
        y_sorted = torch.empty_like(x_sorted)
        
        # Process each local expert
        # for e in 0..E_local-1:
        for e in range(self.num_local_experts):
            start = offsets[e].item()
            end = offsets[e + 1].item()
            if start < end:
                #   x_e = x_sorted[offsets[e]:offsets[e+1]]  # (tokens_e, H)
                x_e = x_sorted[start:end]
                #   y_e = expert_e(x_e)  # (tokens_e, H)
                y_e = self.experts[e](x_e)
                y_sorted[start:end] = y_e
        
        return y_sorted


def load_balancing_loss(
    gate_probs: torch.Tensor,
    top_indices: torch.Tensor,
    num_experts: int,
    top_k: int,
    aux_loss_coef: float = 0.01,
) -> torch.Tensor:
    """
    Compute load balancing auxiliary loss to encourage even expert utilization.
    
    Args:
        gate_probs: [N, E] routing probabilities from softmax
        top_indices: [N, K] selected expert indices
        num_experts: E total experts
        top_k: K experts per token
        aux_loss_coef: α coefficient
        
    Returns:
        aux_loss: scalar loss term
        
    Formula:
        f_e = (# tokens routed to e) / (N * K)  (fraction of selections)
        p_e = mean(gate_probs[:, e])  (mean routing probability)
        L_aux = α * E * Σ_e (f_e * p_e)
    """
    N = gate_probs.size(0)
    device = gate_probs.device
    
    # f_e: fraction of tokens routed to each expert
    # Count occurrences of each expert in top_indices
    # counts: [E] <- scatter_add ones at positions in top_indices
    counts = torch.zeros(num_experts, dtype=gate_probs.dtype, device=device)
    ones = torch.ones_like(top_indices.flatten(), dtype=gate_probs.dtype)
    counts.scatter_add_(0, top_indices.flatten(), ones)
    
    # f = counts / (N * K)  # [E] fraction of total selections
    f = counts / (N * top_k)
    
    # p_e: mean routing probability per expert
    # p = gate_probs.mean(dim=0)  # [E]
    p = gate_probs.mean(dim=0)
    
    # aux_loss = α * E * Σ_e (f_e * p_e)
    aux_loss = aux_loss_coef * num_experts * (f * p).sum()
    
    return aux_loss


class MoELayer(nn.Module):
    """Full MoE layer with routing, dispatch, expert computation, and combine.
    
    Replaces dense MLP in transformer blocks. Supports Expert Parallelism.
    """
    
    def __init__(
        self,
        config: MoEConfig,
        ep_group: Optional[dist.ProcessGroup] = None,
        ep_rank: int = 0,
        ep_world_size: int = 1,
    ):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.hidden_size = config.hidden_size
        self.aux_loss_coef = config.aux_loss_coef
        
        # EP settings
        self.ep_group = ep_group
        self.ep_rank = ep_rank
        self.ep_world_size = ep_world_size
        
        # Validate expert count is divisible by EP world size
        assert config.num_experts % ep_world_size == 0, \
            f"num_experts {config.num_experts} must be divisible by ep_world_size {ep_world_size}"
        self.num_local_experts = config.num_experts // ep_world_size
        
        # Router (replicated across all ranks)
        self.router = Router(config.hidden_size, config.num_experts, config.top_k)
        
        # Local expert group
        self.expert_group = ExpertGroup(
            num_local_experts=self.num_local_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            ep_rank=ep_rank,
            num_experts=config.num_experts,
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MoE forward pass with optional Expert Parallelism.
        
        Args:
            x: [B, S, H] input hidden states
            
        Returns:
            y: [B, S, H] output hidden states
            aux_loss: scalar load balancing loss
            
        Full shape flow (see extension_plan.md Section 11):
            1. Flatten: x[B,S,H] -> x_flat[N,H] where N=B*S
            2. Route: x_flat -> top_indices[N,K], top_weights[N,K]
            3. Expand: x_flat[N,H] -> expanded_x[N*K,H]
            4. Sort by expert: expanded_x -> x_sorted[N*K,H]
            5. (EP) Dispatch: all_to_all -> x_recv[N',H]
            6. Expert compute: x_recv -> y_local[N',H]
            7. (EP) Combine: all_to_all -> y_sorted[N*K,H]
            8. Unsort: y_sorted -> y_expanded[N*K,H]
            9. Weight & reduce: y_expanded -> y[N,H]
            10. Reshape: y[N,H] -> y[B,S,H]
        """
        B, S, H = x.shape
        N = B * S
        
        # 1. Flatten tokens
        # x_flat: (N, H) where N = B*S
        x_flat = rearrange(x, 'b s h -> (b s) h')
        
        # 2. Routing
        # top_indices: (N, K), top_weights: (N, K), gate_probs: (N, E)
        top_indices, top_weights, gate_probs = self.router(x_flat)
        
        # 3. Compute aux loss (before any EP communication)
        aux_loss = load_balancing_loss(
            gate_probs, top_indices, 
            self.num_experts, self.top_k, 
            self.aux_loss_coef
        )

        # 3b. Stash per-expert token counts for diagnostics (detached, no grad)
        with torch.no_grad():
            self._expert_counts = torch.zeros(self.num_experts, device=x.device)
            ones = torch.ones_like(top_indices.flatten(), dtype=torch.float32)
            self._expert_counts.scatter_add_(0, top_indices.flatten(), ones)
        
        # 4. Expand tokens for K selections
        # expanded_x = repeat(x_flat, 'n h -> (n k) h', k=K)  # (N*K, H)
        expanded_x = repeat(x_flat, 'n h -> (n k) h', k=self.top_k)
        # expanded_indices: (N*K,) flatten of top_indices
        expanded_indices = rearrange(top_indices, 'n k -> (n k)')
        # expanded_weights: (N*K,) flatten of top_weights
        expanded_weights = rearrange(top_weights, 'n k -> (n k)')
        
        # 5. Sort by expert id (groups tokens by global expert)
        sort_idx = torch.argsort(expanded_indices, stable=True)
        x_sorted = expanded_x[sort_idx]
        indices_sorted = expanded_indices[sort_idx]
        weights_sorted = expanded_weights[sort_idx]
        
        # 6. Expert Parallel dispatch (optional)
        if self.ep_world_size > 1:
            # Map each token to destination EP rank
            experts_per_rank = self.num_local_experts
            ep_indices = indices_sorted // experts_per_rank  # (N*K,)
            send_counts = torch.bincount(ep_indices, minlength=self.ep_world_size)
            
            # Group by EP rank to match all_to_all splits
            ep_sort_idx = torch.argsort(ep_indices, stable=True)
            x_for_a2a = x_sorted[ep_sort_idx]
            idx_for_a2a = indices_sorted[ep_sort_idx]
            
            # All-to-all tokens and expert ids
            x_recv, recv_counts, idx_recv = all_to_all_dispatch(
                x_for_a2a, send_counts, self.ep_group, self.ep_world_size, payload=idx_for_a2a
            )
            
            # Map to local expert indices [0, num_local_experts)
            local_start = self.ep_rank * self.num_local_experts
            local_indices = idx_recv - local_start
            
            # Sort by local expert for contiguous processing
            local_sort_idx = torch.argsort(local_indices, stable=True)
            x_local_sorted = x_recv[local_sort_idx]
            local_indices_sorted = local_indices[local_sort_idx]
            local_counts = torch.bincount(local_indices_sorted, minlength=self.num_local_experts)
            
            # 7. Run local experts
            y_local_sorted = self.expert_group(x_local_sorted, local_counts, local_indices_sorted)
            y_local = y_local_sorted[torch.argsort(local_sort_idx)]
            
            # 8. Send results back to original EP ranks
            y_return = all_to_all_combine(
                y_local, send_counts, recv_counts, self.ep_group, self.ep_world_size
            )
            
            # 9. Restore order to match x_sorted
            ep_unsort_idx = torch.argsort(ep_sort_idx)
            y_sorted = y_return[ep_unsort_idx]
        else:
            # No EP: process locally (x_sorted already grouped by expert)
            local_indices = indices_sorted
            local_counts = torch.bincount(local_indices, minlength=self.num_local_experts)
            y_sorted = self.expert_group(x_sorted, local_counts, local_indices)
        
        # 10. Unsort back to expanded order
        inverse_sort = torch.argsort(sort_idx)
        y_expanded = y_sorted[inverse_sort]
        weights_expanded = expanded_weights  # (N*K,)
        
        # 11. Apply weights and reduce over K experts
        y_weighted = y_expanded * weights_expanded.unsqueeze(-1)
        y = rearrange(y_weighted, '(n k) h -> n k h', k=self.top_k).sum(dim=1)  # (N, H)
        
        # 12. Reshape back to [B, S, H]
        y = rearrange(y, '(b s) h -> b s h', b=B, s=S)
        
        return y, aux_loss


def make_moe_config(
    hidden_size: int,
    num_experts: int = 8,
    top_k: int = 2,
    intermediate_mult: int = 4,
    aux_loss_coef: float = 0.01,
) -> MoEConfig:
    """Helper to create MoEConfig with sensible defaults."""
    return MoEConfig(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * intermediate_mult,
        aux_loss_coef=aux_loss_coef,
    )
