#!/usr/bin/env python3
"""Test MoE layer and routing correctness."""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from moe import Router, Expert, ExpertGroup, MoELayer, MoEConfig, load_balancing_loss


def test_router():
    """Test top-k routing produces valid indices and weights."""
    print("Testing Router...")
    
    B, S, H = 2, 4, 64
    E, K = 8, 2
    
    router = Router(hidden_size=H, num_experts=E, top_k=K)
    x = torch.randn(B * S, H)
    
    top_indices, top_weights, gate_probs = router(x)
    
    # Check shapes
    assert top_indices.shape == (B * S, K), f"Expected {(B*S, K)}, got {top_indices.shape}"
    assert top_weights.shape == (B * S, K), f"Expected {(B*S, K)}, got {top_weights.shape}"
    assert gate_probs.shape == (B * S, E), f"Expected {(B*S, E)}, got {gate_probs.shape}"
    
    # Check indices are valid
    assert top_indices.min() >= 0 and top_indices.max() < E, "Invalid expert indices"
    
    # Check weights sum to 1
    weight_sum = top_weights.sum(dim=-1)
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum)), "Weights should sum to 1"
    
    # Check gate_probs sum to 1
    prob_sum = gate_probs.sum(dim=-1)
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum)), "Gate probs should sum to 1"
    
    print("  ✓ Router test passed")


def test_expert():
    """Test single expert MLP forward pass."""
    print("Testing Expert...")
    
    H, I = 64, 256
    tokens = 10
    
    expert = Expert(hidden_size=H, intermediate_size=I)
    x = torch.randn(tokens, H)
    
    y = expert(x)
    
    assert y.shape == (tokens, H), f"Expected {(tokens, H)}, got {y.shape}"
    print("  ✓ Expert test passed")


def test_load_balancing_loss():
    """Test aux loss computation."""
    print("Testing load_balancing_loss...")
    
    N, E, K = 16, 4, 2
    
    # Create mock routing outputs
    gate_probs = torch.softmax(torch.randn(N, E), dim=-1)
    top_indices = torch.randint(0, E, (N, K))
    
    aux_loss = load_balancing_loss(gate_probs, top_indices, E, K, aux_loss_coef=0.01)
    
    assert aux_loss.shape == (), f"Expected scalar, got {aux_loss.shape}"
    assert aux_loss >= 0, "Aux loss should be non-negative"
    
    print(f"  aux_loss = {aux_loss.item():.6f}")
    print("  ✓ load_balancing_loss test passed")


def test_moe_layer_forward():
    """Test full MoE layer forward pass."""
    print("Testing MoELayer forward...")
    
    B, S, H = 2, 8, 64
    E, K = 4, 2
    I = 256
    
    config = MoEConfig(
        num_experts=E,
        top_k=K,
        hidden_size=H,
        intermediate_size=I,
        aux_loss_coef=0.01,
    )
    
    moe = MoELayer(config)
    x = torch.randn(B, S, H)
    
    y, aux_loss = moe(x)
    
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    assert aux_loss.shape == (), f"Expected scalar aux_loss, got {aux_loss.shape}"
    
    print(f"  output shape: {y.shape}")
    print(f"  aux_loss: {aux_loss.item():.6f}")
    print("  ✓ MoELayer forward test passed")


def test_moe_layer_backward():
    """Test MoE layer backward pass."""
    print("Testing MoELayer backward...")
    
    B, S, H = 2, 4, 32
    E, K = 4, 2
    I = 128
    
    config = MoEConfig(
        num_experts=E,
        top_k=K,
        hidden_size=H,
        intermediate_size=I,
        aux_loss_coef=0.01,
    )
    
    moe = MoELayer(config)
    x = torch.randn(B, S, H, requires_grad=True)
    
    y, aux_loss = moe(x)
    
    # Backward through output
    loss = y.sum() + aux_loss
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None, "Input should have gradient"
    assert x.grad.shape == x.shape, "Gradient shape mismatch"
    
    # Check expert weights have gradients
    for i, expert in enumerate(moe.expert_group.experts):
        assert expert.w1.weight.grad is not None, f"Expert {i} w1 should have gradient"
        assert expert.w2.weight.grad is not None, f"Expert {i} w2 should have gradient"
    
    print("  ✓ MoELayer backward test passed")


def test_expert_selection_diversity():
    """Test that different tokens can select different experts."""
    print("Testing expert selection diversity...")
    
    B, S, H = 4, 16, 64
    E, K = 8, 2
    
    config = MoEConfig(num_experts=E, top_k=K, hidden_size=H, intermediate_size=256)
    moe = MoELayer(config)
    
    # Create diverse inputs
    x = torch.randn(B, S, H)
    
    # Get routing decisions
    x_flat = x.view(-1, H)
    top_indices, _, _ = moe.router(x_flat)
    
    # Check that multiple experts are used
    unique_experts = torch.unique(top_indices)
    print(f"  Unique experts selected: {len(unique_experts)}/{E}")
    
    assert len(unique_experts) > 1, "Should select multiple different experts"
    print("  ✓ Expert selection diversity test passed")


if __name__ == "__main__":
    print("=" * 50)
    print("MoE Unit Tests")
    print("=" * 50)
    
    test_router()
    test_expert()
    test_load_balancing_loss()
    test_moe_layer_forward()
    test_moe_layer_backward()
    test_expert_selection_diversity()
    
    print("=" * 50)
    print("All MoE tests passed! ✓")
    print("=" * 50)
