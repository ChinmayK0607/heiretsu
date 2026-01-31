#!/usr/bin/env python3
"""Forward/grad parity checks for TP (pp=1, dp=1).

Run:
  torchrun --standalone --nproc_per_node=TP tests_equiv.py --tp TP
"""
from __future__ import annotations

import argparse
import os
import torch
import torch.distributed as dist

from gpt_model import GPT, GPTConfig
from topo import init_topology, cleanup


def main():
    p = argparse.ArgumentParser(description="TP parity tests (forward + selected grads)")
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_embed", type=int, default=32)
    p.add_argument("--vocab_size", type=int, default=128)
    args = p.parse_args()

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    topo = init_topology(dp=args.dp, tp=args.tp, pp=args.pp)
    if topo.dp != 1 or topo.pp != 1:
        raise ValueError("tests_equiv.py currently supports dp=1, pp=1 only")

    device = torch.device("cuda", topo.rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    cfg = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embed=args.n_embed,
        dropout=0.0,
    )

    # baseline model on rank0 only
    if topo.rank == 0:
        torch.manual_seed(args.seed)
        base = GPT(cfg).to(device)
    else:
        base = None

    # TP model on all ranks
    torch.manual_seed(args.seed)
    tp_model = GPT(cfg, topo=topo).to(device)

    # shared inputs
    if topo.rank == 0:
        x = torch.randint(0, cfg.vocab_size, (args.batch_size, args.block_size), device=device)
        y = torch.randint(0, cfg.vocab_size, (args.batch_size, args.block_size), device=device)
    else:
        x = torch.empty((args.batch_size, args.block_size), device=device, dtype=torch.long)
        y = torch.empty((args.batch_size, args.block_size), device=device, dtype=torch.long)
    dist.broadcast(x, src=0)
    dist.broadcast(y, src=0)

    # forward
    tp_logits, tp_loss = tp_model(x, y)
    if topo.rank == 0:
        base_logits, base_loss = base(x, y)
        diff = (tp_logits - base_logits).abs()
        print(f"forward diff max={diff.max().item():.6f} mean={diff.mean().item():.6f}")
        print(f"loss diff={abs(tp_loss.item()-base_loss.item()):.6f}")

    # backward (selected grads)
    tp_model.zero_grad(set_to_none=True)
    tp_loss.backward()
    if topo.rank == 0:
        base.zero_grad(set_to_none=True)
        base_loss.backward()

    # compare column-parallel grad (c_attn.weight)
    tp_grad = tp_model.tr.h[0].attn.c_attn.weight.grad
    if args.tp > 1:
        shards = [torch.zeros_like(tp_grad) for _ in range(args.tp)]
        dist.all_gather(shards, tp_grad)
        if topo.rank == 0:
            tp_full = torch.cat(shards, dim=0)
            base_full = base.tr.h[0].attn.c_attn.weight.grad
            diff = (tp_full - base_full).abs()
            print(f"grad c_attn.weight diff max={diff.max().item():.6f} mean={diff.mean().item():.6f}")
    else:
        if topo.rank == 0:
            base_full = base.tr.h[0].attn.c_attn.weight.grad
            diff = (tp_grad - base_full).abs()
            print(f"grad c_attn.weight diff max={diff.max().item():.6f} mean={diff.mean().item():.6f}")

    # compare row-parallel grad (c_proj.weight)
    tp_grad = tp_model.tr.h[0].attn.c_proj.weight.grad
    if args.tp > 1:
        shards = [torch.zeros_like(tp_grad) for _ in range(args.tp)]
        dist.all_gather(shards, tp_grad)
        if topo.rank == 0:
            tp_full = torch.cat(shards, dim=1)
            base_full = base.tr.h[0].attn.c_proj.weight.grad
            diff = (tp_full - base_full).abs()
            print(f"grad c_proj.weight diff max={diff.max().item():.6f} mean={diff.mean().item():.6f}")
    else:
        if topo.rank == 0:
            base_full = base.tr.h[0].attn.c_proj.weight.grad
            diff = (tp_grad - base_full).abs()
            print(f"grad c_proj.weight diff max={diff.max().item():.6f} mean={diff.mean().item():.6f}")

    if dist.is_initialized():
        cleanup()


if __name__ == "__main__":
    main()
