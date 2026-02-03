#!/usr/bin/env python3
"""Max boundary-pushing test for DP/TP/PP.

Runs several steps with multiple microbatches to stress PP boundaries.
Uses synthetic token data (no dataset dependency).
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist

from gpt_model import GPT, GPTConfig
from pipeline import StageModule, GPipeEngine
from topo import init_topology, cleanup
from dp import average_gradients


class RandomTokenLoader:
    def __init__(self, vocab_size: int, B: int, T: int, seed: int, dp_rank: int):
        self.vocab_size = vocab_size
        self.B = B
        self.T = T
        self.gen = torch.Generator()
        self.gen.manual_seed(seed + dp_rank)

    def next_batch(self, device: torch.device):
        # generate on CPU for deterministic generator, then move to device
        x = torch.randint(self.vocab_size, (self.B, self.T), generator=self.gen, device="cpu").to(device)
        y = torch.randint(self.vocab_size, (self.B, self.T), generator=self.gen, device="cpu").to(device)
        return x, y


def main():
    p = argparse.ArgumentParser(description="Boundary-pushing test for DP/TP/PP")
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--micro_batches", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embed", type=int, default=64)
    p.add_argument("--vocab_size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--backend", type=str, default="nccl")
    # MoE arguments
    p.add_argument("--num_experts", type=int, default=0, help="Number of experts (0=dense)")
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--moe_freq", type=int, default=2, help="MoE layer frequency")
    p.add_argument("--aux_loss_coef", type=float, default=0.01)
    p.add_argument(
        "--progress_interval",
        type=int,
        default=1,
        help="If >0, print microbatch progress every N microbatches (rank0 only)",
    )
    args = p.parse_args()

    if args.backend == "nccl" and not torch.cuda.is_available():
        args.backend = "gloo"

    dist.init_process_group(backend=args.backend)
    topo = init_topology(dp=args.dp, ep=args.ep, tp=args.tp, pp=args.pp)

    device = torch.device("cuda", topo.rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # deterministic init across replicas
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cfg = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embed=args.n_embed,
        dropout=args.dropout,
        num_experts=args.num_experts,
        top_k=args.top_k,
        moe_freq=args.moe_freq,
        aux_loss_coef=args.aux_loss_coef,
    )

    use_pipeline = topo.pp > 1
    if use_pipeline:
        stage = StageModule(cfg, topo, topo.pp_rank, topo.pp).to(device)
        engine = GPipeEngine(stage, topo)
        local_module = stage
    else:
        model = GPT(cfg, topo=topo).to(device)
        engine = None
        local_module = model

    optimizer = torch.optim.AdamW(local_module.parameters(), lr=args.lr)
    loader = RandomTokenLoader(cfg.vocab_size, args.batch_size, args.block_size, args.seed, topo.dp_rank)

    local_module.train()
    for step in range(1, args.steps + 1):
        if use_pipeline:
            loss_scalar, _ = engine.forward_backward(
                loader,
                micro_batches=args.micro_batches,
                device=device,
                use_amp=False,
                amp_dtype=torch.float32,
                scaler=None,
                batch_size=args.batch_size,
            )
        else:
            for _ in range(args.micro_batches):
                x, y = loader.next_batch(device)
                _, loss, aux = local_module(x, y)
                ((loss + aux) / args.micro_batches).backward()
            loss_scalar = loss.detach().float()

        average_gradients(local_module, topo.dp_group)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # For PP configs, send loss from last stage to rank 0 for printing
        if use_pipeline and topo.pp > 1:
            # Determine the rank of the canonical last stage (dp_rank=0, tp_rank=0, pp_rank=pp-1)
            from topo import rank_from_coords
            last_stage_rank = rank_from_coords(0, 0, topo.pp - 1, 0, topo.dp, topo.ep, topo.tp, topo.pp)
            if engine.is_last and topo.dp_rank == 0 and topo.tp_rank == 0 and topo.rank != 0:
                # Last stage sends loss to rank 0
                dist.send(loss_scalar.unsqueeze(0), dst=0, tag=4000 + step)
            elif topo.rank == 0:
                # Rank 0 receives loss from last stage
                recv_loss = torch.empty(1, device=device)
                dist.recv(recv_loss, src=last_stage_rank, tag=4000 + step)
                loss_scalar = recv_loss[0]

        if topo.rank == 0:
            mb_msg = f"(mb={args.micro_batches})"
            if loss_scalar is not None:
                print(f"step {step}/{args.steps} loss {loss_scalar.item():.4f} {mb_msg}", flush=True)
            else:
                print(f"step {step}/{args.steps} loss n/a {mb_msg}", flush=True)
        if use_pipeline and topo.rank == 0 and args.progress_interval > 0:
            # lightweight textual progress indicator
            if step % 1 == 0:
                print(f"  progress step {step}: processed {args.micro_batches} microbatches", flush=True)

    if dist.is_initialized():
        cleanup()


if __name__ == "__main__":
    main()
