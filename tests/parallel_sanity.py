#!/usr/bin/env python3
"""Forward/backward sanity for DP/TP/PP with max logit diff vs single reference.

Uses synthetic tokens and a small model for speed and deterministic behavior.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.distributed as dist

from gpt_model import GPT, GPTConfig
from tp_linear import ColumnParallelLinear, ColumnParallelLinearQKV, RowParallelLinear
from pipeline import StageModule, GPipeEngine
from topo import init_topology, cleanup, rank_from_coords
from dp import average_gradients


def setup_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def build_ref_logits(cfg: GPTConfig, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict]:
    model = GPT(cfg).to(device)
    model.eval()
    with torch.no_grad():
        logits, loss, _ = model(x, y)
    return logits, loss, model.state_dict()


def _slice_col(full_w: torch.Tensor, tp_rank: int, tp: int) -> torch.Tensor:
    out = full_w.size(0)
    out_per = out // tp
    start = tp_rank * out_per
    end = start + out_per
    return full_w[start:end]


def _slice_row(full_w: torch.Tensor, tp_rank: int, tp: int) -> torch.Tensor:
    in_dim = full_w.size(1)
    in_per = in_dim // tp
    start = tp_rank * in_per
    end = start + in_per
    return full_w[:, start:end]


def _slice_qkv(full_w: torch.Tensor, tp_rank: int, tp: int) -> torch.Tensor:
    # full_w: (3*out, in), shard q/k/v separately
    out = full_w.size(0) // 3
    out_per = out // tp
    shards = []
    for i in range(3):
        start = i * out + tp_rank * out_per
        end = start + out_per
        shards.append(full_w[start:end])
    return torch.cat(shards, dim=0)


def _is_moe_block(blk) -> bool:
    """Check if a block is MoE (has 'moe' attr) vs dense (has 'mlp' attr)."""
    return hasattr(blk, "moe")


def _load_moe_weights(blk, full_state: dict, prefix: str, ep_rank: int = 0) -> None:
    """Load MoE layer weights (router + experts) with EP sharding.
    
    Each EP rank only has local experts. We need to map:
    - local_expert_idx -> global_expert_idx = ep_rank * num_local_experts + local_expert_idx
    """
    # Router gate (replicated across EP ranks)
    blk.moe.router.gate.weight.data.copy_(full_state[prefix + "moe.router.gate.weight"])
    # Experts (via expert_group) - load only local experts
    num_local_experts = len(blk.moe.expert_group.experts)
    expert_start = ep_rank * num_local_experts
    for local_idx, expert in enumerate(blk.moe.expert_group.experts):
        global_idx = expert_start + local_idx
        expert.w1.weight.data.copy_(full_state[f"{prefix}moe.expert_group.experts.{global_idx}.w1.weight"])
        expert.w2.weight.data.copy_(full_state[f"{prefix}moe.expert_group.experts.{global_idx}.w2.weight"])


def _load_dense_mlp_weights(blk, full_state: dict, prefix: str, tp_rank: int, tp: int) -> None:
    """Load dense MLP weights with TP slicing."""
    full_fc1_w = full_state[prefix + "mlp.c_fc.weight"]
    full_fc1_b = full_state[prefix + "mlp.c_fc.bias"]
    blk.mlp.c_fc.weight.data.copy_(_slice_col(full_fc1_w, tp_rank, tp))
    blk.mlp.c_fc.bias.data.copy_(_slice_col(full_fc1_b.unsqueeze(1), tp_rank, tp).squeeze(1))

    full_fc2_w = full_state[prefix + "mlp.c_proj.weight"]
    full_fc2_b = full_state[prefix + "mlp.c_proj.bias"]
    blk.mlp.c_proj.weight.data.copy_(_slice_row(full_fc2_w, tp_rank, tp))
    blk.mlp.c_proj.bias.data.copy_(full_fc2_b)


def load_gpt_from_full_state(model: GPT, full_state: dict, tp_rank: int, tp: int, ep_rank: int = 0) -> None:
    # Check if model has MoE layers (which require special EP handling)
    has_moe = any(_is_moe_block(blk) for blk in model.tr.h)
    if tp == 1 and not has_moe:
        model.load_state_dict(full_state)
        return
    # embeddings/head
    model.tr.wte.weight.data.copy_(full_state["tr.wte.weight"])
    model.tr.wpe.weight.data.copy_(full_state["tr.wpe.weight"])
    model.tr.ln_f.weight.data.copy_(full_state["tr.ln_f.weight"])
    model.tr.ln_f.bias.data.copy_(full_state["tr.ln_f.bias"])
    model.lm_head.weight.data.copy_(full_state["lm_head.weight"])

    # blocks
    for i, blk in enumerate(model.tr.h):
        prefix = f"tr.h.{i}."
        blk.ln_1.weight.data.copy_(full_state[prefix + "ln_1.weight"])
        blk.ln_1.bias.data.copy_(full_state[prefix + "ln_1.bias"])
        blk.ln_2.weight.data.copy_(full_state[prefix + "ln_2.weight"])
        blk.ln_2.bias.data.copy_(full_state[prefix + "ln_2.bias"])

        # attention qkv
        full_qkv_w = full_state[prefix + "attn.c_attn.weight"]
        full_qkv_b = full_state[prefix + "attn.c_attn.bias"]
        blk.attn.c_attn.weight.data.copy_(_slice_qkv(full_qkv_w, tp_rank, tp))
        blk.attn.c_attn.bias.data.copy_(_slice_qkv(full_qkv_b.unsqueeze(1), tp_rank, tp).squeeze(1))

        # attention out proj
        full_o_w = full_state[prefix + "attn.c_proj.weight"]
        full_o_b = full_state[prefix + "attn.c_proj.bias"]
        blk.attn.c_proj.weight.data.copy_(_slice_row(full_o_w, tp_rank, tp))
        blk.attn.c_proj.bias.data.copy_(full_o_b)

        # MoE vs dense MLP
        if _is_moe_block(blk):
            _load_moe_weights(blk, full_state, prefix, ep_rank)
        else:
            _load_dense_mlp_weights(blk, full_state, prefix, tp_rank, tp)


def load_stage_from_full_state(stage: StageModule, full_state: dict, tp_rank: int, tp: int, ep_rank: int = 0) -> None:
    # embeddings/head
    if stage.is_first:
        stage.wte.weight.data.copy_(full_state["tr.wte.weight"])
        stage.wpe.weight.data.copy_(full_state["tr.wpe.weight"])
    if stage.is_last:
        stage.ln_f.weight.data.copy_(full_state["tr.ln_f.weight"])
        stage.ln_f.bias.data.copy_(full_state["tr.ln_f.bias"])
        stage.lm_head.weight.data.copy_(full_state["lm_head.weight"])

    # blocks
    for local_i, global_i in enumerate(range(stage.block_start, stage.block_end)):
        blk = stage.blocks[local_i]
        prefix = f"tr.h.{global_i}."
        # layer norms
        blk.ln_1.weight.data.copy_(full_state[prefix + "ln_1.weight"])
        blk.ln_1.bias.data.copy_(full_state[prefix + "ln_1.bias"])
        blk.ln_2.weight.data.copy_(full_state[prefix + "ln_2.weight"])
        blk.ln_2.bias.data.copy_(full_state[prefix + "ln_2.bias"])

        # attention qkv (col-parallel)
        full_qkv_w = full_state[prefix + "attn.c_attn.weight"]
        full_qkv_b = full_state[prefix + "attn.c_attn.bias"]
        blk.attn.c_attn.weight.data.copy_(_slice_qkv(full_qkv_w, tp_rank, tp))
        blk.attn.c_attn.bias.data.copy_(_slice_qkv(full_qkv_b.unsqueeze(1), tp_rank, tp).squeeze(1))

        # attention out proj (row-parallel)
        full_o_w = full_state[prefix + "attn.c_proj.weight"]
        full_o_b = full_state[prefix + "attn.c_proj.bias"]
        blk.attn.c_proj.weight.data.copy_(_slice_row(full_o_w, tp_rank, tp))
        blk.attn.c_proj.bias.data.copy_(full_o_b)

        # MoE vs dense MLP
        if _is_moe_block(blk):
            _load_moe_weights(blk, full_state, prefix, ep_rank)
        else:
            _load_dense_mlp_weights(blk, full_state, prefix, tp_rank, tp)


def pp_forward_logits(stage: StageModule, topo, x: torch.Tensor, y: torch.Tensor, cfg: GPTConfig) -> Optional[torch.Tensor]:
    """Run a single microbatch forward through PP stages and return logits on last stage.
    
    For MoE+PP: EP ranks must sync before forward_blocks since MoE uses all-to-all.
    """
    device = x.device
    B, T = x.shape
    H = stage.cfg.n_embed
    
    has_moe = cfg.num_experts > 0 and cfg.moe_freq > 0

    if stage.is_first:
        with torch.no_grad():
            h = stage.embed(x)
            # Sync EP ranks before MoE forward all-to-all
            if has_moe and topo.ep_group is not None:
                dist.barrier(group=topo.ep_group)
            h, _ = stage.forward_blocks(h)
        if topo.pp_rank < topo.pp - 1:
            dist.send(h, dst=rank_from_coords(topo.dp_rank, topo.ep_rank, topo.pp_rank + 1, topo.tp_rank, topo.dp, topo.ep, topo.tp, topo.pp), tag=1000)
        return None

    if stage.is_last:
        h = torch.empty((B, T, H), device=device)
        dist.recv(h, src=rank_from_coords(topo.dp_rank, topo.ep_rank, topo.pp_rank - 1, topo.tp_rank, topo.dp, topo.ep, topo.tp, topo.pp), tag=1000)
        with torch.no_grad():
            # Sync EP ranks before MoE forward all-to-all
            if has_moe and topo.ep_group is not None:
                dist.barrier(group=topo.ep_group)
            h, _ = stage.forward_blocks(h)
            logits, _ = stage.forward_head(h, y)
        return logits

    # middle stage
    h = torch.empty((B, T, H), device=device)
    dist.recv(h, src=rank_from_coords(topo.dp_rank, topo.ep_rank, topo.pp_rank - 1, topo.tp_rank, topo.dp, topo.ep, topo.tp, topo.pp), tag=1000)
    with torch.no_grad():
        # Sync EP ranks before MoE forward all-to-all
        if has_moe and topo.ep_group is not None:
            dist.barrier(group=topo.ep_group)
        h, _ = stage.forward_blocks(h)
    dist.send(h, dst=rank_from_coords(topo.dp_rank, topo.ep_rank, topo.pp_rank + 1, topo.tp_rank, topo.dp, topo.ep, topo.tp, topo.pp), tag=1000)
    return None


class _LoaderAdapter:
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
        self.used = False

    def next_batch(self, device: torch.device):
        if self.used:
            return self.x.to(device), self.y.to(device)
        self.used = True
        return self.x.to(device), self.y.to(device)


def main():
    p = argparse.ArgumentParser(description="DP/TP/PP forward+backward sanity with logit diff")
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--backend", type=str, default="nccl")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embed", type=int, default=64)
    p.add_argument("--vocab_size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    # MoE arguments
    p.add_argument("--num_experts", type=int, default=0, help="Number of experts (0=dense)")
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--moe_freq", type=int, default=2, help="MoE layer frequency")
    p.add_argument("--aux_loss_coef", type=float, default=0.01)
    args = p.parse_args()

    if args.backend == "nccl" and not torch.cuda.is_available():
        args.backend = "gloo"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.backend == "nccl":
        dist.init_process_group(backend=args.backend, device_id=torch.device("cuda", local_rank))
    else:
        dist.init_process_group(backend=args.backend)
    topo = init_topology(dp=args.dp, ep=args.ep, tp=args.tp, pp=args.pp)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    setup_seed(args.seed, device)

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

    # synthetic batch, broadcast to all ranks for comparability
    if topo.rank == 0:
        x = torch.randint(cfg.vocab_size, (args.batch_size, args.block_size), device=device)
        y = torch.randint(cfg.vocab_size, (args.batch_size, args.block_size), device=device)
    else:
        x = torch.empty((args.batch_size, args.block_size), device=device, dtype=torch.long)
        y = torch.empty((args.batch_size, args.block_size), device=device, dtype=torch.long)
    dist.broadcast(x, src=0)
    dist.broadcast(y, src=0)

    # reference logits/loss on rank0
    if topo.rank == 0:
        ref_logits, ref_loss, ref_state = build_ref_logits(cfg, x, y, device)
    else:
        ref_logits = None
        ref_loss = None
        ref_state = None

    # broadcast reference state for PP loading
    obj_list = [ref_state]
    dist.broadcast_object_list(obj_list, src=0)
    ref_state = obj_list[0]

    # reset seed before building distributed model to align init with reference
    dist.barrier()
    setup_seed(args.seed, device)

    # forward path under target topology
    if topo.pp > 1:
        stage = StageModule(cfg, topo, topo.pp_rank, topo.pp).to(device)
        load_stage_from_full_state(stage, ref_state, topo.tp_rank, topo.tp, topo.ep_rank)
        stage.eval()
        logits = pp_forward_logits(stage, topo, x, y, cfg)
        # Only the last stage in dp_rank=0, ep_rank=0 sends logits to rank0 for comparison.
        # Other DP/EP replicas skip the send/recv to avoid hangs.
        if stage.is_last and topo.dp_rank == 0 and topo.ep_rank == 0 and topo.tp_rank == 0:
            # send logits to rank0 for comparison
            dist.send(logits, dst=0, tag=2000)
        if topo.rank == 0:
            # receive logits from last stage (dp_rank=0, tp_rank=0)
            last_rank = rank_from_coords(0, 0, topo.pp - 1, 0, topo.dp, topo.ep, topo.tp, topo.pp)
            recv = torch.empty_like(ref_logits)
            dist.recv(recv, src=last_rank, tag=2000)
            logits = recv
        # Synchronize all ranks before logit comparison
        dist.barrier()
    else:
        model = GPT(cfg, topo=topo).to(device)
        load_gpt_from_full_state(model, ref_state, topo.tp_rank, topo.tp, topo.ep_rank)
        model.eval()
        with torch.no_grad():
            logits, loss, _ = model(x, y)
        if topo.rank == 0:
            ref_loss = ref_loss  # keep for prints

    # compare logits on rank0 only
    if topo.rank == 0:
        diff = (logits - ref_logits).abs()
        diff_max = float(diff.max().item())
        diff_mean = float(diff.mean().item())
        print(f"LOGIT_DIFF_MAX={diff_max}")
        print(f"LOGIT_DIFF_MEAN={diff_mean}")

    # backward sanity (single step)
    if topo.pp > 1:
        stage = StageModule(cfg, topo, topo.pp_rank, topo.pp).to(device)
        engine = GPipeEngine(stage, topo)
        stage.train()

        loss, _ = engine.forward_backward(
            loader=_LoaderAdapter(x, y),
            micro_batches=1,
            device=device,
            use_amp=False,
            amp_dtype=torch.float32,
            scaler=None,
            batch_size=args.batch_size,
        )

        # Average gradients across DP replicas if dp > 1
        average_gradients(stage, topo.dp_group)
    else:
        model = GPT(cfg, topo=topo).to(device)
        model.train()
        logits, loss, aux = model(x, y)
        (loss + aux).backward()
        average_gradients(model, topo.dp_group)

    # Synchronize all ranks before printing and cleanup
    dist.barrier()

    if topo.rank == 0:
        print("BACKWARD_OK=1")

    if dist.is_initialized():
        cleanup()


if __name__ == "__main__":
    main()
