#!/usr/bin/env python3
"""Backward parity test vs single-GPU baseline.

Checks gradients for a small subset of parameters across DP/TP/PP.
"""
from __future__ import annotations

import argparse
import random
from typing import Dict, Any, List

import numpy as np
import torch
import torch.distributed as dist

from gpt_model import GPT, GPTConfig
from pipeline import StageModule, GPipeEngine
from topo import init_topology, cleanup
from dp import average_gradients


def setup_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _gather_param_grads(model: torch.nn.Module, names: List[str]) -> Dict[str, torch.Tensor]:
    """Collect grads keyed by canonical GPT-style names.

    For StageModule (PP), parameter names differ from full GPT. We remap:
    - blocks.{local}.<rest> -> tr.h.{global}.<rest>
    - wte/wpe/ln_f/lm_head -> tr.* equivalents
    """
    grads: Dict[str, torch.Tensor] = {}
    is_stage = hasattr(model, "block_start")

    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        canon = name
        if is_stage:
            if name.startswith("blocks."):
                parts = name.split(".")
                local_idx = int(parts[1])
                rest = ".".join(parts[2:])
                global_idx = model.block_start + local_idx
                canon = f"tr.h.{global_idx}.{rest}"
            elif name.startswith("wte."):
                canon = "tr.wte." + name.split(".", 1)[1]
            elif name.startswith("wpe."):
                canon = "tr.wpe." + name.split(".", 1)[1]
            elif name.startswith("ln_f."):
                canon = "tr.ln_f." + name.split(".", 1)[1]
            elif name.startswith("lm_head."):
                canon = "lm_head." + name.split(".", 1)[1]

        grads[canon] = p.grad.detach().float().cpu()

    # If a specific subset is requested, filter; otherwise return all collected.
    if names:
        return {k: v for k, v in grads.items() if k in names}
    return grads


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
    out = full_w.size(0) // 3
    out_per = out // tp
    shards = []
    for i in range(3):
        start = i * out + tp_rank * out_per
        end = start + out_per
        shards.append(full_w[start:end])
    return torch.cat(shards, dim=0)


def load_gpt_from_full_state(model: GPT, full_state: dict, tp_rank: int, tp: int) -> None:
    if tp == 1:
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

        # mlp fc1
        full_fc1_w = full_state[prefix + "mlp.c_fc.weight"]
        full_fc1_b = full_state[prefix + "mlp.c_fc.bias"]
        blk.mlp.c_fc.weight.data.copy_(_slice_col(full_fc1_w, tp_rank, tp))
        blk.mlp.c_fc.bias.data.copy_(_slice_col(full_fc1_b.unsqueeze(1), tp_rank, tp).squeeze(1))

        # mlp fc2
        full_fc2_w = full_state[prefix + "mlp.c_proj.weight"]
        full_fc2_b = full_state[prefix + "mlp.c_proj.bias"]
        blk.mlp.c_proj.weight.data.copy_(_slice_row(full_fc2_w, tp_rank, tp))
        blk.mlp.c_proj.bias.data.copy_(full_fc2_b)


def load_stage_from_full_state(stage: StageModule, full_state: dict, tp_rank: int, tp: int) -> None:
    if stage.is_first:
        stage.wte.weight.data.copy_(full_state["tr.wte.weight"])
        stage.wpe.weight.data.copy_(full_state["tr.wpe.weight"])
    if stage.is_last:
        stage.ln_f.weight.data.copy_(full_state["tr.ln_f.weight"])
        stage.ln_f.bias.data.copy_(full_state["tr.ln_f.bias"])
        stage.lm_head.weight.data.copy_(full_state["lm_head.weight"])

    for local_i, global_i in enumerate(range(stage.block_start, stage.block_end)):
        blk = stage.blocks[local_i]
        prefix = f"tr.h.{global_i}."
        blk.ln_1.weight.data.copy_(full_state[prefix + "ln_1.weight"])
        blk.ln_1.bias.data.copy_(full_state[prefix + "ln_1.bias"])
        blk.ln_2.weight.data.copy_(full_state[prefix + "ln_2.weight"])
        blk.ln_2.bias.data.copy_(full_state[prefix + "ln_2.bias"])

        full_qkv_w = full_state[prefix + "attn.c_attn.weight"]
        full_qkv_b = full_state[prefix + "attn.c_attn.bias"]
        blk.attn.c_attn.weight.data.copy_(_slice_qkv(full_qkv_w, tp_rank, tp))
        blk.attn.c_attn.bias.data.copy_(_slice_qkv(full_qkv_b.unsqueeze(1), tp_rank, tp).squeeze(1))

        full_o_w = full_state[prefix + "attn.c_proj.weight"]
        full_o_b = full_state[prefix + "attn.c_proj.bias"]
        blk.attn.c_proj.weight.data.copy_(_slice_row(full_o_w, tp_rank, tp))
        blk.attn.c_proj.bias.data.copy_(full_o_b)

        full_fc1_w = full_state[prefix + "mlp.c_fc.weight"]
        full_fc1_b = full_state[prefix + "mlp.c_fc.bias"]
        blk.mlp.c_fc.weight.data.copy_(_slice_col(full_fc1_w, tp_rank, tp))
        blk.mlp.c_fc.bias.data.copy_(_slice_col(full_fc1_b.unsqueeze(1), tp_rank, tp).squeeze(1))

        full_fc2_w = full_state[prefix + "mlp.c_proj.weight"]
        full_fc2_b = full_state[prefix + "mlp.c_proj.bias"]
        blk.mlp.c_proj.weight.data.copy_(_slice_row(full_fc2_w, tp_rank, tp))
        blk.mlp.c_proj.bias.data.copy_(full_fc2_b)


def main() -> None:
    p = argparse.ArgumentParser(description="Gradient parity test for DP/TP/PP")
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--backend", type=str, default="nccl")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--block_size", type=int, default=8)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embed", type=int, default=64)
    p.add_argument("--vocab_size", type=int, default=256)
    # MoE arguments
    p.add_argument("--num_experts", type=int, default=0, help="Number of experts (0=dense)")
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--moe_freq", type=int, default=2, help="MoE layer frequency")
    p.add_argument("--aux_loss_coef", type=float, default=0.01)
    args = p.parse_args()

    if args.backend == "nccl" and not torch.cuda.is_available():
        args.backend = "gloo"

    dist.init_process_group(backend=args.backend)
    topo = init_topology(dp=args.dp, ep=args.ep, tp=args.tp, pp=args.pp)

    # Ensure pipeline partitioning is valid: must have n_layer >= pp
    if args.pp > args.n_layer:
        if topo.rank == 0:
            print(f"[grad_parity] bumping n_layer from {args.n_layer} to {args.pp} to satisfy pp>n_layer constraint")
        args.n_layer = args.pp

    device = torch.device("cuda", topo.rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    setup_seed(args.seed, device)

    cfg = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embed=args.n_embed,
        dropout=0.0,
        num_experts=args.num_experts,
        top_k=args.top_k,
        moe_freq=args.moe_freq,
        aux_loss_coef=args.aux_loss_coef,
    )

    # synthetic batch (broadcast to all ranks)
    if topo.rank == 0:
        x = torch.randint(cfg.vocab_size, (args.batch_size, args.block_size), device=device)
        y = torch.randint(cfg.vocab_size, (args.batch_size, args.block_size), device=device)
    else:
        x = torch.empty((args.batch_size, args.block_size), device=device, dtype=torch.long)
        y = torch.empty((args.batch_size, args.block_size), device=device, dtype=torch.long)
    dist.broadcast(x, src=0)
    dist.broadcast(y, src=0)

    # baseline on rank0
    if topo.rank == 0:
        setup_seed(args.seed, device)
        base = GPT(cfg).to(device)
        base.train()
        _, base_loss = base(x, y)
        base_loss.backward()
        base_state = base.state_dict()
        base_grads = _gather_param_grads(
            base,
            [
                "tr.wte.weight",
                "tr.h.0.ln_1.weight",
                "tr.h.0.attn.c_attn.weight",
                "tr.h.0.attn.c_proj.weight",
            ],
        )
    else:
        base_state = None
        base_grads = None

    obj_list = [base_state]
    dist.broadcast_object_list(obj_list, src=0)
    base_state = obj_list[0]

    # distributed forward/backward
    if topo.pp > 1:
        stage = StageModule(cfg, topo, topo.pp_rank, topo.pp).to(device)
        engine = GPipeEngine(stage, topo)
        load_stage_from_full_state(stage, base_state, topo.tp_rank, topo.tp)
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
        average_gradients(stage, topo.dp_group)
        local_module = stage
    else:
        model = GPT(cfg, topo=topo).to(device)
        load_gpt_from_full_state(model, base_state, topo.tp_rank, topo.tp)
        model.train()
        _, loss, aux = model(x, y)
        (loss + aux).backward()
        average_gradients(model, topo.dp_group)
        local_module = model

    local_grads = _gather_param_grads(
        local_module,
        [
            "tr.wte.weight",
            "tr.h.0.ln_1.weight",
            "tr.h.0.attn.c_attn.weight",
            "tr.h.0.attn.c_proj.weight",
        ],
    )

    # gather all grad shards to rank0 for comparison
    payload = {
        "rank": topo.rank,
        "tp_rank": topo.tp_rank,
        "pp_rank": topo.pp_rank,
        "dp_rank": topo.dp_rank,
        "grads": local_grads,
    }
    gathered: List[Any] = [None for _ in range(topo.world_size)] if topo.rank == 0 else None
    dist.gather_object(payload, gathered, dst=0)

    if topo.rank == 0:
        # reconstruct sharded grads for c_attn (QKV interleaved) and c_proj (axis=1)
        # Note: For PP, layer 0 (tr.h.0.*) is only on pp_rank=0 (first stage)
        def recon_shards(name: str, axis: int, target_pp_rank: int = 0) -> torch.Tensor:
            shards = []
            for item in gathered:
                if item["dp_rank"] != 0 or item["pp_rank"] != target_pp_rank:
                    continue
                g = item["grads"].get(name)
                if g is None:
                    continue
                shards.append((item["tp_rank"], g))
            shards.sort(key=lambda t: t[0])
            if not shards:
                return None
            return torch.cat([g for _, g in shards], dim=axis)

        def recon_qkv_shards(name: str, target_pp_rank: int = 0) -> torch.Tensor:
            """Reconstruct QKV weight gradient from interleaved TP shards.
            
            Each TP shard has shape (3 * out_per_tp, in_features) with layout:
              [Q_shard, K_shard, V_shard] stacked along dim 0.
            Full weight has shape (3 * out_features, in_features) with layout:
              [Q_full, K_full, V_full] stacked along dim 0.
            We need to reassemble Q/K/V separately then restack.
            """
            shards = []
            for item in gathered:
                if item["dp_rank"] != 0 or item["pp_rank"] != target_pp_rank:
                    continue
                g = item["grads"].get(name)
                if g is None:
                    continue
                shards.append((item["tp_rank"], g))
            shards.sort(key=lambda t: t[0])
            if not shards:
                return None
            
            tp_size = len(shards)
            if tp_size == 1:
                return shards[0][1]
            
            # Each shard: (3 * out/tp, in)
            # Split each shard into Q, K, V parts
            q_parts, k_parts, v_parts = [], [], []
            for _, g in shards:
                out_per_tp = g.shape[0] // 3
                q_parts.append(g[:out_per_tp])
                k_parts.append(g[out_per_tp:2*out_per_tp])
                v_parts.append(g[2*out_per_tp:])
            
            # Reconstruct full Q, K, V by concatenating along axis 0
            q_full = torch.cat(q_parts, dim=0)
            k_full = torch.cat(k_parts, dim=0)
            v_full = torch.cat(v_parts, dim=0)
            
            # Stack Q, K, V back together
            return torch.cat([q_full, k_full, v_full], dim=0)

        results = {}
        # wte.weight: For PP > 1, wte is only on first stage and NOT tied to lm_head.
        # This means wte gradient only includes embedding backward, not lm_head backward.
        # Baseline has tied weights so wte.grad = embed_grad + lm_head_grad.
        # This difference is EXPECTED for PP > 1 when stages are on different GPUs.
        # We still report the diff but note it's expected.
        for pname in ["tr.wte.weight", "tr.h.0.ln_1.weight"]:
            # wte is on pp_rank=0, ln_1 for layer 0 is also on pp_rank=0
            target_pp = 0
            g = None
            for item in gathered:
                if item["dp_rank"] != 0 or item["pp_rank"] != target_pp:
                    continue
                if pname in item["grads"]:
                    g = item["grads"][pname]
                    break
            if g is not None and base_grads and pname in base_grads:
                diff = (g - base_grads[pname]).abs()
                results[pname] = (float(diff.max()), float(diff.mean()))

        # sharded params - layer 0 is on pp_rank=0
        # c_attn uses QKV interleaved sharding, c_proj uses row sharding
        if base_grads and "tr.h.0.attn.c_attn.weight" in base_grads:
            g_full = recon_qkv_shards("tr.h.0.attn.c_attn.weight", target_pp_rank=0)
            if g_full is not None:
                diff = (g_full - base_grads["tr.h.0.attn.c_attn.weight"]).abs()
                results["tr.h.0.attn.c_attn.weight"] = (float(diff.max()), float(diff.mean()))
        if base_grads and "tr.h.0.attn.c_proj.weight" in base_grads:
            g_full = recon_shards("tr.h.0.attn.c_proj.weight", axis=1, target_pp_rank=0)
            if g_full is not None:
                diff = (g_full - base_grads["tr.h.0.attn.c_proj.weight"]).abs()
                results["tr.h.0.attn.c_proj.weight"] = (float(diff.max()), float(diff.mean()))

        for k, (mx, mn) in results.items():
            print(f"GRAD_DIFF {k} max={mx} mean={mn}")

    if dist.is_initialized():
        cleanup()


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


if __name__ == "__main__":
    main()
