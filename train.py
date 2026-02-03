#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, os, time, math, glob, random
from dataclasses import asdict
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist

from gpt_model import GPT, GPTConfig
from pipeline import StageModule, GPipeEngine
from topo import init_topology, cleanup
from dp import average_gradients

# Optional wandb (graceful fallback)
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore


# -------------------------
# Binary token loaders (uint16 GPT-2 ids)
# -------------------------

def list_split_bins(data_dir: str, split: str, num_chunks: Optional[int]) -> List[str]:
    """
    Finds split files:
      train: fineweb_train_%06d.bin
      val:   fineweb_val_%06d.bin
    If num_chunks is given, truncates the list (useful for quick tests).
    """
    pattern = os.path.join(data_dir, f"fineweb_{split}_*.bin")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {split} .bin files found under {data_dir} (looked for {pattern}).")
    if num_chunks is not None:
        files = files[:num_chunks]
    return files


class BinChunkSampler:
    """
    Wraps multiple uint16 memmaps and provides random contiguous slices of length (T+1)
    suitable for next-token prediction batches.

    Sampling is weighted by each file's available start positions (len - (T+1)).
    """
    def __init__(self, files: List[str], T: int):
        self.T = T
        self.arrs: List[np.memmap] = []
        self.lengths: List[int] = []
        self.starts_cap: List[int] = []  # how many valid start positions per file
        for f in files:
            m = np.memmap(f, dtype=np.uint16, mode="r")
            self.arrs.append(m)
            L = int(m.shape[0])
            self.lengths.append(L)
            cap = max(0, L - (T + 1))
            self.starts_cap.append(cap)
        self.total_cap = sum(self.starts_cap)
        if self.total_cap <= 0:
            raise ValueError("No valid start positions across files (increase T or use larger files).")
        # cumulative weights for weighted file sampling
        self.cum_caps = np.cumsum(self.starts_cap)

    def _sample_file_index(self) -> int:
        r = random.randrange(self.total_cap)  # [0, total_cap)
        # binary search cumulative caps
        return int(np.searchsorted(self.cum_caps, r, side="right"))

    def _sample_file_index_with_rng(self, rng: random.Random) -> int:
        r = rng.randrange(self.total_cap)
        return int(np.searchsorted(self.cum_caps, r, side="right"))

    def sample_one(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return one (x, y) pair of shape (T,) from a random file and start."""
        fi = self._sample_file_index()
        L = self.lengths[fi]
        start = random.randint(0, L - (self.T + 1))
        buf = self.arrs[fi][start : start + self.T + 1].astype(np.int64)  # upcast to fit vocab indices
        x = buf[:-1]
        y = buf[1:]
        return x, y


class RandomBatchLoader:
    def __init__(self, data_dir, split, B, T, num_chunks, seed=1337, rank=0, dp_rank=0, dp_size=1, log_rank=None):
        # Partition files across DP ranks to avoid overlap: files[dp_rank :: dp_size]
        files_all = list_split_bins(data_dir, split, num_chunks)
        files = files_all[dp_rank::dp_size] if dp_size > 1 else files_all
        if not files:
            raise FileNotFoundError(f"No {split} files assigned to dp_rank={dp_rank} out of {len(files_all)} total.")
        self.sampler = BinChunkSampler(files, T)
        self.B, self.T = B, T
        self.rng = random.Random(seed + rank)   # data-rank-sharded RNG
        log_rank = rank if log_rank is None else log_rank
        if log_rank == 0:
            print(f"[{split}] dp_shard={dp_rank}/{dp_size} files={len(files)} tokens={sum(self.sampler.lengths):,} T={T}")

    def next_batch(self, device):
        xs, ys = [], []
        for _ in range(self.B):
            fi = self.sampler._sample_file_index_with_rng(self.rng)
            L  = self.sampler.lengths[fi]
            start = self.rng.randint(0, L - (self.T + 1))
            buf = self.sampler.arrs[fi][start:start + self.T + 1].astype(np.int64)
            xs.append(buf[:-1]); ys.append(buf[1:])
        x = torch.tensor(np.stack(xs), dtype=torch.long, device=device)
        y = torch.tensor(np.stack(ys), dtype=torch.long, device=device)
        return x, y


# -------------------------
# Eval helper
# -------------------------
@torch.no_grad()
def estimate_loss(model, loader, eval_iters, device, amp_mode: str) -> float:
    model.eval()
    losses = []
    use_amp = amp_mode in ("fp16", "bf16") and device.type in ("cuda", "mps")
    dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    ctx = torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp)
    with ctx:
        for _ in range(eval_iters):
            x, y = loader.next_batch(device)
            _, loss, _ = model(x, y)  # (logits, loss, aux_loss)
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def estimate_loss_pipeline(engine: GPipeEngine, loader, eval_iters, device, amp_mode: str, micro_batches: int, dp_group) -> Optional[float]:
    engine.stage.eval()
    losses = []
    use_amp = amp_mode in ("fp16", "bf16") and device.type in ("cuda", "mps")
    dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    for _ in range(eval_iters):
        loss = engine.forward_only(
            loader,
            micro_batches=micro_batches,
            device=device,
            use_amp=use_amp,
            amp_dtype=dtype,
            batch_size=loader.B,
        )
        if loss is not None and dp_group is not None and dist.is_initialized():
            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=dp_group)
            loss = loss / dist.get_world_size(group=dp_group)
        if loss is not None:
            losses.append(float(loss.item()))
    engine.stage.train()
    if losses:
        return sum(losses) / len(losses)
    return None


# -------------------------
# Main
# -------------------------

def main():
    p = argparse.ArgumentParser(description="GPT trainer on FineWeb10B .bin tokens (AMP + grad accumulation + wandb)")

    # data / batching
    p.add_argument("--data_dir", type=str, default="fineweb10B", help="Directory containing fineweb_*_*.bin")
    p.add_argument("--num_train_chunks", type=int, default=None, help="Limit number of train bins (debug)")
    p.add_argument("--num_val_chunks", type=int, default=1, help="Limit number of val bins (e.g., 1 for fineweb_val_000000.bin)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--eval_interval", type=int, default=1000)
    p.add_argument("--eval_iters", type=int, default=200)

    # model
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embed", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.2)

    # optimization
    p.add_argument("--max_iters", type=int, default=50000)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)

    # AMP
    p.add_argument("--amp", type=str, choices=["none", "fp16", "bf16"], default="none")

    # device / seed
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=1337)

    # parallelism topo (DP/EP/TP/PP)
    p.add_argument("--dp", type=int, default=1, help="Data parallel degree")
    p.add_argument("--ep", type=int, default=1, help="Expert parallel degree (MoE)")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel degree")
    p.add_argument("--pp", type=int, default=1, help="Pipeline parallel degree")
    p.add_argument("--dp_share_data", action="store_true", help="Force all DP ranks to read identical data (parity/debug)")
    p.add_argument("--dist_backend", type=str, default="nccl", choices=["nccl", "gloo"], help="torch.distributed backend")

    # MoE config
    p.add_argument("--num_experts", type=int, default=0, help="Number of experts (0=dense model)")
    p.add_argument("--top_k", type=int, default=2, help="Experts per token")
    p.add_argument("--moe_freq", type=int, default=2, help="MoE every N layers (0=all dense)")
    p.add_argument("--aux_loss_coef", type=float, default=0.01, help="MoE load balancing loss coefficient")

    # wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="fineweb10B-gpt")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--save_interval", type=int, default=0, help="0=disable periodic checkpointing")

    args = p.parse_args()

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size_env > 1

    if args.device == "auto":
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device = torch.device("cuda", local_rank)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
            local_rank = 0
        else:
            device = torch.device("cpu")
            local_rank = 0
    else:
        device = torch.device(args.device)
        local_rank = device.index if device.type == "cuda" else 0

    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    if distributed:
        if args.dist_backend == "nccl" and device.type != "cuda":
            raise RuntimeError("nccl backend requires CUDA; use gloo or run on CUDA.")
        dist.init_process_group(backend=args.dist_backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    topo = init_topology(dp=args.dp, ep=args.ep, tp=args.tp, pp=args.pp)
    rank = topo.rank
    world_size = topo.world_size

    # model init must be identical across DP ranks -> seed with base seed only
    # (per-TP seeding is handled inside TP linear layers' reset_parameters)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # model / stage init (with MoE config)
    cfg = GPTConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embed=args.n_embed,
        dropout=args.dropout,
        # MoE config
        num_experts=args.num_experts,
        top_k=args.top_k,
        moe_freq=args.moe_freq,
        aux_loss_coef=args.aux_loss_coef,
        # Init config
        init_seed=args.seed,
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
    
    # Log MoE info
    if rank == 0 and args.num_experts > 0:
        moe_layers = sum(1 for i in range(args.n_layer) if args.moe_freq > 0 and (i + 1) % args.moe_freq == 0)
        print(f"[MoE] num_experts={args.num_experts}, top_k={args.top_k}, moe_layers={moe_layers}/{args.n_layer}, ep={args.ep}")

    # per-rank RNG:
    # - data RNG keyed by dp_rank so all PP/TP ranks within a DP replica share batches
    # - torch RNG keyed by (dp_rank, pp_rank) so TP ranks stay in sync for dropout
    # For parity/debug, optionally make all DP ranks consume identical data by sharing seed + file shard
    data_seed = args.seed if args.dp_share_data else args.seed + topo.dp_rank
    random.seed(data_seed)
    np.random.seed(data_seed)
    torch_seed = args.seed if args.dp_share_data else args.seed + topo.dp_rank * 1000 + topo.pp_rank
    torch.manual_seed(torch_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(torch_seed)

    # loaders (dp-rank sharded RNG)
    dp_rank_for_data = 0 if args.dp_share_data else topo.dp_rank
    dp_size_for_data = 1 if args.dp_share_data else topo.dp
    rank_for_data = 0 if args.dp_share_data else topo.dp_rank
    train_loader = RandomBatchLoader(
        args.data_dir, "train", args.batch_size, args.block_size,
        args.num_train_chunks, seed=args.seed, rank=rank_for_data,
        dp_rank=dp_rank_for_data, dp_size=dp_size_for_data, log_rank=rank
    )
    # keep val shared across ranks for now
    val_loader = RandomBatchLoader(
        args.data_dir, "val", args.batch_size, args.block_size,
        args.num_val_chunks, seed=args.seed + 999, rank=0,
        dp_rank=0, dp_size=1, log_rank=rank
    )

    optimizer = torch.optim.AdamW(
        local_module.parameters(),
        lr=args.learning_rate,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
    )

    # AMP
    use_amp = args.amp in ("fp16", "bf16") and device.type in ("cuda", "mps")
    amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler('cuda',enabled=(args.amp == "fp16"))
    is_main = (topo.dp_rank == 0 and topo.tp_rank == 0 and topo.pp_rank == (topo.pp - 1))

    def log_wandb(payload, step=None):
        if is_main and run is not None:
            wandb.log(payload, step=step)

    if is_main:
        print(f"device={device} | world_size={world_size} | dp={topo.dp} ep={topo.ep} tp={topo.tp} pp={topo.pp}")

    # wandb - only init on main rank
    run = None
    if is_main and args.wandb and WANDB_AVAILABLE and args.wandb_mode != "disabled":
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity else None,
            config={**vars(args), **asdict(cfg), "device": device.type},
            mode=args.wandb_mode,
            name=args.run_name if args.run_name else None,
        )
    elif is_main and args.wandb and not WANDB_AVAILABLE:
        print("wandb not available; proceeding without external logging.")

    # training
    local_module.train()
    t0 = time.time()
    tokens_per_micro = args.batch_size * args.block_size
    micro_in_macro = args.grad_accum_steps
    optimizer.zero_grad(set_to_none=True)
    last_log_time = time.time()

    for step in range(1, args.max_iters + 1):
        if use_pipeline:
            loss_scalar, aux_loss = engine.forward_backward(
                train_loader,
                micro_batches=micro_in_macro,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
                batch_size=args.batch_size,
                aux_loss_coef=args.aux_loss_coef,
            )
        else:
            # accumulate on a single stage (DP/TP only)
            total_aux_loss = torch.tensor(0.0, device=device)
            for _ in range(1, micro_in_macro + 1):
                x, y = train_loader.next_batch(device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    _, loss, aux_loss = local_module(x, y)
                    # Combined loss = data_loss + aux_loss (aux_loss already scaled by coef in MoE layer)
                    combined_loss = (loss + aux_loss) / micro_in_macro
                    total_aux_loss = total_aux_loss + aux_loss.detach()

                if scaler.is_enabled():
                    scaler.scale(combined_loss).backward()
                else:
                    combined_loss.backward()
            loss_scalar = loss.detach().float()
            aux_loss = total_aux_loss

        if loss_scalar is not None and topo.dp_group is not None and dist.is_initialized():
            dist.all_reduce(loss_scalar, op=dist.ReduceOp.SUM, group=topo.dp_group)
            loss_scalar = loss_scalar / dist.get_world_size(group=topo.dp_group)
        if aux_loss is not None and topo.dp_group is not None and dist.is_initialized():
            dist.all_reduce(aux_loss, op=dist.ReduceOp.SUM, group=topo.dp_group)
            aux_loss = aux_loss / dist.get_world_size(group=topo.dp_group)

        # unscale first when using AMP
        if scaler.is_enabled():
            scaler.unscale_(optimizer)

        # DP grad sync: p.grad -> all_reduce(mean, dp_group)
        average_gradients(local_module, topo.dp_group)

        # clip
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(local_module.parameters(), args.grad_clip)

        # step
        if scaler.is_enabled():
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # throughput (log rank prints)
        now = time.time()
        dt = now - last_log_time
        eff_toks_per_sec = (tokens_per_micro * micro_in_macro) / dt if dt > 0 else float("inf")
        last_log_time = now

        if is_main and ((step % max(1, args.eval_interval // 10)) == 0 or step == 1):
            aux_str = f" | aux_loss {aux_loss.item():.4f}" if args.num_experts > 0 else ""
            if loss_scalar is not None:
                print(f"step {step:6d} | train_loss (mean micro) {loss_scalar.item():.4f}{aux_str} | eff_tok/s {eff_toks_per_sec:,.0f}")
            else:
                print(f"step {step:6d} | train_loss (mean micro) n/a{aux_str} | eff_tok/s {eff_toks_per_sec:,.0f}")

        if is_main and run:
            log_dict = {
                "train/loss_mean_micro": float(loss_scalar.item()) if loss_scalar is not None else float("nan"),
                "opt/lr": float(optimizer.param_groups[0]["lr"]),
                "speed/eff_tokens_per_sec": eff_toks_per_sec,
                "amp/enabled": int(use_amp),
                "amp/mode": args.amp,
                "accum/steps": args.grad_accum_steps,
                "step": step,
            }
            if args.num_experts > 0:
                log_dict["train/aux_loss"] = float(aux_loss.item())
            wandb.log(log_dict, step=step)

        # periodic eval
        if (step % args.eval_interval) == 0 or step == args.max_iters:
            if use_pipeline:
                eval_loss = estimate_loss_pipeline(engine, val_loader, args.eval_iters, device, args.amp, micro_in_macro, topo.dp_group)
            else:
                eval_loss = estimate_loss(local_module, val_loader, args.eval_iters, device, args.amp)
            if is_main and eval_loss is not None:
                print(f"[eval] step {step} | eval_loss {eval_loss:.4f} | elapsed {time.time()-t0:.1f}s")
                if run:
                    wandb.log({"eval/loss": eval_loss, "step": step}, step=step)

        # checkpointing
        if args.save_interval and (step % args.save_interval == 0) and (topo.dp_rank == 0):
            if topo.tp == 1 and topo.pp == 1 and topo.ep == 1 and is_main:
                torch.save({"model": local_module.state_dict(), "cfg": asdict(cfg), "step": step}, f"ckpt_{step:07d}.pt")
            else:
                torch.save({
                    "model": local_module.state_dict(),
                    "cfg": asdict(cfg),
                    "step": step,
                    "topo": {"dp": topo.dp, "ep": topo.ep, "tp": topo.tp, "pp": topo.pp, 
                             "pp_rank": topo.pp_rank, "tp_rank": topo.tp_rank, "ep_rank": topo.ep_rank},
                }, f"ckpt_{step:07d}_pp{topo.pp_rank}_ep{topo.ep_rank}_tp{topo.tp_rank}.pt")

    if is_main and run:
        run.finish()
        print("training finished.")

    if distributed and dist.is_initialized():
        cleanup()


if __name__ == "__main__":
    main()
