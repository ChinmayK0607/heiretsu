#!/usr/bin/env python3
import os
import torch
import torch.distributed as dist
import numpy as np
from gpt_model import GPT, GPTConfig
from topo import init_topology
from train import RandomBatchLoader

# settings
DATA_DIR = os.environ.get("DATA_DIR", "/root/heiretsu/heiretsu/data/fineweb10B")
B = int(os.environ.get("B", 4))
T = int(os.environ.get("T", 32))
SEED = int(os.environ.get("SEED", 1234))
OUT = os.environ.get("OUT", "logits_tp.npy")
BACKEND = os.environ.get("BACKEND", "nccl")
TP = int(os.environ.get("TP", 2))

local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

dist.init_process_group(backend=BACKEND)
rank = dist.get_rank()
world = dist.get_world_size()
assert world == world_size, f"world_size mismatch {world} vs env {world_size}"
assert world == TP, f"TP mismatch: world_size {world} vs TP {TP}"

# deterministic weights across TP ranks
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# topology and model
cfg = GPTConfig(block_size=T)
topo = init_topology(dp=1, tp=TP, pp=1)
model = GPT(cfg, topo=topo)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.eval()

# per-rank RNG for data, but batch will be broadcast from rank0
loader = RandomBatchLoader(DATA_DIR, "val", B, T, num_chunks=1, seed=SEED, rank=rank, dp_rank=0, dp_size=1)
x, _ = loader.next_batch(model.lm_head.weight.device)

# broadcast batch from rank0 to ensure identical inputs
if rank == 0:
    x_bcast = x
else:
    x_bcast = torch.empty_like(x)
dist.broadcast(x_bcast, src=0)

with torch.no_grad():
    logits, _ = model(x_bcast)

# check TP parity (all ranks vs rank0)
logits_ref = logits.detach().clone()
dist.broadcast(logits_ref, src=0)
diff = (logits - logits_ref).abs()
diff_max = diff.max().item()
diff_mean = diff.mean().item()
diff_max_t = torch.tensor([diff_max], device=logits.device)
diff_mean_t = torch.tensor([diff_mean], device=logits.device)
dist.all_reduce(diff_max_t, op=dist.ReduceOp.MAX)
dist.all_reduce(diff_mean_t, op=dist.ReduceOp.MAX)

if rank == 0:
    np.save(OUT, logits.float().cpu().numpy())
    print(f"saved {OUT} shape {logits.shape}")
    print(f"TP parity diff_max {diff_max_t.item()} diff_mean {diff_mean_t.item()}")

dist.barrier()
dist.destroy_process_group()
