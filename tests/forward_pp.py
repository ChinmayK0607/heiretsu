#!/usr/bin/env python3
import os
import torch
import torch.distributed as dist
import numpy as np

from gpt_model import GPT, GPTConfig
from topo import init_topology
from pipeline import StageModule
from train import RandomBatchLoader

# settings
DATA_DIR = os.environ.get("DATA_DIR", "/root/heiretsu/heiretsu/data/fineweb10B")
B = int(os.environ.get("B", 4))
T = int(os.environ.get("T", 32))
SEED = int(os.environ.get("SEED", 1234))
OUT = os.environ.get("OUT", "loss_pp.npy")
BACKEND = os.environ.get("BACKEND", "nccl")
PP = int(os.environ.get("PP", 2))

local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

dist.init_process_group(backend=BACKEND)
rank = dist.get_rank()
world = dist.get_world_size()
assert world == world_size, f"world_size mismatch {world} vs env {world_size}"
assert world == PP, f"PP mismatch: world_size {world} vs PP {PP}"

# deterministic weights across PP ranks
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# topology and stage model
cfg = GPTConfig(block_size=T)
topo = init_topology(dp=1, tp=1, pp=PP)
stage = StageModule(cfg, topo, topo.pp_rank, topo.pp)
stage.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
stage.eval()

# build a full GPT on rank0 and copy weights into each stage
if rank == 0:
    torch.manual_seed(SEED)
    full = GPT(cfg)
    full_state = full.state_dict()
else:
    full_state = None

# broadcast full_state to all ranks
obj_list = [full_state]
dist.broadcast_object_list(obj_list, src=0)
full_state = obj_list[0]

# copy weights into this stage
if stage.is_first:
    stage.wte.load_state_dict({"weight": full_state["tr.wte.weight"]})
    stage.wpe.load_state_dict({"weight": full_state["tr.wpe.weight"]})
if stage.is_last:
    stage.ln_f.load_state_dict({
        "weight": full_state["tr.ln_f.weight"],
        "bias": full_state["tr.ln_f.bias"],
    })
    stage.lm_head.load_state_dict({"weight": full_state["lm_head.weight"]})

# blocks
for local_i, global_i in enumerate(range(stage.block_start, stage.block_end)):
    block = stage.blocks[local_i]
    prefix = f"tr.h.{global_i}."
    block.load_state_dict({
        "ln_1.weight": full_state[prefix + "ln_1.weight"],
        "ln_1.bias": full_state[prefix + "ln_1.bias"],
        "attn.c_attn.weight": full_state[prefix + "attn.c_attn.weight"],
        "attn.c_attn.bias": full_state[prefix + "attn.c_attn.bias"],
        "attn.c_proj.weight": full_state[prefix + "attn.c_proj.weight"],
        "attn.c_proj.bias": full_state[prefix + "attn.c_proj.bias"],
        "ln_2.weight": full_state[prefix + "ln_2.weight"],
        "ln_2.bias": full_state[prefix + "ln_2.bias"],
        "mlp.c_fc.weight": full_state[prefix + "mlp.c_fc.weight"],
        "mlp.c_fc.bias": full_state[prefix + "mlp.c_fc.bias"],
        "mlp.c_proj.weight": full_state[prefix + "mlp.c_proj.weight"],
        "mlp.c_proj.bias": full_state[prefix + "mlp.c_proj.bias"],
    })

# data: rank0 loads, broadcast to all for loss parity
loader = RandomBatchLoader(DATA_DIR, "val", B, T, num_chunks=1, seed=SEED, rank=0, dp_rank=0, dp_size=1)
if rank == 0:
    x, y = loader.next_batch(stage.lm_head.weight.device if stage.is_last else stage.blocks[0].ln_1.weight.device)
else:
    x = torch.empty((B, T), device=stage.blocks[0].ln_1.weight.device, dtype=torch.long)
    y = torch.empty((B, T), device=stage.blocks[0].ln_1.weight.device, dtype=torch.long)

dist.broadcast(x, src=0)
dist.broadcast(y, src=0)

# PP forward: single microbatch
if stage.is_first:
    with torch.no_grad():
        h = stage.embed(x)
        h = stage.forward_blocks(h)
    # send boundary to next
    if topo.pp_rank < topo.pp - 1:
        dist.send(h, dst=topo.rank + 1, tag=1000)
elif stage.is_last:
    # recv boundary
    h = torch.empty((B, T, cfg.n_embed), device=stage.blocks[0].ln_1.weight.device)
    dist.recv(h, src=topo.rank - 1, tag=1000)
    with torch.no_grad():
        h = stage.forward_blocks(h)
        logits, loss = stage.forward_head(h, y)
    loss_val = loss.detach().float().cpu().numpy()
else:
    # middle stages (if PP>2): recv, run blocks, send
    h = torch.empty((B, T, cfg.n_embed), device=stage.blocks[0].ln_1.weight.device)
    dist.recv(h, src=topo.rank - 1, tag=1000)
    with torch.no_grad():
        h = stage.forward_blocks(h)
    dist.send(h, dst=topo.rank + 1, tag=1000)

# compare vs single on rank0
if rank == 0:
    torch.manual_seed(SEED)
    full = GPT(cfg)
    full.eval()
    with torch.no_grad():
        _, ref_loss = full(x.cpu(), y.cpu())
    ref_loss_val = ref_loss.detach().float().cpu().numpy()
else:
    ref_loss_val = None

obj_list = [ref_loss_val]
dist.broadcast_object_list(obj_list, src=0)
ref_loss_val = obj_list[0]

if stage.is_last:
    np.save(OUT, loss_val)
    diff = abs(loss_val - ref_loss_val)
    print(f"PP loss {loss_val} | ref {ref_loss_val} | diff {diff}")

if rank == 0:
    print("PP forward test complete")

dist.barrier()
dist.destroy_process_group()
