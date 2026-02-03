#!/usr/bin/env python3
import os
import torch
import numpy as np
from gpt_model import GPT, GPTConfig
from train import RandomBatchLoader

data_dir = os.environ.get("DATA_DIR", "/root/heiretsu/heiretsu/data/fineweb10B")
B = int(os.environ.get("B", 4))
T = int(os.environ.get("T", 32))
seed = int(os.environ.get("SEED", 1234))
ckpt = os.environ.get("CKPT", None)
out_path = os.environ.get("OUT", "logits_single.npy")

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if ckpt:
    state = torch.load(ckpt, map_location="cpu")
    cfg = GPTConfig(**state["cfg"])
    model = GPT(cfg)
    model.load_state_dict(state["model"])
else:
    cfg = GPTConfig(block_size=T)
    model = GPT(cfg)
model.eval()

loader = RandomBatchLoader(data_dir, "val", B, T, num_chunks=1, seed=seed, rank=0, dp_rank=0, dp_size=1)
x, _ = loader.next_batch(torch.device("cpu"))
with torch.no_grad():
    logits, _, _ = model(x)
np.save(out_path, logits.float().cpu().numpy())
print(f"saved {out_path} shape {logits.shape}")
