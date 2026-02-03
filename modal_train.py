#!/usr/bin/env python3
"""
Modal deployment for Heiretsu distributed training.

Run with:
    modal run modal_train.py
"""

import modal
from pathlib import Path

# Get the local directory path
LOCAL_DIR = Path(__file__).parent

# Modal app definition
app = modal.App("heiretsu-training")

# Docker image with all dependencies + local code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "numpy",
        "einops",
        "huggingface_hub",
        "wandb",
    )
    .env({"TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING": "0"})
    .add_local_file(LOCAL_DIR / "train.py", remote_path="/app/train.py")
    .add_local_file(LOCAL_DIR / "gpt_model.py", remote_path="/app/gpt_model.py")
    .add_local_file(LOCAL_DIR / "topo.py", remote_path="/app/topo.py")
    .add_local_file(LOCAL_DIR / "dp.py", remote_path="/app/dp.py")
    .add_local_file(LOCAL_DIR / "tp_linear.py", remote_path="/app/tp_linear.py")
    .add_local_file(LOCAL_DIR / "pipeline.py", remote_path="/app/pipeline.py")
    .add_local_file(LOCAL_DIR / "moe.py", remote_path="/app/moe.py")
    .add_local_file(LOCAL_DIR / "ep_comm.py", remote_path="/app/ep_comm.py")
)

# Reference existing volumes and secrets
fineweb_volume = modal.Volume.from_name("fineweb-data")
wandb_secret = modal.Secret.from_name("wandb-secret")

# Training configuration
TRAINING_CONFIG = {
    # Model: GPT-2 Medium (~355M params)
    "n_layer": 24,
    "n_head": 16,
    "n_embed": 1024,
    "block_size": 1024,
    "dropout": 0.1,
    # MoE config (8 experts, top-2)
    "num_experts": 8,
    "top_k": 2,
    "moe_freq": 2,
    "aux_loss_coef": 0.01,
    # Parallelism: DP=2, TP=2 (to test TP seed fix)
    "dp": 2,
    "tp": 2,
    "pp": 1,
    "ep": 1,
    # Training
    "batch_size": 8,
    "grad_accum_steps": 4,
    "max_iters": 2000,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "amp": "bf16",
    # Eval
    "eval_interval": 100,
    "eval_iters": 50,
    # Data
    "num_train_chunks": 9,  # Use 9 chunks (~900M tokens) - matches available data
    "num_val_chunks": 1,
}


@app.function(
    image=image,
    gpu="a100-40gb:4",
    timeout=3600,  # 1 hour max
    secrets=[wandb_secret],
    volumes={"/data": fineweb_volume},
)
def train():
    """Run distributed training with torchrun."""
    import subprocess
    import sys
    import os
    
    os.chdir("/app")
    
    cfg = TRAINING_CONFIG
    world_size = cfg["dp"] * cfg["tp"] * cfg["pp"] * cfg["ep"]
    
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        "train.py",
        "--data_dir", "/data",
        f"--batch_size={cfg['batch_size']}",
        f"--block_size={cfg['block_size']}",
        f"--n_layer={cfg['n_layer']}",
        f"--n_head={cfg['n_head']}",
        f"--n_embed={cfg['n_embed']}",
        f"--dropout={cfg['dropout']}",
        f"--max_iters={cfg['max_iters']}",
        f"--learning_rate={cfg['learning_rate']}",
        f"--weight_decay={cfg['weight_decay']}",
        f"--grad_clip={cfg['grad_clip']}",
        f"--grad_accum_steps={cfg['grad_accum_steps']}",
        f"--amp={cfg['amp']}",
        f"--dp={cfg['dp']}",
        f"--tp={cfg['tp']}",
        f"--pp={cfg['pp']}",
        f"--ep={cfg['ep']}",
        f"--num_experts={cfg['num_experts']}",
        f"--top_k={cfg['top_k']}",
        f"--moe_freq={cfg['moe_freq']}",
        f"--aux_loss_coef={cfg['aux_loss_coef']}",
        f"--eval_interval={cfg['eval_interval']}",
        f"--eval_iters={cfg['eval_iters']}",
        f"--num_train_chunks={cfg['num_train_chunks']}",
        f"--num_val_chunks={cfg['num_val_chunks']}",
        "--wandb",
        "--wandb_project=heiretsu-moe-training",
        "--wandb_mode=online",
        "--run_name=heiretsu-moe-8k2-tp-fix",
        "--seed=1337",
    ]
    
    print("=" * 60)
    print("Heiretsu Training with TP Seeding Fix")
    print("=" * 60)
    print(f"Config: DP={cfg['dp']} TP={cfg['tp']} PP={cfg['pp']} EP={cfg['ep']}")
    print(f"Model: {cfg['n_layer']}L {cfg['n_head']}H {cfg['n_embed']}D")
    print(f"MoE: {cfg['num_experts']} experts, top-{cfg['top_k']}")
    print(f"Training: {cfg['max_iters']} steps, bs={cfg['batch_size']}, accum={cfg['grad_accum_steps']}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run training
    result = subprocess.run(cmd)
    return result.returncode


@app.local_entrypoint()
def main():
    """Entry point for `modal run modal_train.py`."""
    print("Starting Heiretsu training on Modal...")
    print("Using TP seeding fix (SmolLM3 Playbook)")
    print("=" * 60)
    
    returncode = train.remote()
    
    if returncode == 0:
        print("Training completed successfully!")
    else:
        print(f"Training failed with return code: {returncode}")
    
    return returncode
