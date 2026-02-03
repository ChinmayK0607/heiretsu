#!/usr/bin/env python3
"""
Modal deployment for Heiretsu distributed training.

Run with:
    modal run modal_train.py

Or deploy as a scheduled job:
    modal deploy modal_train.py
"""

import modal
import os

# Modal app definition
app = modal.App("heiretsu-training")

# Docker image with all dependencies
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
    "num_train_chunks": 10,  # Use 10 chunks (~1B tokens) for faster startup
    "num_val_chunks": 1,
}


@app.function(
    image=image,
    gpu="a100-40gb:4",
    timeout=3600,  # 1 hour max
    secrets=[wandb_secret],
    volumes={"/data": fineweb_volume},
    _experimental_scheduler_placement=modal.scheduler_placement.Scheduler(
        # Request all 4 GPUs on the same node for NCCL
        zone="us-east-1",
    ),
)
def train():
    """Run distributed training with torchrun."""
    import subprocess
    import sys
    
    # Copy training code to container
    # (In production, you'd mount the code or include it in the image)
    code_files = [
        "train.py",
        "gpt_model.py", 
        "topo.py",
        "dp.py",
        "tp_linear.py",
        "pipeline.py",
        "moe.py",
        "ep_comm.py",
    ]
    
    # For now, we'll download from the mounted volume or include inline
    # This example assumes the code is copied to /app
    
    # Build torchrun command
    cfg = TRAINING_CONFIG
    world_size = cfg["dp"] * cfg["tp"] * cfg["pp"] * cfg["ep"]
    
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        "train.py",
        "--data_dir", "/data/fineweb10B",
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
        f"--run_name=heiretsu-moe-8k2-tp-fix",
        "--seed=1337",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Config: DP={cfg['dp']} TP={cfg['tp']} PP={cfg['pp']} EP={cfg['ep']}")
    print(f"Model: {cfg['n_layer']}L {cfg['n_head']}H {cfg['n_embed']}D")
    print(f"MoE: {cfg['num_experts']} experts, top-{cfg['top_k']}")
    print(f"Training: {cfg['max_iters']} steps, bs={cfg['batch_size']}, accum={cfg['grad_accum_steps']}")
    
    # Run training
    result = subprocess.run(cmd, cwd="/app")
    return result.returncode


# Alternative: Mount local code directly
@app.function(
    image=image,
    gpu="a100-40gb:4",
    timeout=3600,
    secrets=[wandb_secret],
    volumes={"/data": fineweb_volume},
    mounts=[
        modal.Mount.from_local_dir(
            ".",
            remote_path="/app",
            condition=lambda path: path.endswith(".py") and "modal" not in path,
        )
    ],
)
def train_with_local_code():
    """Run distributed training with locally mounted code."""
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
        "--data_dir", "/data/fineweb10B",
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
        f"--run_name=heiretsu-moe-8k2-tp-fix",
        "--seed=1337",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Config: DP={cfg['dp']} TP={cfg['tp']} PP={cfg['pp']} EP={cfg['ep']}")
    print(f"Model: {cfg['n_layer']}L {cfg['n_head']}H {cfg['n_embed']}D")
    print(f"MoE: {cfg['num_experts']} experts, top-{cfg['top_k']}")
    print(f"Training: {cfg['max_iters']} steps")
    
    result = subprocess.run(cmd)
    return result.returncode


@app.local_entrypoint()
def main():
    """Entry point for `modal run modal_train.py`."""
    print("Starting Heiretsu training on Modal...")
    print("Using TP seeding fix (SmolLM3 Playbook)")
    print("=" * 60)
    
    # Use the version with local code mounting
    returncode = train_with_local_code.remote()
    
    if returncode == 0:
        print("Training completed successfully!")
    else:
        print(f"Training failed with return code: {returncode}")
    
    return returncode
