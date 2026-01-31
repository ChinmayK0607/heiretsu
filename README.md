Minimal 3D parallelism (DP + TP + PP) in pure PyTorch.

What’s here
- Manual DP gradient averaging (no DDP/FSDP).
- Megatron-style TP (column/row parallel linears) for attention + MLP.
- GPipe-style PP with microbatching (fill/drain schedule).
- Optional AMP (fp16/bf16).

Quick start
- Single GPU:
  python train.py --device auto
- Data parallel:
  torchrun --standalone --nproc_per_node=2 train.py --dp 2
- Tensor parallel:
  torchrun --standalone --nproc_per_node=2 train.py --tp 2
- Pipeline parallel:
  torchrun --standalone --nproc_per_node=2 train.py --pp 2 --grad_accum_steps 4

Notes
- PP uses `--grad_accum_steps` as the microbatch count.
- Embedding/LM-head weight tying is only enabled when pp=1.

TP parity test
- torchrun --standalone --nproc_per_node=2 tests_equiv.py --tp 2
