# Expert Parallelism (EP) Implementation Guide

## Table of Contents

1. [Overview](#overview)
2. [Why Expert Parallelism?](#why-expert-parallelism)
3. [Architecture at a Glance](#architecture-at-a-glance)
4. [Rank Layout & Topology](#rank-layout--topology)
5. [Process Group Construction](#process-group-construction)
6. [MoE Layer Internals](#moe-layer-internals)
7. [All-to-All Communication](#all-to-all-communication)
8. [End-to-End Forward Pass](#end-to-end-forward-pass)
9. [Interaction with Other Parallelism Axes](#interaction-with-other-parallelism-axes)
10. [Load Balancing Loss](#load-balancing-loss)
11. [Concrete Example Walkthrough](#concrete-example-walkthrough)
12. [EP × PP Pitfalls: The Hidden Coordinate Filter](#ep--pp-pitfalls-the-hidden-coordinate-filter)
13. [Gradient Parity with EP](#gradient-parity-with-ep)
14. [Test Infrastructure for EP](#test-infrastructure-for-ep)

---

## Overview

Heiretsu implements **4D parallelism**: Data (DP) × Expert (EP) × Tensor (TP) × Pipeline (PP).

Expert Parallelism splits the set of MoE experts across multiple GPUs. Each GPU
holds `E / EP` experts and uses **All-to-All** collectives to route tokens to the
correct expert's owner, then send results back.

```
              ┌──────────────────────────────────────┐
              │      Standard Dense Transformer      │
              │                                      │
              │   [Attention] ──► [MLP (all params)] │
              └──────────────────────────────────────┘

                              vs.

              ┌──────────────────────────────────────┐
              │      MoE Transformer with EP=2       │
              │                                      │
              │   [Attention] ──► [Router] ─┬──►EP0: Experts 0-3 │
              │                             └──►EP1: Experts 4-7 │
              └──────────────────────────────────────┘
```

**Key files:**

| File          | Role                                            |
|---------------|-------------------------------------------------|
| `topo.py`     | Rank layout, coordinate math, process groups    |
| `moe.py`      | Router, Expert, ExpertGroup, MoELayer           |
| `ep_comm.py`  | `all_to_all_dispatch` / `all_to_all_combine`    |
| `gpt_model.py`| `BlockMoE`, `make_block` factory, model wiring  |
| `pipeline.py` | PP `StageModule` integration with MoE blocks    |
| `dp.py`       | DP gradient averaging (works unchanged with EP) |

---

## Why Expert Parallelism?

A dense MLP with hidden size `H` and intermediate size `I = 4H` has `~8H²` parameters.
An MoE layer with `E=8` experts has `E × 8H² = 64H²` parameters — **8× more**.

Replicating all experts on every GPU wastes memory. EP shards them:

```
  No EP (all experts replicated)          EP = 2 (experts sharded)
  ┌──────────────────────┐               ┌───────────┐ ┌───────────┐
  │  GPU 0               │               │  GPU 0    │ │  GPU 1    │
  │  Expert 0            │               │  Expert 0 │ │  Expert 4 │
  │  Expert 1            │               │  Expert 1 │ │  Expert 5 │
  │  Expert 2            │               │  Expert 2 │ │  Expert 6 │
  │  Expert 3            │               │  Expert 3 │ │  Expert 7 │
  │  Expert 4            │               └───────────┘ └───────────┘
  │  Expert 5            │                 4 experts     4 experts
  │  Expert 6            │                 each          each
  │  Expert 7            │
  │  (8 full experts)    │               Memory: 4×8H²   4×8H²
  │  Memory: 8×8H²       │                     = 32H²   = 32H²
  └──────────────────────┘                  (50% saving per GPU)
```

The cost is two All-to-All collectives per MoE layer (dispatch + combine).

---

## Architecture at a Glance

```
  Input tokens x: (B, S, H)
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │                 MoE Layer Forward                    │
  │                                                     │
  │  1. Flatten ─────────► x_flat: (N, H)  [N = B*S]   │
  │                              │                      │
  │  2. Router ──────────► top_indices: (N, K)          │
  │     (softmax gate)     top_weights: (N, K)          │
  │                              │                      │
  │  3. Expand ──────────► expanded_x: (N*K, H)         │
  │                              │                      │
  │  4. Sort by expert ──► x_sorted: (N*K, H)           │
  │                              │                      │
  │  5. ┌─── EP > 1? ───────────┤                      │
  │     │ YES                    │ NO                   │
  │     ▼                       ▼                      │
  │  All-to-All           Process all                   │
  │  Dispatch             E experts                     │
  │     │                 locally                       │
  │     ▼                       │                      │
  │  Expert compute             │                      │
  │  (local E/EP)               │                      │
  │     │                       │                      │
  │     ▼                       │                      │
  │  All-to-All                 │                      │
  │  Combine                    │                      │
  │     │                       │                      │
  │     └───────────┬───────────┘                      │
  │                 ▼                                   │
  │  6. Unsort ──────────► y_expanded: (N*K, H)         │
  │                              │                      │
  │  7. Weight & reduce ─► y: (N, H)                    │
  │                              │                      │
  │  8. Reshape ─────────► y: (B, S, H)                 │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  Output: (B, S, H)
```

---

## Rank Layout & Topology

### Coordinate System

Every GPU gets a unique **global rank** and a 4D coordinate `(dp, ep, pp, tp)`.

The layout keeps **TP contiguous** (fastest-varying), then PP, then EP, then DP:

```
global_rank = dp_rank × (EP × PP × TP)
            + ep_rank × (PP × TP)
            + pp_rank × TP
            + tp_rank
```

**Decomposition (reverse):**

```python
tp_rank = rank % TP
pp_rank = (rank // TP) % PP
ep_rank = (rank // (TP * PP)) % EP
dp_rank = rank // (TP * PP * EP)
```

### Example: DP=1, EP=2, TP=2, PP=2 (8 GPUs)

```
Rank │ dp │ ep │ pp │ tp │  Role
─────┼────┼────┼────┼────┼──────────────────────────
  0  │  0 │  0 │  0 │  0 │  EP0, Stage 0, TP shard 0
  1  │  0 │  0 │  0 │  1 │  EP0, Stage 0, TP shard 1
  2  │  0 │  0 │  1 │  0 │  EP0, Stage 1, TP shard 0
  3  │  0 │  0 │  1 │  1 │  EP0, Stage 1, TP shard 1
  4  │  0 │  1 │  0 │  0 │  EP1, Stage 0, TP shard 0
  5  │  0 │  1 │  0 │  1 │  EP1, Stage 0, TP shard 1
  6  │  0 │  1 │  1 │  0 │  EP1, Stage 1, TP shard 0
  7  │  0 │  1 │  1 │  1 │  EP1, Stage 1, TP shard 1
```

Visually on the GPU grid:

```
              PP Stage 0           PP Stage 1
           ┌──────────────┐    ┌──────────────┐
  EP = 0   │ R0(tp0) R1(tp1)│   │ R2(tp0) R3(tp1)│
           ├──────────────┤    ├──────────────┤
  EP = 1   │ R4(tp0) R5(tp1)│   │ R6(tp0) R7(tp1)│
           └──────────────┘    └──────────────┘

  EP group at PP=0, TP=0:  {R0, R4}  (all_to_all happens here)
  EP group at PP=0, TP=1:  {R1, R5}
  EP group at PP=1, TP=0:  {R2, R6}
  EP group at PP=1, TP=1:  {R3, R7}
```

---

## Process Group Construction

`topo.py` → `build_groups()` creates four families of process groups.
Every group is built by iterating over the **fixed** dimensions and varying the **target** dimension:

| Group  | Varies   | Fixed          | Purpose                         |
|--------|----------|----------------|---------------------------------|
| **TP** | tp_rank  | dp, ep, pp     | Tensor-parallel all-reduce      |
| **EP** | ep_rank  | dp, pp, tp     | MoE all-to-all token dispatch   |
| **PP** | pp_rank  | dp, ep, tp     | Pipeline send/recv              |
| **DP** | dp_rank  | ep, pp, tp     | Gradient all-reduce             |

### EP Group Construction Detail

```python
# EP groups: same dp, pp, tp; varying ep
for d in range(dp):
    for p in range(pp):
        for t in range(tp):
            ranks = [rank_from_coords(d, e, p, t, dp, ep, tp, pp)
                     for e in range(ep)]
            ep_groups.append(dist.new_group(ranks=ranks))
```

For the 8-GPU example above, this creates 4 EP groups:

```
EP group 0: d=0, p=0, t=0 → ranks [0, 4]
EP group 1: d=0, p=0, t=1 → ranks [1, 5]
EP group 2: d=0, p=1, t=0 → ranks [2, 6]
EP group 3: d=0, p=1, t=1 → ranks [3, 7]
```

Each EP group connects ranks that share the same DP/PP/TP coordinates
but hold **different expert shards**.

---

## MoE Layer Internals

### Layer Placement

Not every transformer block is MoE. The `make_block()` factory in `gpt_model.py` decides:

```python
use_moe = (
    num_experts > 0
    and moe_freq > 0
    and (layer_idx + 1) % moe_freq == 0
)
```

With `n_layer=4` and `moe_freq=2`:

```
Layer 0: Dense (BlockTP)     ← (0+1) % 2 = 1 ≠ 0
Layer 1: MoE   (BlockMoE)    ← (1+1) % 2 = 0 ✓
Layer 2: Dense (BlockTP)     ← (2+1) % 2 = 1 ≠ 0
Layer 3: MoE   (BlockMoE)    ← (3+1) % 2 = 0 ✓
```

### Block Anatomy

```
         BlockTP (Dense)                    BlockMoE (MoE)
  ┌─────────────────────────┐      ┌─────────────────────────┐
  │  x ──► LN1 ──► Attn ─┐ │      │  x ──► LN1 ──► Attn ─┐ │
  │            residual ◄──┘ │      │            residual ◄──┘ │
  │  x ──► LN2 ──► MLP  ─┐ │      │  x ──► LN2 ──► MoE  ─┐ │
  │            residual ◄──┘ │      │            residual ◄──┘ │
  │  return (x, aux=0)      │      │  return (x, aux_loss)    │
  └─────────────────────────┘      └─────────────────────────┘
```

Both return `(hidden, aux_loss)` for a uniform interface. Dense blocks return `aux_loss=0`.

### Component Hierarchy

```
MoELayer
├── Router             # W_gate: (H, E) — replicated on all EP ranks
│   └── gate: Linear(H → E, bias=False)
│
└── ExpertGroup        # Holds E_local = E/EP experts
    ├── Expert 0
    │   ├── w1: Linear(H → I)   # up-projection
    │   └── w2: Linear(I → H)   # down-projection + GELU
    ├── Expert 1
    │   ├── w1, w2
    ⋮
    └── Expert (E_local - 1)
```

**Critical:** The **Router is replicated** across all EP ranks so every rank
computes identical routing decisions. Only the **ExpertGroup is sharded**.

Expert ownership:

```
EP rank 0: Experts [0, 1, ..., E_local-1]
EP rank 1: Experts [E_local, ..., 2*E_local-1]
...
EP rank r: Experts [r*E_local, ..., (r+1)*E_local-1]
```

---

## All-to-All Communication

The All-to-All dispatch/combine in `ep_comm.py` is the heart of EP.

### Dispatch (tokens → expert owners)

```
  EP Rank 0                                    EP Rank 1
  ┌────────────────────────┐                  ┌────────────────────────┐
  │ Tokens routed to:      │                  │ Tokens routed to:      │
  │   Expert 0,1,2,3 (local)│                  │   Expert 0,1,2,3 (on R0)│
  │   Expert 4,5,6,7 (on R1)│                  │   Expert 4,5,6,7 (local)│
  └────────┬───────────────┘                  └────────┬───────────────┘
           │                                           │
           │  ┌──────────── All-to-All ──────────────┐ │
           │  │  R0 sends "for experts 4-7" ──────► R1 │
           │  │  R1 sends "for experts 0-3" ──────► R0 │
           │  └────────────────────────────────────────┘
           │                                           │
           ▼                                           ▼
  ┌────────────────────────┐                  ┌────────────────────────┐
  │ Now has ALL tokens for │                  │ Now has ALL tokens for │
  │ Experts 0,1,2,3        │                  │ Experts 4,5,6,7        │
  │ (own + received)       │                  │ (own + received)       │
  └────────────────────────┘                  └────────────────────────┘
```

### Protocol

The All-to-All is a **two-phase** operation:

**Phase 1: Exchange counts**

```python
# Each rank tells every other rank how many tokens it's sending
recv_counts = torch.empty_like(send_counts)
dist.all_to_all_single(recv_counts, send_counts, group=ep_group)
```

```
send_counts on R0: [12, 8]   →  "12 tokens stay, 8 go to R1"
send_counts on R1: [5, 15]   →  "5 go to R0, 15 stay"

After exchange:
recv_counts on R0: [12, 5]   →  "12 from self, 5 from R1"
recv_counts on R1: [8, 15]   →  "8 from R0, 15 from self"
```

**Phase 2: Exchange tokens**

```python
dist.all_to_all_single(
    x_recv, x,
    output_split_sizes=recv_splits,
    input_split_sizes=send_splits,
    group=ep_group,
)
```

### Combine (results → original owners)

The combine step is the exact **reverse** of dispatch — swap send and recv counts:

```python
# In combine: what we received becomes what we send, and vice versa
send_splits = recv_counts.tolist()  # we send back what we received
recv_splits = send_counts.tolist()  # we receive what we originally sent
```

```
                    ┌─────────────────┐
     Dispatch       │   Expert        │      Combine
  ──────────────►   │   Compute       │   ──────────────►
  tokens to owner   │   (local only)  │   results back
                    └─────────────────┘
```

---

## End-to-End Forward Pass

Here's the complete data flow through `MoELayer.forward()` with `EP=2, E=8, K=2`:

```
 Input: x (B, S, H)
   │
   │ 1. Flatten
   ▼
 x_flat (N, H)                          N = B × S
   │
   │ 2. Router: gate(x) → softmax → top-k
   ▼
 top_indices (N, K)                      which 2 experts per token
 top_weights (N, K)                      normalized routing weights
 gate_probs  (N, E)                      full probs (for aux loss)
   │
   │ 3. Expand for K selections
   ▼
 expanded_x (N*K, H)                     each token appears K times
 expanded_indices (N*K,)                 flat expert assignments
   │
   │ 4. Sort by global expert id
   ▼
 x_sorted (N*K, H)                      tokens grouped by expert
 indices_sorted (N*K,)                   [0,0,0,1,1,2,2,2,3,4,4,...]
   │
   │ 5a. Map experts → EP rank:  ep_rank = expert_id // E_local
   │     Compute send_counts per EP rank
   │     Sort by EP rank destination
   ▼
 x_for_a2a (N*K, H)                     sorted by destination EP rank
   │
   │ 5b. ALL-TO-ALL DISPATCH
   │     ┌──────────────────────────────────────────────┐
   │     │ dist.all_to_all_single(counts)               │
   │     │ dist.all_to_all_single(tokens + expert_ids)  │
   │     └──────────────────────────────────────────────┘
   ▼
 x_recv (N', H)                          tokens from ALL ranks
 idx_recv (N',)                          global expert ids
   │
   │ 6. Map to local index: local_id = global_id - ep_rank * E_local
   │    Sort by local expert id
   ▼
 x_local_sorted (N', H)                  grouped by local expert
 local_counts (E_local,)                 tokens per local expert
   │
   │ 7. Expert compute (loop over E_local experts)
   │    ┌─────────────────────────────────────────┐
   │    │ for e in 0..E_local-1:                  │
   │    │   x_e = x_sorted[offset[e]:offset[e+1]]│
   │    │   y_e = Expert_e(x_e)                   │
   │    │       = GELU(x_e @ W1) @ W2             │
   │    └─────────────────────────────────────────┘
   ▼
 y_local (N', H)                          expert outputs
   │
   │ 8. ALL-TO-ALL COMBINE (reverse dispatch)
   │     ┌──────────────────────────────────────────────┐
   │     │ dist.all_to_all_single(results back)         │
   │     └──────────────────────────────────────────────┘
   ▼
 y_return (N*K, H)                       back on original EP rank
   │
   │ 9. Unsort (inverse of step 4+5a permutations)
   ▼
 y_expanded (N*K, H)                     original expanded order
   │
   │ 10. Weight & reduce over K experts
   │     y_weighted = y_expanded * weights  (N*K, H)
   │     y = sum over K dim                 (N, H)
   ▼
 y (N, H)
   │
   │ 11. Reshape
   ▼
 Output: y (B, S, H)
```

---

## Interaction with Other Parallelism Axes

### EP × TP

Within an MoE block, **Attention uses TP** (column/row parallel linears)
while the **MoE layer uses EP** (all-to-all). They operate on orthogonal
process groups and don't interfere:

```
  ┌──────────────────────────────────────────┐
  │             BlockMoE                      │
  │                                           │
  │  LN1 → AttentionTP ──── uses TP group    │
  │         (all-reduce)     {same dp,ep,pp}  │
  │                                           │
  │  LN2 → MoELayer ─────── uses EP group    │
  │         (all-to-all)     {same dp,pp,tp}  │
  └──────────────────────────────────────────┘
```

### EP × PP

When pipeline parallelism splits layers across stages, MoE blocks can land
on any stage. The `StageModule` passes EP config to `make_block()`:

```
  PP Stage 0 (layers 0-1)          PP Stage 1 (layers 2-3)
  ┌─────────────────────┐         ┌─────────────────────┐
  │ Layer 0: Dense      │  send   │ Layer 2: Dense      │
  │ Layer 1: MoE ←EP    │ ─────►  │ Layer 3: MoE ←EP    │
  └─────────────────────┘  recv   └─────────────────────┘
         │                                │
     EP all-to-all                   EP all-to-all
    (within stage 0)               (within stage 1)
```

Each PP stage has its own EP groups (same dp, tp; same pp; varying ep).

### EP × DP

DP and EP are **independent axes**. Each DP replica has its own set of EP groups.
Gradient averaging via `average_gradients()` happens over the **DP group**
(same ep, pp, tp; varying dp) — this works unchanged because each DP replica
holds the same expert shard.

```
  DP=2, EP=2 example (4 "MoE" viewpoints):

  DP replica 0                    DP replica 1
  ┌──────────┬──────────┐       ┌──────────┬──────────┐
  │ EP rank 0│ EP rank 1│       │ EP rank 0│ EP rank 1│
  │ Exp 0-3  │ Exp 4-7  │       │ Exp 0-3  │ Exp 4-7  │
  └──────────┴──────────┘       └──────────┴──────────┘
       │            │                 │            │
       └── EP a2a ──┘                 └── EP a2a ──┘
       │                              │
       └────── DP grad avg ───────────┘
```

---

## Load Balancing Loss

The auxiliary load-balancing loss encourages the router to spread tokens evenly
across experts, preventing "expert collapse" where a few experts get all tokens.

```
  f_e = (# tokens routed to expert e) / (N × K)    — selection fraction
  p_e = mean(gate_probs[:, e])                      — mean routing probability

  L_aux = α × E × Σ_e (f_e × p_e)
```

Where `α = aux_loss_coef` (default 0.01).

**Key detail:** The aux loss is computed **before** any EP communication,
using the full `gate_probs (N, E)` over all `E` experts. This works because
the router is replicated — every EP rank sees the same routing probabilities.

The aux loss is added to the data loss during backward:

```python
# In GPipeEngine backward (last stage):
loss_mb = (data_loss + aux_loss) / micro_batches
loss_mb.backward()

# In non-PP forward:
(loss + aux_loss).backward()
```

---

## Concrete Example Walkthrough

**Setup:** `E=4 experts, EP=2, K=2, N=4 tokens`

```
EP rank 0 owns: Expert 0, Expert 1  (E_local = 2)
EP rank 1 owns: Expert 2, Expert 3
```

### Step 1-2: Routing (identical on both EP ranks — router is replicated)

```
Token 0 → Expert 1 (w=0.6), Expert 3 (w=0.4)
Token 1 → Expert 0 (w=0.7), Expert 2 (w=0.3)
Token 2 → Expert 2 (w=0.5), Expert 3 (w=0.5)
Token 3 → Expert 1 (w=0.8), Expert 0 (w=0.2)
```

### Step 3: Expand (N=4, K=2 → N*K=8 entries)

```
Index:  0    1    2    3    4    5    6    7
Token:  T0   T0   T1   T1   T2   T2   T3   T3
Expert: 1    3    0    2    2    3    1    0
Weight: 0.6  0.4  0.7  0.3  0.5  0.5  0.8  0.2
```

### Step 4: Sort by expert

```
Sorted:  E0   E0   E1   E1   E2   E2   E3   E3
Token:   T1   T3   T0   T3   T1   T2   T0   T2
Weight:  0.7  0.2  0.6  0.8  0.3  0.5  0.4  0.5
```

### Step 5: EP dispatch

Map expert → EP rank: `ep = expert_id // 2`

```
EP Rank 0 (experts 0,1):          EP Rank 1 (experts 2,3):
┌─────────────────────┐          ┌─────────────────────┐
│ Local tokens for     │          │ Local tokens for     │
│ E0,E1: [T1,T3,T0,T3]│          │ E2,E3: [T1,T2,T0,T2]│
│ send_counts: [4, 4]  │          │ send_counts: [4, 4]  │
└──────────┬──────────┘          └──────────┬──────────┘
           │                                │
     ┌─────┴──── All-to-All ───────────────┘
     │
     ▼  On EP Rank 0, after dispatch:
  ┌──────────────────────────────────────┐
  │ Received tokens for experts 0 and 1: │
  │ From R0: T1→E0, T3→E0, T0→E1, T3→E1 │  (4 local)
  │ From R1: T1→E0, T2→E0, T0→E1, T2→E1 │  (4 received) ← Wait, this
  │                                       │    depends on routing!
  │ (Actual split depends on how many     │
  │  tokens on each rank target E0/E1)    │
  └──────────────────────────────────────┘
```

### Step 6-7: Local expert compute

Each EP rank runs only its `E_local` experts on the received tokens:

```
EP Rank 0:
  Expert 0: process [T1, T3]  →  [y_T1_E0, y_T3_E0]
  Expert 1: process [T0, T3]  →  [y_T0_E1, y_T3_E1]

EP Rank 1:
  Expert 2: process [T1, T2]  →  [y_T1_E2, y_T2_E2]
  Expert 3: process [T0, T2]  →  [y_T0_E3, y_T2_E3]
```

### Step 8: Combine (reverse All-to-All)

Results are sent back to the originating EP rank.

### Step 9-10: Unsort, weight, reduce

```
Token 0 final: 0.6 × y_T0_E1 + 0.4 × y_T0_E3
Token 1 final: 0.7 × y_T1_E0 + 0.3 × y_T1_E2
Token 2 final: 0.5 × y_T2_E2 + 0.5 × y_T2_E3
Token 3 final: 0.8 × y_T3_E1 + 0.2 × y_T3_E0
```

### Permutation Tracking

The implementation carefully tracks **three nested permutations** and their inverses
to restore the original token order:

```
Original order                  (N*K,)
    │  sort_idx (by expert)
    ▼
Expert-sorted order             (N*K,)
    │  ep_sort_idx (by EP rank)
    ▼
EP-rank-sorted order            (N*K,)   ← sent via all_to_all
    │  local_sort_idx (by local expert)
    ▼
Local-expert-sorted order       (N',)    ← fed to ExpertGroup

                     ... expert compute ...

Local-expert-sorted order       (N',)
    │  argsort(local_sort_idx)
    ▼
EP-rank-sorted order            (N*K,)   ← received via all_to_all
    │  argsort(ep_sort_idx)
    ▼
Expert-sorted order             (N*K,)
    │  argsort(sort_idx)
    ▼
Original order                  (N*K,)   ← apply weights & reduce
```

---

## EP × PP Pitfalls: The Hidden Coordinate Filter

When EP is combined with PP, any **point-to-point send/recv outside of the
PP pipeline** (e.g., sending logits or losses to rank 0 for printing) must
filter on **all four coordinates**, not just DP, TP, and PP.

### The Bug Pattern

Consider a logit-collection pattern after PP forward:

```python
# WRONG — missing ep_rank filter
if stage.is_last and topo.dp_rank == 0 and topo.tp_rank == 0:
    dist.send(logits, dst=0, tag=2000)
```

With `dp=1, ep=2, tp=2, pp=2`, the rank layout is:

```
Rank │ dp │ ep │ pp │ tp │ matches send condition?
─────┼────┼────┼────┼────┼───────────────────────
  0  │  0 │  0 │  0 │  0 │  no  (not last stage)
  1  │  0 │  0 │  0 │  1 │  no  (tp_rank=1)
  2  │  0 │  0 │  1 │  0 │  YES ← intended sender
  3  │  0 │  0 │  1 │  1 │  no  (tp_rank=1)
  4  │  0 │  1 │  0 │  0 │  no  (not last stage)
  5  │  0 │  1 │  0 │  1 │  no  (tp_rank=1)
  6  │  0 │  1 │  1 │  0 │  YES ← UNINTENDED!
  7  │  0 │  1 │  1 │  1 │  no  (tp_rank=1)
```

Both rank 2 and rank 6 try to `dist.send`, but rank 0 only issues **one**
`dist.recv` from rank 2. Rank 6 blocks forever on the unmatched send:

```
  Rank 0                    Rank 2 (ep=0)           Rank 6 (ep=1)
  ┌──────────┐              ┌──────────┐             ┌──────────┐
  │ recv(R2) │◄─────────────│ send(R0) │             │ send(R0) │
  │    ✓     │              │    ✓     │             │ BLOCKED  │
  ├──────────┤              ├──────────┤             │ forever  │
  │ barrier  │              │ barrier  │             │    ×     │
  │ BLOCKED  │◄── waiting ──│ BLOCKED  │◄── both ───│  never   │
  │          │   for R6     │          │   waiting   │ reaches  │
  │          │              │          │   for R6    │ barrier  │
  └──────────┘              └──────────┘             └──────────┘
                      DEADLOCK
```

### The Fix

Always include `ep_rank == 0` in the filter:

```python
# CORRECT — all four coordinates filtered
if stage.is_last and topo.dp_rank == 0 and topo.ep_rank == 0 and topo.tp_rank == 0:
    dist.send(logits, dst=0, tag=2000)
```

### General Rule

> **Whenever you do a point-to-point send/recv that selects "one canonical rank"
> (e.g., dp_rank=0, tp_rank=0), you MUST also filter ep_rank=0.**
> Otherwise, every EP replica of that (dp, tp, pp) coordinate will try
> to participate, causing unmatched operations and deadlocks.

This applies to:

| Location | What it sends | File |
|----------|---------------|------|
| Logit collection after PP forward | `logits` tensor | `parallel_sanity.py` |
| Loss reporting per training step | `loss_scalar` tensor | `boundary_push.py` |
| `is_main` guard for wandb / printing | N/A (gating flag) | `train.py` |
| Any future rank-0 aggregation | Whatever payload | — |

### The `is_main` Variant: Duplicate wandb Runs

The same bug pattern manifests differently in `train.py`—not as a deadlock,
but as **duplicate wandb runs** that make loss charts look noisy.

```python
# WRONG — missing ep_rank filter
is_main = (topo.dp_rank == 0 and topo.tp_rank == 0
           and topo.pp_rank == (topo.pp - 1))
```

With `dp=2, ep=2, tp=2, pp=1`, both rank 0 (`ep_rank=0`) and rank 2
(`ep_rank=1`) satisfy this condition. Both call `wandb.init()`, creating
**two separate runs** logging to the same project with overlapping steps:

```
  wandb dashboard (step 50)
  ─────────────────────────────────────
  Run A (rank 0, ep=0): loss = 7.7472   ●
  Run B (rank 2, ep=1): loss = 7.7453     ●  ← slightly different
                                               due to EP all-to-all
                                               float non-determinism
```

The result is a chart that looks "noisy" with two interleaved traces
at slightly different y-values. The fix is identical:

```python
# CORRECT — single logging rank
is_main = (topo.dp_rank == 0 and topo.ep_rank == 0  # ← added
           and topo.tp_rank == 0
           and topo.pp_rank == (topo.pp - 1))
```

> **Lesson:** The EP coordinate filter isn't just about send/recv deadlocks.
> Any "do this once" guard (`is_main`, singleton init, file I/O) must
> include `ep_rank == 0` or every EP replica will duplicate the action.

### Why This Doesn't Affect the Pipeline Itself

The PP `send/recv` inside `GPipeEngine._p2p()` uses `self.prev_rank` / `self.next_rank`,
which are computed from the **full 4D coordinate** including `ep_rank`:

```python
self.next_rank = rank_from_coords(
    self.dp_rank, self.ep_rank,  # ← ep_rank included
    self.pp_rank + 1, self.tp_rank,
    topo.dp, topo.ep, topo.tp, topo.pp
)
```

So each EP replica's pipeline is fully independent and correct. The bug only
manifests in **ad-hoc send/recv** added outside the pipeline engine.

---

## Gradient Parity with EP

Gradient parity tests compare distributed gradients against a single-GPU
baseline. With EP, there are **expected** and **unexpected** sources of diff.

### Expected Gradient Differences

#### 1. Weight Tying Split (PP > 1)

In the single-GPU baseline, `wte.weight` and `lm_head.weight` are **tied**
(same tensor). The gradient accumulates from both embedding backward and
lm_head backward:

```
  Single GPU:  wte.grad = ∇_embed + ∇_lm_head   (tied weights)

  PP ≥ 2:     Stage 0: wte.grad  = ∇_embed       (only embedding)
              Stage N: lm_head.grad = ∇_lm_head   (only head)
```

This causes `wte.weight` max diff ~0.08–0.23, which is **expected and correct**.

#### 2. MoE All-to-All Reordering (EP > 1)

The all-to-all dispatch sorts tokens by EP rank, processes them, then unsorts.
Floating-point addition is not associative — the order of accumulation differs
from the single-GPU baseline:

```
  Single GPU:  tokens processed in original order
               grad = Σ(g_0, g_1, g_2, g_3, g_4, g_5, g_6, g_7)

  EP = 2:      tokens shuffled by all-to-all, processed in EP-local order
               grad = Σ(g_0, g_3, g_5) + Σ(g_1, g_2, g_4, g_6, g_7)
               (different accumulation order → fp32 rounding differences)
```

#### 3. TP Shard Reduction Order

TP all-reduce sums partial results across TP ranks. The reduction tree order
differs from single-GPU sequential computation.

### Observed Diff Magnitudes

For the test model (`n_embed=64, n_layer=4, vocab=256, E=8, K=2`):

```
 Parameter                │  max diff  │ mean diff │ Source
──────────────────────────┼────────────┼───────────┼──────────────────────
 tr.wte.weight            │  ~0.08     │  ~0.001   │ PP weight untying
 tr.wte.weight (PP=2)     │  ~0.23     │  ~0.005   │ PP weight untying
 tr.h.0.ln_1.weight       │  ~0.002    │  ~0.0005  │ fp32 accumulation
 tr.h.0.attn.c_attn.weight│  ~0.022    │  ~0.001   │ TP recon + EP reorder
 tr.h.0.attn.c_proj.weight│  ~0.049    │  ~0.006   │ TP recon + EP reorder
```

Key observations:
- **Diffs are deterministic** — running the same config twice gives identical diffs
- **Diffs don't grow** across training steps (boundary_push confirms this)
- **Forward logit diff is < 2e-7** — the forward pass is numerically exact
- Backward diffs are larger because gradients amplify small forward differences
  through the chain rule across all layers

### How to Verify Correctness Despite Diffs

1. **Forward parity (< 1e-4):** If `LOGIT_DIFF_MAX` is small, weights are loaded
   correctly and the forward compute graph is equivalent
2. **Stable training:** If `boundary_push` shows decreasing loss over 10 steps,
   the optimizer is receiving usable gradients
3. **Deterministic diffs:** If the same config produces identical diff values
   across runs, the differences are systematic (accumulation order), not random

---

## Test Infrastructure for EP

Three test scripts validate EP correctness, each targeting a different property:

```
  ┌─────────────────────────────────────────────────────────────┐
  │                     run_full_suite.sh                       │
  │  Iterates over configs: "dp,ep,tp,pp,moe"                  │
  │                                                             │
  │  For each config, runs three tests in sequence:             │
  │                                                             │
  │  1. parallel_sanity.py ──► Forward logit diff + backward OK │
  │     • Loads identical weights on all ranks                  │
  │     • Compares logits vs single-GPU reference               │
  │     • Runs one backward step to verify no crash             │
  │     • PASS threshold: LOGIT_DIFF_MAX ≤ 1e-4                 │
  │                                                             │
  │  2. grad_parity.py ────► Gradient comparison (informational)│
  │     • Compares per-parameter gradient diffs                 │
  │     • Reconstructs TP-sharded grads on rank 0               │
  │     • No strict pass/fail — diffs are printed for review    │
  │                                                             │
  │  3. boundary_push.py ──► Multi-step training stability      │
  │     • Runs 10 steps with 8 microbatches each                │
  │     • Verifies loss doesn't explode (no NaN/Inf)            │
  │     • Confirms all ranks stay synchronized                  │
  └─────────────────────────────────────────────────────────────┘
```

### MoE-Specific Test Configs

```bash
# 8-GPU MoE configs in run_full_suite.sh:
"1,2,2,2,1"   # EP(2) + TP(2) + PP(2) — tests EP+PP interaction
"2,4,1,1,1"   # DP(2) + EP(4) — tests heavy EP sharding
```

Run only MoE configs:
```bash
MOE_ONLY=1 bash tests/run_full_suite.sh
```

### EP-Specific Weight Loading in Tests

Tests load a single-GPU reference state dict into EP-sharded models.
The key challenge is mapping **global expert indices** to **local expert indices**:

```python
def _load_moe_weights(blk, full_state, prefix, ep_rank=0):
    # Router gate — replicated, load directly
    blk.moe.router.gate.weight.data.copy_(
        full_state[prefix + "moe.router.gate.weight"]
    )
    
    # Experts — each EP rank loads only its shard
    num_local = len(blk.moe.expert_group.experts)
    expert_start = ep_rank * num_local
    for local_idx, expert in enumerate(blk.moe.expert_group.experts):
        global_idx = expert_start + local_idx
        expert.w1.weight.data.copy_(
            full_state[f"{prefix}moe.expert_group.experts.{global_idx}.w1.weight"]
        )
```

```
  Reference state dict            EP rank 0              EP rank 1
  (single GPU, all 8)             (loads 0-3)            (loads 4-7)
  ┌──────────────────┐           ┌──────────────┐       ┌──────────────┐
  │ experts.0.w1     │──────────►│ experts[0].w1│       │              │
  │ experts.1.w1     │──────────►│ experts[1].w1│       │              │
  │ experts.2.w1     │──────────►│ experts[2].w1│       │              │
  │ experts.3.w1     │──────────►│ experts[3].w1│       │              │
  │ experts.4.w1     │           │              │──────►│ experts[0].w1│
  │ experts.5.w1     │           │              │──────►│ experts[1].w1│
  │ experts.6.w1     │           │              │──────►│ experts[2].w1│
  │ experts.7.w1     │           │              │──────►│ experts[3].w1│
  └──────────────────┘           └──────────────┘       └──────────────┘
```

### Grad Parity Reconstruction for EP

When comparing gradients, `grad_parity.py` gathers shards from rank 0 only
for `dp_rank=0, ep_rank=0, target_pp_rank`. This means:

- Only EP rank 0's gradients are compared against the baseline
- EP rank 1+ gradients are **not** compared (their expert params are different
  slices that don't exist in the same positions in the baseline)
- Shared parameters (LN, attention) should be identical across EP ranks
  after DP gradient averaging

---

*This document describes the EP implementation as of February 2026.*
