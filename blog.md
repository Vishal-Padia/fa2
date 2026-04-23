# Flash Attention 2 in CuteDSL: A Naive Kernel, Three Optimizations, and Where I Got Stuck

After writing [Flash Attention 1](https://www.lowlevelml.com/blog/flash-attention-1) in plain CUDA, I wanted to take the next step and port Flash Attention 2 to **CuteDSL**, the Python front-end NVIDIA ships with CUTLASS 4. CuteDSL is new, under-documented, and surprisingly fun once you wrap your head around it. I do not write kernels every day, and I'm still building intuition for both GPU hardware and CuteDSL itself, so this project involved a lot of iteration, debugging, and help from Claude along the way.

Spoiler: I did not beat PyTorch SDPA. I got close, and along the way I learned a lot about where my mental model of GPU optimization was wrong. The failed optimizations are in here too, because those were the ones I actually learned from.

> 💡 **Thanks** to [Sriram](https://x.com/s_gowindone) for getting me into GPU work in the first place and patiently unblocking me whenever something made no sense, [GPU Mode](https://x.com/GPU_MODE) for the lectures and the community that makes kernels feel approachable, and [Gau Nernst](https://x.com/gaunernst), whose [FA on the 5090 post](https://gau-nernst.github.io/fa-5090/) is what gave me the idea to try this in the first place. Huge thanks to everyone.

What we'll do in this post:

1. Recap what FA2 actually does, and how it differs from FA1.
2. Write the algorithm in PyTorch so the math is unambiguous.
3. Translate it to a naive CuteDSL kernel, explaining CuteDSL's `TiledCopy`, `TiledMMA`, and partition abstractions as we go.
4. Profile it with Nsight Compute.
5. Walk through the three optimizations I tried: shared-memory pipelining (which didn't help), swizzling (which did), and `ldmatrix` (which did).
6. Look at the final gap vs SDPA and be honest about why it's still there.

> 💡 This post assumes you're comfortable with CUDA matmul, shared-memory tiling, and Tensor Cores. If "mma.m16n8k16", "warp", or "bank conflict" mean nothing to you, the [FA1 post](https://www.lowlevelml.com/blog/flash-attention-1) is a gentler on-ramp.

The running perf table:


| Version                  | 1024x1024, B=2, N=8, H=128 | TFLOP/s   | vs SDPA          | Key change                                                    |
| ------------------------ | -------------------------- | --------- | ---------------- | ------------------------------------------------------------- |
| Naive CuteDSL            | 0.417 ms                   | 20.61     | 2.02x slower     | baseline                                                      |
| + Pipelining + Swizzle   | 0.352 ms                   | 24.41     | 1.70x slower     | K double-buffer on top of a swizzled layout (V buffer broken) |
| + Swizzle (no pipeline)  | 0.308 ms                   | 27.93     | 1.48x slower     | swizzled smem layout alone, kills bank conflicts              |
| + LDMatrix (on swizzle)  | **0.268 ms**               | **32.01** | **1.30x slower** | actual `ldmatrix.sync` instead of generic smem copy           |
| PyTorch SDPA (reference) | 0.207 ms                   | 41.55     | 1.00x            |                                                               |


Benchmarked on an A10G (sm_86), fp16 inputs, fp32 accumulator.

---

## 1. What is Flash Attention 2?

Flash Attention is an algorithmic trick that computes exact attention in a single fused kernel, streaming over `KV` tiles while keeping everything resident in on-chip memory. The original FA1 paper (Tri Dao, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)) made attention IO-bound rather than FLOP-bound, which is why it was 2 to 4x faster than a naive `softmax(QK^T)V`. FA2 ([Dao 2023, arXiv:2307.08691](https://arxiv.org/abs/2307.08691)) keeps the same math and just rearranges the work.

If you're new to the family, the one-paragraph summary: instead of materializing the full `S = QK^T` matrix (which is quadratic in sequence length and blows up HBM), you iterate over blocks of `K` and `V`, maintain an **online softmax** (a running max and normalizer), and accumulate into the output tile as you go. Never see the full attention matrix, never write it to HBM, done.

## 2. What FA2 Improved Over FA1

Three things, in decreasing order of how much they matter for our kernel:

1. **Swap the loop order.** FA1 has the outer loop over `KV` tiles and the inner loop over `Q` tiles. FA2 flips this: outer loop is `Q`, inner loop is `KV`. That means each threadblock owns one `Q` tile and streams all of `KV` through it. The `Q` tile is loaded into shared memory *once*, and the output accumulator `O` stays in registers for the whole inner loop. No shared-memory reduction across threadblocks.
2. **Defer the softmax normalization.** FA1 rescaled the output every inner iteration and divided by the running sum `l_i`. FA2 still does the rescale on max-updates, but defers the **final division by `l`** to the very end. One division per row per kernel call, not per iteration.
3. **Better warp partitioning.** FA1 split work across warps along the `K` dimension, which forced a reduction through shared memory. FA2 splits along the `Q` dimension, so each warp owns an independent slice of rows with no cross-warp communication during the inner loop.

For this post we care mostly about (1) and (2). (3) lives inside the `TiledMMA` configuration and we'll set it up but not really talk about it.

---

## 3. The Algorithm, in PyTorch

Before writing a single line of CuteDSL, it's worth making sure the algorithm is crystal clear. Here's the whole thing in ~60 lines of PyTorch. No fused kernels, no tricks, just nested loops. If this makes sense, the CuteDSL version is "the same thing, but with explicit thread-level tiling."

```python
def flash_attn_v2(Q, K, V, BLOCK_Q=64, BLOCK_KV=64):
    # Q, K, V: [B, L, D]
    B, L, D = Q.shape
    scale = 1.0 / (D**0.5)
    O = torch.zeros_like(Q)

    num_q_tiles = (L + BLOCK_Q - 1) // BLOCK_Q
    num_kv_tiles = (L + BLOCK_KV - 1) // BLOCK_KV

    for b in range(B):
        for q_tile_idx in range(num_q_tiles):
            q_start = q_tile_idx * BLOCK_Q
            q_end = min(q_start + BLOCK_Q, L)
            Q_tile = Q[b, q_start:q_end, :]  # [Bq, D]

            # running softmax state, one value per Q row
            m = torch.full((q_end - q_start,), -torch.inf, device=Q.device)
            l = torch.zeros(q_end - q_start, device=Q.device)
            O_tile = torch.zeros((q_end - q_start, D), device=Q.device)

            for kv_tile_idx in range(num_kv_tiles):
                kv_start = kv_tile_idx * BLOCK_KV
                kv_end = min(kv_start + BLOCK_KV, L)
                K_tile = K[b, kv_start:kv_end, :]  # [Bkv, D]
                V_tile = V[b, kv_start:kv_end, :]

                # 1. scores for this (Q, KV) pair
                S_tile = (Q_tile @ K_tile.T) * scale  # [Bq, Bkv]

                # 2. online softmax update
                m_new = torch.maximum(m, S_tile.max(dim=1).values)
                exp_old = torch.exp(m - m_new)  # rescale factor
                exp_scores = torch.exp(S_tile - m_new.unsqueeze(1))  # P

                # 3. update normalizer and accumulator
                l = l * exp_old + exp_scores.sum(dim=1)
                O_tile = O_tile * exp_old.unsqueeze(1) + exp_scores @ V_tile

                m = m_new

            # 4. final normalize, FA2 defers this to the end
            O[b, q_start:q_end, :] = O_tile / l.unsqueeze(1)
    return O
```

A few things worth calling out, because they map 1:1 onto what the CuteDSL kernel has to do per-thread:

- `m` is the running row max. Every time we see a larger score, we need to rescale everything we've accumulated so far by `exp(m_old - m_new)`. This is the only reason the loop needs any state between iterations.
- `l` is the running denominator. It accumulates `sum(exp(S - m))` but with the same rescaling as `O_tile`, so it stays consistent.
- The final `O / l` is applied **once**, outside the inner loop. In FA1 it happened inside. In the CuteDSL kernel this turns into a per-row divide right before the output store.

> 💡 Gut-check for the rescaling trick: at any point in the loop, `O_tile_i / l_i == softmax(QK^T[:, :kv_end]) @ V[:kv_end]`. The online update keeps this invariant as new `KV` tiles arrive, which is why the final result is exact, not approximate.

Run this against `F.scaled_dot_product_attention` and it matches to `atol=1e-5`. Now we replicate it on a GPU.

---

## 4. The Naive CuteDSL Kernel

This is the section I wish existed when I started. CuteDSL's docs assume you already grok [CuTe's C++ layout algebra](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md), and most online examples are either trivial elementwise copies or so heavily templated that it's hard to see the forest. So I'll introduce each abstraction the first time we reach for it.

### 4.1 The tiling plan

One threadblock per `(batch, head, q_tile)`. Each block:

1. Loads its `Q` tile from HBM into shared memory once.
2. Loops over all `KV` tiles. For each:
   - Async-copy `K` into shared memory.
   - Compute `S = Q @ K^T` into register-resident fp32 accumulators.
   - Scale by `1/√D`, update the online softmax, rescale `O_accum`.
   - Async-copy `V` into shared memory.
   - Compute `O += P @ V` where `P = softmax(S)` cast back to fp16.
3. Divide `O` by the running sum `l`, cast to fp16, store back to HBM.

Concretely:

```python
BLOCK_Q = 64
BLOCK_KV = 64
HEAD_DIM = 128
NUM_THREADS = 128  # 4 warps per block
```

Grid is `(ceil(Sq / BLOCK_Q), batch, num_heads)`. For a `B=2, N=8, Sq=1024` run that's `(16, 2, 8) = 256` blocks, plenty to fill an A10G's 80 SMs.

> 💡 Mental picture: one threadblock is a horizontal band of `BLOCK_Q` rows. It sits still. The `KV` tiles slide left-to-right through it, and the block accumulates `O` in registers the whole time. That picture is the entire FA2 outer-loop schedule.

### 4.2 Shared-memory layouts

First new CuteDSL concept: a **Layout** is a pair `(shape, stride)` that maps a logical coordinate to an offset in a linear buffer. For our `Q` tile in shared memory:

```python
sQ_layout = cute.make_layout(
    (BLOCK_Q, HEAD_DIM),
    stride=(HEAD_DIM, 1),  # row-major
)
```

That's all, no swizzling yet. Just plain row-major. The upside is it's trivial to reason about. The downside is that several threads in a warp will hit the same shared-memory bank, which we'll see in the profile.

The shared-memory **struct** is how CuteDSL lets you carve up dynamic shared memory into named regions:

```python
@cute.struct
class SharedStorage:
    sQ: cute.struct.MemRange[mQ.element_type, BLOCK_Q * HEAD_DIM]
    sK: cute.struct.MemRange[mQ.element_type, BLOCK_KV * HEAD_DIM]
    sV: cute.struct.MemRange[mQ.element_type, BLOCK_KV * HEAD_DIM]
```

For the naive kernel, `sQ`, `sK`, and `sV` get their own regions.

### 4.3 `TiledCopy`: the global-to-shared copy

The first abstraction that really earns its keep is `TiledCopy`. It's CuteDSL's way of saying: *I have a tile of global memory, I have a tile of shared memory, and I have `N` threads. Figure out which thread copies which element, using which hardware instruction, vectorized however it wants.*

Three steps. First, pick the hardware primitive. Here, Ampere's `cp.async` streaming 128-bit chunks directly from L2 to shared memory, bypassing the register file:

```python
cp_atom = cute.make_copy_atom(
    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
    mQ.element_type,
    num_bits_per_copy=128,  # 8 fp16 elements per thread per instruction
)
``` 

Second, describe how threads are laid out across the tile:

```python
elems_per_copy = 128 // mQ.element_type.width  # 8 for fp16
tQKV_layout = cute.make_layout(
    (NUM_THREADS // (HEAD_DIM // elems_per_copy), HEAD_DIM // elems_per_copy),
    stride=(HEAD_DIM // elems_per_copy, 1),
)
vQKV_layout = cute.make_layout((1, elems_per_copy))  # each thread: 1x8
```

Third, glue them together:

```python
gmem_tiled_copy = cute.make_tiled_copy_tv(cp_atom, tQKV_layout, vQKV_layout)
```

Now `gmem_tiled_copy` knows everything: which warp, which thread, which byte, what instruction. When we call `cute.copy(gmem_tiled_copy, src, dst)`, it expands into the right `cp.async` issues with no further work from us.

> 💡 Mental model: a `TiledCopy` is a compiled plan for moving a tile. You build it once outside the kernel, and inside the kernel you slice it per-thread using `get_slice(tidx)` to get exactly *this* thread's view.

### 4.4 `TiledMMA`: the Tensor Core plan

Same idea, for the MMA pipeline. Ampere's `mma.m16n8k16` takes a 16x16 `A` tile, an 8x16 `B` tile, and accumulates into a 16x8 fp32 `D`. One warp issues it, and each lane holds 4 fragments of `A`, 2 of `B`, and 4 of `D`.

`TiledMMA` tiles that up across our 4 warps:

```python
tiled_mma = cute.make_tiled_mma(
    warp.MmaF16BF16Op(mQ.element_type, cutlass.Float32, (16, 8, 16)),
    (NUM_THREADS // 32, 1, 1),  # 4 warps stacked along M
    permutation_mnk=(NUM_THREADS // 32 * 16, 16, 16),
)
```

Read this as: "4 warps, stacked along `M`, so each warp covers 16 rows, 64 rows total per issue, which is exactly `BLOCK_Q`." FA2's warp-split-along-`M` is literally this line.

### 4.5 Partitioning: turning tiles into per-thread fragments

Both `TiledCopy` and `TiledMMA` have `get_slice(tidx).partition_*` methods that take a tile and return *this thread's view* of it. Three you see constantly:

- `partition_S(tile)`, source view for loads
- `partition_D(tile)`, destination view for stores
- `partition_A(tile)` / `partition_B(tile)` / `partition_C(tile)`, MMA operand views

Here's how we set up the global to shared copy of `Q`:

```python
gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
tQgQ = gmem_thr_copy.partition_S(gQ)  # this thread's source in gmem Q
tQsQ = gmem_thr_copy.partition_D(sQ)  # where it goes in smem

cute.copy(gmem_tiled_copy, tQgQ, tQsQ)  # issue cp.async
cute.arch.cp_async_commit_group()
cute.arch.cp_async_wait_group(0)
cute.arch.sync_threads()
```

Four lines of Python and we've issued a full async HBM to SMEM copy with correct addressing, vectorization, and wait semantics. This is the part of CuteDSL that feels magical once it clicks.

### 4.6 The KV loop, where the algorithm lives

Strip away the scaffolding and the loop body is strikingly close to the PyTorch version:

```python
for kv in range(num_kv_tiles):
    # load K into smem
    cute.copy(gmem_tiled_copy, tKgK[None, None, None, kv], tKsK)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # S = Q @ K.T, first GEMM
    acc_S = cute.make_rmem_tensor(acc_S_shape, cutlass.Float32)
    acc_S.fill(0.0)
    for k in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
        cute.copy(smem_tiled_copy_Q, tSsQ[None, None, k], tsrQ_copy_view[None, None, k])
        cute.copy(smem_tiled_copy_K, tSsK[None, None, k], tSrK_copy_view[None, None, k])
        cute.gemm(tiled_mma, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)

    # scale + online softmax update
    ...

    # load V into smem
    cute.copy(gmem_tiled_copy, tVgV[None, None, None, kv], tVsV)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # O += P @ V, second GEMM
    for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
        cute.copy(smem_tiled_copy_V, tOsVt[None, None, k], tOrVt_copy_view[None, None, k])
        cute.gemm(tiled_mma, acc_O, tOrS[None, None, k], tOrVt[None, None, k], acc_O)
```

Two things are worth noting even at this naive stage:

1. **Every copy is followed by a full `cp_async_wait_group(0)` and `sync_threads()`.** No overlap, the MMA waits for the copy, then the copy waits for the MMA. This will be the target of the pipelining attempt.
2. **`V` has to be transposed for the second GEMM.** The MMA wants the `B` operand column-major, but we stored `V` row-major. We cheat:

    ```python
    sVt = cute.make_tensor(
        sV.iterator,
        cute.make_layout((HEAD_DIM, BLOCK_KV), stride=(1, HEAD_DIM)),
    )
    ```

    Same underlying buffer, swapped shape and stride. No data movement, just a relabelling.

### 4.7 The online softmax, per-thread

Inside the inner loop, after computing `S`, we do the FA2 update. This is the part that looked clean in PyTorch and turns into a nest of scalar loops here, because each thread owns only a **fragment** of `S`: a handful of rows and columns scattered across the 32x8 lane grid of the MMA.

```python
for row in cutlass.range_constexpr(cute.size(row_max)):
    mi = row % m_atom
    mt = row // m_atom

    # 1. row max over this thread's columns
    tile_max = -cutlass.Float32.inf
    for nt in cutlass.range_constexpr(n_tiles):
        for ni in cutlass.range_constexpr(n_atom):
            tile_max = max(tile_max, acc_S[(ni, mi), mt, nt])

    # 2. quad reduction, each MMA row is spread across 4 lanes
    tile_max = cute.arch.fmax(tile_max, cute.arch.shuffle_sync_bfly(tile_max, offset=2, ...))
    tile_max = cute.arch.fmax(tile_max, cute.arch.shuffle_sync_bfly(tile_max, offset=1, ...))

    new_rowmax = max(row_max[row], tile_max)
    rescale = cute.math.exp(row_max[row] - new_rowmax)

    # 3. rescale previously-accumulated O for this row
    for ot in cutlass.range_constexpr(o_n_tiles):
        for oi in cutlass.range_constexpr(o_n_atom):
            acc_O[(oi, mi), mt, ot] *= rescale
    row_sum[row] *= rescale

    # 4. exponentiate S, accumulate into row_sum
    for nt in cutlass.range_constexpr(n_tiles):
        for ni in cutlass.range_constexpr(n_atom):
            p = cute.math.exp(acc_S[(ni, mi), mt, nt] - new_rowmax)
            acc_S[(ni, mi), mt, nt] = p
            row_sum[row] += p

    row_max[row] = new_rowmax
```

The butterfly shuffles are the key detail. `mma.m16n8k16` lays out each output row across **4 lanes** (lanes `(0,1,2,3)`, `(4,5,6,7)`, and so on). To get the true row max, each lane needs to see the partial maxes from the other three. Two butterfly shuffles do exactly that. No shared memory, no sync, just 2 warp-level instructions. This is the single cleanest example of why FA2's warp-split-along-`M` matters: a split along `K` would need a cross-warp reduction here, which is orders of magnitude more expensive.

### 4.8 Correctness

Running against PyTorch SDPA:

```
CORRECTNESS CHECK  (B=2, N=8, Sq=1024, Sk=1024, H=128)
  Max  absolute error: 2.44e-04
  Mean absolute error: 7.58e-06
  Min  cosine sim:     1.000000
  Mean cosine sim:     1.000000
  VERDICT: PASS ✓
```

Max abs error of `2.4e-4` is normal for fp16 accumulated through softmax, and cosine sim of 1.0 means we match SDPA row-for-row. The algorithm works. Now: how fast is it?

---

## 5. Profiling the Naive Kernel

I profiled with:

```bash
ncu --set full --target-processes all \
    --export fa2_naive_profile \
    .venv/bin/python fa2_naive_cutedsl.py
```

Raw timing:

```
TIMING  (1024x1024, B=2, N=8, H=128)
  Mean:     0.417 ms  (trimmed)
  TFLOP/s:  20.61

PyTorch SDPA
  Mean:     0.206 ms  (trimmed)
  TFLOP/s:  41.64
  Ours vs SDPA: 2.02x slower
```

Half of SDPA. Not catastrophic for a v1, but not good. The interesting question is *why*.

### 5.1 Speed of light

```
Section: GPU Speed Of Light Throughput
    Memory Throughput             26.05 %
    DRAM Throughput                6.97 %
    L1/TEX Cache Throughput       68.34 %
    L2 Cache Throughput            6.70 %
    Compute (SM) Throughput        9.93 %
```

Both compute and memory are well below peak, which almost always means **latency-bound**. The SMs have work queued, they're waiting on something. ncu even calls it out:

```
OPT   This workload exhibits low compute throughput and memory bandwidth
      utilization relative to the peak performance of this device.
      Achieved compute throughput and/or memory bandwidth below 60.0% of peak
      typically indicate latency issues.
```

### 5.2 The smoking gun

```
Section: Scheduler Statistics
    One or More Eligible                 10.96 %
    Issued Warp Per Scheduler             0.11
    No Eligible                          89.04 %
```

**89% of cycles issue nothing.** The schedulers have 0.11 eligible warps per cycle on average. Every active warp is stalled waiting for a dependency.

### 5.3 Shared memory is a minefield

```
OPT   Est. Speedup: 55.81%
      The memory access pattern for shared loads causes on average a 5.5-way
      bank conflict across all 214,016 shared load requests. This results in
      966,656 bank conflicts, 81.66% of the overall 1,183,744 wavefronts.

OPT   Est. Speedup: 59.8%
      Shared stores: 8.0-way bank conflicts, 87.50% of wavefronts.
```

**81% of shared loads and 87% of shared stores are bank-conflicting.** Both GEMMs use generic smem copies that collide on banks because we gave them a plain row-major layout. This is the biggest single leak in the naive kernel.

### 5.4 Occupancy pinned by shared memory

```
Section: Occupancy
    Block Limit Shared Mem                2 blocks / SM   <- binding
    Theoretical Occupancy                16.67 %
    Achieved Occupancy                    8.32 %
```

48 KB of shared memory per block caps us at 2 blocks per SM, theoretical occupancy 16.7%. Because warps stall so often we only hit half of that.

So we have three candidate fixes to try:

1. **Pipeline the cp.async loads** so the MMA doesn't wait for data (targets the 89% "no eligible" stalls).
2. **Swizzle the shared-memory layout** so `ldmatrix` doesn't bank-conflict (targets the 81–87% conflicts).
3. **Use the real `ldmatrix` instruction** instead of the generic `CopyUniversalOp` (this is what wants the swizzled layout in the first place).

I tried them in that order. Two of the three helped. One did not. Here's how it went.

---

## 6. Attempt 1: Shared-Memory Pipelining (the one that didn't help)

The idea is textbook. Right now, the inner loop does:

```
load K  →  wait  →  MMA1  →  load V  →  wait  →  MMA2
```

Everything is sequential. If I double-buffer `K` and `V`, I can start the next iteration's loads while the current MMA is running:

```
load K[0]                    (prologue)
for kv in range(N):
    wait K[kv]
    start V[kv] load          // overlaps with MMA1
    MMA1 (using K[kv])
    wait V[kv]
    start K[kv+1] load        // overlaps with MMA2
    MMA2 (using V[kv])
```

I allocate two buffers for K, two for V, and use `kv % 2` to toggle between them. The code is in [fa2_shared_mem_pipelining_cutedsl.py](https://github.com/Vishal-Padia/fa2/blob/main/fa2_shared_mem_pipelining_cutedsl.py).

Running it:

```
TIMING  (1024x1024)
  Mean:     0.352 ms  (trimmed)
  TFLOP/s:  24.41
  Ours vs SDPA: 1.70x slower
```

0.352 ms vs the naive 0.417 ms. A 1.18x speedup, but that's entirely because this version also happens to use a swizzled smem layout (I was iterating on both at once). When I later isolated swizzling into its own file, swizzle *alone* went to 0.308 ms. So pipelining, on top of swizzling, was a **0.12 ms regression**. Not a win, a loss.

### Why it didn't work

A few things stacked:

1. **V double-buffering didn't survive contact.** I spent hours trying to get `sV0` / `sV1` to work and couldn't get correct results out of the kernel. At some point I stopped fighting it and shipped a version where only `K` is actually double-buffered; `V` reuses a single buffer. The committed code shows the vestige, there's a `tVsV0` and `tVsV1` in the partition setup but both point at `sV0`. So the `V` load still serializes with MMA2.
2. **Even with K-only pipelining, the `wait_group(0)` pattern kills the overlap.** The correct CuteDSL idiom for 2-stage pipelining is `cp_async_wait_group(N-1)` where `N` is the number of in-flight groups. You want to wait for *all but the most recent* group. `wait_group(0)` waits for everything, which is basically the same as the naive kernel.
3. **The stalls ncu was flagging might not even be `cp.async` stalls.** Looking at the swizzle-only profile (below), removing bank conflicts brought "no eligible" stalls from 89% down to 40%. A lot of what I was attributing to memory latency was actually bank-conflict serialization in `ldmatrix`. Pipelining can't fix that.

In hindsight I should have done swizzle first and pipelining second. Or third. Or not at all at this tile size; at `BLOCK_KV=64` the `cp.async` for K is only 16 KB, the whole MMA loop inside runs in a few hundred cycles, and there's not actually that much latency to hide. Pipelining pays for itself on larger tiles.

I'm leaving the broken-ish version in the repo because I think the *attempt* is more useful than a clean final. If someone wants to finish it properly, the change list is: get full `sV0`/`sV1` double-buffering working, switch every `cp_async_wait_group(0)` in the main loop to `cp_async_wait_group(1)`, and move the first `K[0]` load into the prologue before the loop.

---

## 7. Attempt 2: Swizzling (the one that worked)

`ldmatrix` wants its source laid out in shared memory in a specific permuted pattern so that the 32 lanes of a warp hit 32 distinct banks. A plain row-major layout doesn't do that; an `H=128` row lands on the same banks as the next row modulo 32, and you get the 5.5-way conflicts ncu was flagging.

CuTe handles this with a **swizzle function** that remaps addresses within a layout. The spell in CuteDSL is:

```python
sw = cute.make_swizzle(3, 3, 3)

sQ_layout_atom = cute.make_composed_layout(
    sw, 0,
    cute.make_layout((8, 64), stride=(64, 1)),
)
sQ_layout = cute.tile_to_shape(sQ_layout_atom, (BLOCK_Q, HEAD_DIM), (0, 1))
```

`Swizzle<3, 3, 3>` is the one to memorize for sm_80+ with 128-byte rows: it XORs 3 bits of the column index into 3 bits of the row index at a stride of 3 bits, which is exactly what you need to get conflict-free `ldmatrix` for 8x8 fp16 tiles. `make_composed_layout` then composes this permutation with a plain layout, and `tile_to_shape` stamps out the swizzled atom across the full `(BLOCK_Q, HEAD_DIM)` tile.

The beautiful part of CuteDSL is that **literally nothing else in the kernel changes.** The partitions, the copies, the MMA, the online softmax, all the same. The swizzle is entirely contained inside the layout objects; logical coordinates behave the same, only the physical address computation differs. The full file is [fa2_swizzle_cutedsl.py](https://github.com/Vishal-Padia/fa2/blob/main/fa2_swizzle_cutedsl.py) and the diff from naive is just the layout construction. Every kernel body line is identical.

### Results

```
TIMING  (1024x1024)
  Mean:     0.308 ms  (trimmed)
  TFLOP/s:  27.93
  Ours vs SDPA: 1.48x slower
```

**1.35x speedup.** And the profile explains exactly why:

```
Section: Memory Workload Analysis
    No warnings about bank conflicts.

Section: Scheduler Statistics
    One or More Eligible                 64.67 %   (was 10.96)
    Issued Warp Per Scheduler             0.65     (was 0.11)
    No Eligible                          35.33 %   (was 89.04)

Section: Occupancy
    Achieved Occupancy                   25.36 %   (was 8.32)
```

Bank conflicts: gone. "No eligible" stalls: cut by more than half. Occupancy: tripled, which I did not expect. My theory is that the bank-conflict serialization was holding warps resident longer than the hardware would have otherwise, and once it's gone the scheduler can actually rotate through warps as designed.

One tile shape detail worth flagging: the swizzle atom's `k_block_size` has to match the row stride the swizzle is permuting over. For `HEAD_DIM=128` and fp16, a 64-element inner dimension (128 bytes) works cleanly. The `8` in the atom shape is `BLOCK_Q / 8`, one row per 8x8 `ldmatrix` tile. Getting these numbers wrong silently gives you the wrong layout and bank conflicts come back.

---

## 8. Attempt 3: Actual `ldmatrix` (the one that got us closest)

Up to this point, the `smem -> register` copy that feeds the Tensor Cores has been using `cute.nvgpu.CopyUniversalOp()`, the "just load each element" generic copy. It works, but it doesn't issue the real `ldmatrix.sync.aligned.m8n8` instruction, which loads a full 8x8 tile into MMA operand layout in a single warp-level op. That's the instruction all the swizzle bit-twiddling was designed for.

The swap is tiny:

```python
# before
smem_copy_atom = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(), mQ.element_type
)

# after
smem_copy_atom_QK = cute.make_copy_atom(
    warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
    mQ.element_type,
)
smem_copy_atom_V = cute.make_copy_atom(
    warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
    mQ.element_type,
)
```

For `Q` and `K` we want the straight `ldmatrix`. For `V`, the second GEMM's B operand needs it transposed, and `ldmatrix.sync.aligned.m8n8.trans` does that transpose for free in one instruction (this is the right way to handle the `sVt` view from the naive kernel). The rest of the kernel body is again unchanged.

Full code: [fa2_ldmatrix_cutedsl.py](https://github.com/Vishal-Padia/fa2/blob/main/fa2_ldmatrix_cutedsl.py).

### Results

```
TIMING  (1024x1024)
  Mean:     0.268 ms  (trimmed)
  TFLOP/s:  32.01
  Ours vs SDPA: 1.30x slower
```

Another **1.15x on top of swizzle**, total **1.55x** vs the naive baseline. This is about what I expected: swizzle set up the smem layout so that a real `ldmatrix` could use it, and a real `ldmatrix` actually pays the dividend. Without the swizzle, `ldmatrix` would re-introduce bank conflicts. Without the `ldmatrix`, the swizzle is correct but under-exploited.

The profile looks almost identical to swizzle-only, which makes sense: memory and occupancy are already in a reasonable place, and the win here comes from fewer instructions issued per smem->reg load. TFLOP/s is what moved, 27.93 -> 32.01.

---

## 9. Where We Landed, and Why Still Behind SDPA

Final table:


| Version                 | 1024x1024 | TFLOP/s | vs SDPA      |
| ----------------------- | --------- | ------- | ------------ |
| Naive CuteDSL           | 0.417 ms  | 20.61   | 2.02x slower |
| + Pipelining + Swizzle  | 0.352 ms  | 24.41   | 1.70x slower |
| + Swizzle (no pipeline) | 0.308 ms  | 27.93   | 1.48x slower |
| + LDMatrix (on swizzle) | 0.268 ms  | 32.01   | 1.30x slower |
| PyTorch SDPA            | 0.207 ms  | 41.55   | 1.00x        |


We got from 2x slower to 1.30x slower. Not bad for someone who's still figuring things out, and frankly I was not going to beat SDPA. PyTorch's SDPA on sm_86 dispatches to Flash Attention 2 underneath, with code tuned by people who have forgotten more about CuTe than I currently know. The remaining 1.30x is, I think, a mix of:

- **My tile shapes aren't ideal.** BLOCK_Q=64, BLOCK_KV=64 is a safe choice, but larger tiles amortize the prologue and expose more parallelism per block. I'd want to sweep at least `{64, 128} x {64, 128}` and probably push BLOCK_KV up.
- **I never got pipelining to work.** That's real throughput left on the table, and it's what the "89% no eligible" -> "35%" improvement hinted at. Getting to 10% or lower requires actual overlap of cp.async with MMA.
- **168 registers per thread is high.** The accumulators eat most of it. A smarter layout (or WGMMA on Hopper, not available here) would use fewer.
- **Bare honesty: my CuteDSL fluency is still limited.** I can recognize what the profile is telling me, but I can't always translate "I need this thing to happen" into the right `make_tiled_copy_B` / `retile` / `partition_*` incantation on the first try. More reps required.

So: couldn't beat SDPA, got within 30%, learned a lot more from what didn't work than what did.

---

## What's Next

Things I want to try, roughly in order:

1. **Larger tiles.** `BLOCK_KV=128` with the same `BLOCK_Q=64` is probably the next easy win.
2. **Port to Hopper.** WGMMA + TMA would be a different kernel, not an incremental change, but the whole point of CuteDSL is that the abstractions should carry across architectures.

I'll extend this post as things land. Meanwhile, everything in here is in the repo: [fa2_using_pytorch.py](https://github.com/Vishal-Padia/fa2/blob/main/fa2_using_pytorch.py), [fa2_naive_cutedsl.py](https://github.com/Vishal-Padia/fa2/blob/main/fa2_naive_cutedsl.py), [fa2_shared_mem_pipelining_cutedsl.py](https://github.com/Vishal-Padia/fa2/blob/main/fa2_shared_mem_pipelining_cutedsl.py), [fa2_swizzle_cutedsl.py](https://github.com/Vishal-Padia/fa2/blob/main/fa2_swizzle_cutedsl.py), [fa2_ldmatrix_cutedsl.py](https://github.com/Vishal-Padia/fa2/blob/main/fa2_ldmatrix_cutedsl.py). Run with `.venv/bin/python <file>.py` and you'll get the numbers above, give or take 10% depending on your GPU.

As always, happy to chat if anything here is unclear or wrong. Just ping me on [Twitter](https://x.com/KyrieBlunders).