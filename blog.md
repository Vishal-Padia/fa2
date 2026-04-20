# Flash Attention 2 in CuteDSL: A Naive Kernel and What the Profiler Thinks of It

After writing [Flash Attention 1](https://www.lowlevelml.com/blog/flash-attention-1) in plain CUDA, I wanted to take the next step and port Flash Attention 2 to **CuteDSL**, the Python front-end NVIDIA ships with CUTLASS 4. CuteDSL is new, under-documented, and surprisingly fun once you wrap your head around it. The catch is that almost no end-to-end FA2 write-ups exist for it, so a lot of this post is me learning the abstractions on the page, hopefully so you don't have to learn them twice.

What we'll do in this post:

1. Recap what FA2 actually does, and how it differs from FA1.
2. Write the algorithm in PyTorch so the math is unambiguous.
3. Translate it to a naive CuteDSL kernel, explaining CuteDSL's `TiledCopy`, `TiledMMA`, and partition abstractions as we go.
4. Profile it with Nsight Compute.
5. Read the profile honestly and pick what to optimize next.

I haven't done the optimization work yet. Once I do, I'll come back and extend this post with the numbers and the changes that got us there. For now, think of this as "the baseline is in, here's what's wrong with it."

> 💡 This post assumes you're comfortable with CUDA matmul, shared-memory tiling, and Tensor Cores. If "mma.m16n8k16", "warp", or "bank conflict" mean nothing to you, the [FA1 post](https://www.lowlevelml.com/blog/flash-attention-1) is a gentler on-ramp.

A running table for the optimizations, to be filled in as they land:

| Version         | 1024×1024, B=2, N=8, H=128 | vs SDPA          | Key change                  |
|-----------------|----------------------------|------------------|-----------------------------|
| Naive CuteDSL   | **0.417 ms**               | **2.02× slower** | baseline, this post         |
| + Swizzle       | ???                        | ???              | kill bank conflicts         |
| + Pipeline      | ???                        | ???              | overlap cp.async with MMA   |
| + Split-K warps | ???                        | ???              | fix occupancy               |

Benchmarked on an A10G (sm_86), fp16 inputs, fp32 accumulator, PyTorch SDPA as the reference.

---

## 1. What is Flash Attention 2?

Flash Attention is an algorithmic trick that computes exact attention in a single fused kernel, streaming over `KV` tiles while keeping everything resident in on-chip memory. The original FA1 paper (Tri Dao, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)) made attention IO-bound rather than FLOP-bound, which is why it was 2 to 4x faster than a naive `softmax(QK^T)V`. FA2 ([Dao 2023, arXiv:2307.08691](https://arxiv.org/abs/2307.08691)) keeps the same math and just rearranges the work.

If you're new to the family, the one-paragraph summary is: instead of materializing the full `S = QK^T` matrix (which is quadratic in sequence length and blows up HBM), you iterate over blocks of `K` and `V`, maintain an **online softmax** (a running max and normalizer), and accumulate into the output tile as you go. Never see the full attention matrix, never write it to HBM, done.

## 2. What FA2 Improved Over FA1

Three things, in decreasing order of how much they matter for our kernel:

1. **Swap the loop order.** FA1 has the outer loop over `KV` tiles and the inner loop over `Q` tiles. FA2 flips this: outer loop is `Q`, inner loop is `KV`. That means each threadblock owns one `Q` tile and streams all of `KV` through it. The `Q` tile is loaded into shared memory *once*, and the output accumulator `O` stays in registers for the whole inner loop. No shared-memory reduction across threadblocks.

2. **Defer the softmax normalization.** FA1 rescaled the output every inner iteration (`O_i *= exp(m_old - m_new)`) and divided by the running sum `l_i`. FA2 still does the rescale on max-updates, but defers the **final division by `l`** to the very end. One division per row per kernel call, not per iteration.

3. **Better warp partitioning.** FA1 split work across warps along the `K` dimension, which forced a reduction through shared memory. FA2 splits along the `Q` dimension, so each warp owns an independent slice of rows with no cross-warp communication during the inner loop.

For this post we care mostly about (1) and (2). (3) lives inside the `TiledMMA` configuration and we'll set it up but not really explain it in depth until we come back to this for the optimization pass.

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

- `m` is the running row max. Every time we see a larger score, we need to rescale everything we've accumulated so far by `exp(m_old - m_new)`. This is the only reason the loop needs any state between iterations. Without the max-tracking, we could reorder the iterations freely.
- `l` is the running denominator. It accumulates `sum(exp(S - m))` but with the same rescaling as `O_tile`, so it stays consistent.
- The final `O / l` is applied **once**, outside the inner loop. In FA1 it happened inside. In the CuteDSL kernel this turns into a per-row divide right before the output store.

> 💡 If you want to gut-check the correctness of the rescaling trick: at any point in the loop, `O_tile_i / l_i == softmax(QK^T[:, :kv_end]) @ V[:kv_end]`. The online update keeps this invariant as new `KV` tiles arrive, which is why the final result is exact (not approximate).

Run this against `F.scaled_dot_product_attention` and it matches to `atol=1e-5`. Now we replicate it on a GPU.

---

## 4. The Naive CuteDSL Kernel

This is the section I wish existed when I started. CuteDSL's docs assume you already grok [CuTe's C++ layout algebra](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md), and most online examples are either trivial (elementwise copies) or so heavily templated that it's hard to see the forest. So I'll introduce each abstraction the first time we reach for it.

The full file is [`fa2_naive_cutedsl.py`](./fa2_naive_cutedsl.py). I'll walk through it in the order the kernel executes, not the order it appears in the file.

### 4.1 The tiling plan

One threadblock per `(batch, head, q_tile)`. Each block:

1. Loads its `Q` tile from HBM into shared memory (once, stays there).
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

Grid is `(ceil(Sq / BLOCK_Q), batch, num_heads)`. For a `B=2, N=8, Sq=1024` run that's `(16, 2, 8) = 256` blocks, plenty to fill an A10G's 80 SMs (3.2 waves). Good, we'll come back to this in the profile.

> 💡 The mental picture: one threadblock is a horizontal band of `BLOCK_Q` rows. It sits still. The `KV` tiles slide left-to-right through it, and the block accumulates `O` in registers the whole time. That picture is the entire FA2 outer-loop schedule.

### 4.2 Shared-memory layouts

First new CuteDSL concept: a **Layout** is a pair `(shape, stride)` that maps a logical coordinate to an offset in a linear buffer. For our `Q` tile in shared memory:

```python
sQ_layout = cute.make_layout(
    (BLOCK_Q, HEAD_DIM),
    stride=(HEAD_DIM, 1),  # row-major
)
```

That's all, no swizzling yet. Just plain row-major. The upside is it's trivial to reason about. The downside is that several threads in a warp will hit the same shared-memory bank on `ldmatrix`, which we'll see in the profile.

The shared-memory **struct** is how CuteDSL lets you carve up dynamic shared memory into named regions:

```python
@cute.struct
class SharedStorage:
    sQ: cute.struct.MemRange[mQ.element_type, BLOCK_Q * HEAD_DIM]
    sK: cute.struct.MemRange[mQ.element_type, BLOCK_KV * HEAD_DIM]
    sV: cute.struct.MemRange[mQ.element_type, BLOCK_KV * HEAD_DIM]
```

For a naive kernel I gave `sQ`, `sK`, `sV` their own regions. A smarter version would alias `sV` with `sQ` (once we've used `Q` we don't touch it until the next block launches) to shave ~16KB per block and improve occupancy. That's on the list for the optimization pass.

### 4.3 `TiledCopy`: the global-to-shared copy

The first abstraction that really earns its keep is `TiledCopy`. It's CuteDSL's way of saying: *I have a tile of global memory, I have a tile of shared memory, and I have `N` threads. Figure out which thread copies which element, using which hardware instruction, vectorized however it wants.*

You build one in three steps. First, pick the hardware primitive. Here, Ampere's `cp.async` streaming 128-bit chunks directly from L2 to shared memory, bypassing the register file:

```python
cp_atom = cute.make_copy_atom(
    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
    mQ.element_type,
    num_bits_per_copy=128,  # 8 fp16 elements per thread per instruction
)
```

Second, describe how threads are laid out across the tile. With 128 threads and a `(64, 128)` tile where each thread moves 8 elements wide, you get a `(8, 16)` thread grid. That's 16 threads per row, 8 rows of threads, each row covering all 128 head-dim elements:

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

> 💡 The mental model: a `TiledCopy` is a compiled plan for moving a tile. You build it once outside the kernel, and inside the kernel you slice it per-thread using `get_slice(tidx)` to get exactly *this* thread's view.

### 4.4 `TiledMMA`: the Tensor Core plan

Same idea, for the MMA pipeline. Ampere's `mma.m16n8k16` takes a 16x16 `A` tile, an 8x16 `B` tile, and accumulates into a 16x8 fp32 `D`. One warp issues it, and each lane holds 4 fragments of `A`, 2 of `B`, and 4 of `D`.

`TiledMMA` lets us tile that up across our 4 warps:

```python
tiled_mma = cute.make_tiled_mma(
    warp.MmaF16BF16Op(mQ.element_type, cutlass.Float32, (16, 8, 16)),
    (NUM_THREADS // 32, 1, 1),  # 4 warps stacked along M
    permutation_mnk=(NUM_THREADS // 32 * 16, 16, 16),
)
```

Read this as: "4 warps, stacked along the `M` dimension, so each warp covers 16 rows, 64 rows total per issue, which is exactly `BLOCK_Q`." FA2's warp-split-along-`M` (point 3 in section 2) is literally this line.

### 4.5 Partitioning: turning tiles into per-thread fragments

Both `TiledCopy` and `TiledMMA` have `get_slice(tidx).partition_*` methods that take a tile and return *this thread's view* of it. There are three you see constantly:

- `partition_S(tile)`, source view, for loads
- `partition_D(tile)`, destination view, for stores
- `partition_A(tile)` / `partition_B(tile)` / `partition_C(tile)`, MMA operand views

For example, here's how we set up the global to shared copy of `Q`:

```python
gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
tQgQ = gmem_thr_copy.partition_S(gQ)  # this thread's source elements in gmem Q
tQsQ = gmem_thr_copy.partition_D(sQ)  # where they go in smem

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

    # scale + online softmax update (see below)
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

1. **Every copy is followed by a full `cp_async_wait_group(0)` and `sync_threads()`.** This means no overlap: the MMA waits for the copy, then the copy waits for the MMA. This is the single biggest reason the naive kernel is slow, there's no pipelining. When I come back to optimize, the first fix will be a double-buffered prologue so `K[kv+1]` starts arriving while we MMA on `K[kv]`.

2. **`V` has to be transposed for the second GEMM.** The MMA wants the `B` operand column-major, but we stored `V` row-major. We cheat:

   ```python
   sVt = cute.make_tensor(
       sV.iterator,
       cute.make_layout((HEAD_DIM, BLOCK_KV), stride=(1, HEAD_DIM)),
   )
   ```

   Same underlying buffer, swapped shape and stride. No data movement, just a relabelling. The cost shows up later as bank conflicts on the `ldmatrix`, because the access pattern is now fighting the row-major smem layout. The proper fix is a swizzled layout. For now we eat the conflict.

### 4.7 The online softmax, per-thread

Inside the inner loop, after computing `S`, we do the FA2 update. This is the part that looked clean in PyTorch and turns into a nest of scalar loops here, because each thread owns only a **fragment** of `S`. A handful of rows and columns scattered across the 32x8 lane grid of the MMA.

```python
for row in cutlass.range_constexpr(cute.size(row_max)):
    mi = row % m_atom  # intra-atom row index
    mt = row // m_atom  # which M-tile

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

The butterfly shuffles (`shuffle_sync_bfly`, offsets 2 and 1) are the key detail. `mma.m16n8k16` lays out each output row across **4 lanes** (lanes `(0,1,2,3)`, `(4,5,6,7)`, and so on). To get the true row max, each lane needs to see the partial maxes from the other three. Two butterfly shuffles do exactly that. No shared memory, no `__syncthreads()`, just 2 warp-level instructions. This is the single cleanest example of why FA2's warp-split-along-`M` matters: a split along `K` would need a cross-warp reduction here, which is orders of magnitude more expensive.

After the KV loop, one more quad-reduce on `row_sum`, divide, and store. The epilogue is a tiny `ldmatrix`-style staging through shared memory before the final gmem write (we're on Ampere, so no TMA).

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

Max abs error of `2.4e-4` is normal for fp16 accumulated through softmax, and cosine sim of 1.0 to the eye means we match SDPA row-for-row. Good, the algorithm works. Now: how fast is it?

---

## 5. Profiling with Nsight Compute

I profiled with:

```bash
ncu --set full --target-processes all \
    --export fa2_naive_profile \
    .venv/bin/python fa2_naive_cutedsl.py
```

Raw timing first:

```
TIMING  (1024x1024, B=2, N=8, H=128)
  Mean:     0.417 ms  (trimmed)
  TFLOP/s:  20.61

PyTorch SDPA
  Mean:     0.206 ms  (trimmed)
  TFLOP/s:  41.64
  Ours vs SDPA: 2.02x slower
```

We're at half of SDPA. Not catastrophic for a v1, but not good. The interesting question is *why*. ncu's `--set full` output is huge, so I'll pull out the damning bits.

### 5.1 Speed-of-light summary

For the FA2 kernel on the small test (B=2, N=4, Sq=256 - 32 blocks, grid is tiny but the per-kernel metrics scale):

```
Section: GPU Speed Of Light Throughput
    Memory Throughput             26.05 %
    DRAM Throughput                6.97 %
    L1/TEX Cache Throughput       68.34 %
    L2 Cache Throughput            6.70 %
    Compute (SM) Throughput        9.93 %
```

Both compute and memory are well below peak. This almost always means **latency-bound**: the SMs have work queued but they're waiting on something. ncu even calls it out:

```
OPT   This workload exhibits low compute throughput and memory bandwidth
      utilization relative to the peak performance of this device.
      Achieved compute throughput and/or memory bandwidth below 60.0% of peak
      typically indicate latency issues. Look at Scheduler Statistics and
      Warp State Statistics for potential reasons.
```

### 5.2 Scheduler and warp state, the smoking gun

```
Section: Scheduler Statistics
    One or More Eligible                 10.96 %
    Issued Warp Per Scheduler             0.11
    No Eligible                          89.04 %
    Active Warps Per Scheduler            1.00 warp
    Eligible Warps Per Scheduler          0.11 warp
```

Every clock, each scheduler has on average **1 active warp and 0.11 eligible warps**. An A10G scheduler can pick from up to 12. We're starving it. 89% of cycles issue **nothing**, because every active warp is stalled waiting for a dependency. Almost certainly the `cp.async` we just issued, since we `cp_async_wait_group(0)` immediately.

### 5.3 Shared memory is a minefield

```
OPT   Est. Speedup: 55.81%
      The memory access pattern for shared loads might not be optimal and
      causes on average a 5.5 - way bank conflict across all 214,016 shared
      load requests. This results in 966,656 bank conflicts, which represent
      81.66% of the overall 1,183,744 wavefronts for shared loads.

OPT   Est. Speedup: 59.8%
      Shared stores: 8.0-way bank conflicts across 4,096 requests,
      28,672 bank conflicts (87.50% of wavefronts).
```

**81% of shared loads and 87% of shared stores are bank-conflicting.** Both GEMMs (`Q @ K^T` and `P @ V`) use `ldmatrix` to feed the Tensor Cores; `ldmatrix` wants its source in a swizzled layout, and we gave it plain row-major. Every warp's ldmatrix issue serializes across 5 to 8 bank-conflict cycles instead of completing in 1.

This is the single biggest leak. Ncu estimates ~56% speedup just from fixing load conflicts and ~60% from fixing store conflicts (these overlap, so the real number is lower, but 1.5x is a reasonable target).

### 5.4 Occupancy is pinned by shared memory

```
Section: Occupancy
    Block Limit Registers                 3 blocks / SM
    Block Limit Shared Mem                2 blocks / SM   <- binding
    Theoretical Active Warps per SM       8 warps
    Theoretical Occupancy                16.67 %
    Achieved Occupancy                    8.32 %

OPT   The 2.00 theoretical warps per scheduler this kernel can issue
      according to its occupancy are below the hardware maximum of 12.
      This kernel's theoretical occupancy (16.7%) is limited by the required
      amount of shared memory.
```

We're using **49 KB of shared memory per block** (`sQ + sK + sV` = `64*128*2 + 64*128*2 + 64*128*2` bytes = 48 KB, plus driver overhead). Each SM has 100 KB usable, so we fit 2 blocks per SM. That caps theoretical occupancy at 16.7%, and because warps stall so often we only achieve *half* of that, 8.32%.

The natural fix here is to be smarter about `Q`'s residency. `Q` gets loaded into shmem, then immediately `ldmatrix`'d to registers every iteration. If we `ldmatrix` it once at the top of the kernel and leave it in registers, shmem drops by 16 KB, occupancy improves, and we save one redundant load per iteration. Double-buffering `K` and `V` for pipelining eats some of that savings back, but the arithmetic still works out.

### 5.5 Instruction mix and stall counts

```
Section: Warp State Statistics
    Warp Cycles Per Issued Instruction       9.14 cycles
    Warp Cycles Per Executed Instruction     9.16 cycles
    Avg. Active Threads Per Warp             32
    Avg. Not Predicated Off Threads Per Warp 31.93
```

9 cycles per issued instruction is the average stall latency each time a warp actually issues. Compare that to the matmul-kernel stall of ~1 to 2 cycles you'd see in a well-tuned GEMM. The stalls live in three places: `cp.async` arrivals, `ldmatrix` bank conflicts, and register-pressure-induced reissues. All three will be addressed in the next round.

```
Registers Per Thread    168
```

168 registers is high, close to the 255 hard cap. Accumulators eat most of it: `acc_O` is `64x128` fp32 = 8192 elements = 256 regs per thread if divided across 128 threads. That's before `row_max`, `row_sum`, `acc_S`, and a fragment of `Q`. This is also why the register block limit says "3": we're lucky we're shared-memory limited first.

---

## 6. Where That Leaves Us

Rolling the profile into a prioritized list:

| Finding                                    | Est. impact  | Fix                                                   |
|--------------------------------------------|--------------|-------------------------------------------------------|
| 81 to 87% shared-mem bank conflicts        | ~1.5x        | Swizzled smem layouts for `Q`, `K`, `V`               |
| 89% "no eligible warp" stalls              | ~1.3 to 1.5x | Multi-stage `cp.async` pipelining (prologue + double buffer) |
| 16.7% theoretical, 8.3% achieved occupancy | ~1.2x        | Keep `Q` in registers, alias where possible           |
| 168 regs/thread                            | softer limit | Revisit after pipelining (may spill, may not)         |

Stacked, those roughly justify the 2.02x SDPA gap. SDPA on sm_86 uses FlashAttention-2 (PyTorch dispatches to it) with well-tuned swizzles and pipelining, so "match SDPA" is a realistic goal and "beat SDPA" is a stretch-but-plausible one.

And just to keep ourselves honest, the running table again:

| Version         | 1024x1024 | vs SDPA      | Key change                |
|-----------------|-----------|--------------|---------------------------|
| Naive CuteDSL   | 0.417 ms  | 2.02x slower | baseline                  |
| + Swizzle       | ???       | ???          | kill bank conflicts       |
| + Pipeline      | ???       | ???          | overlap cp.async with MMA |
| + Split-K warps | ???       | ???          | fix occupancy             |

---

## What's next

The optimizations are the fun part, and I'll fold them into this post as I finish them. In rough order: swizzled shared-memory layouts to kill the bank conflicts, then a multi-stage `cp.async` pipeline with double-buffered `K` and `V`, then register-held `Q` to claw back occupancy. Each of those has its own CuteDSL spell (`composition(Swizzle, layout)`, `cp_async_wait_group(N-1)` tricks, `make_fragment_like`), and each deserves a walk-through rather than a drive-by.

If you want to poke at the baseline yourself, everything is in this repo: [`fa2_using_pytorch.py`](./fa2_using_pytorch.py) for the reference algorithm and [`fa2_naive_cutedsl.py`](./fa2_naive_cutedsl.py) for the kernel in this post. Run with `.venv/bin/python fa2_naive_cutedsl.py` and you'll get the numbers above, give or take 10% depending on your GPU.

As always, happy to chat if anything here is unclear or wrong - reach out to me on X [@KyrieBlunders](https://x.com/KyrieBlunders).