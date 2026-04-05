# fa2
Me trying to optimize Flash Attention 2 (implemented in cuteDSL) using various techniques

I am referencing this blog post by Gau Nerst: https://gau-nernst.github.io/fa-5090/

They wrote that in CUDA C++, but I am trying to implement it in Python using cuteDSL.

Things to understand before actually implementing it:
- Flash Attention 2
- The `mma.m16n8k16` Thread/Register Layout
- The C/D Accumulator (Output) Layout
- MMA layout 
- cp.async pipeline
- Implementing this in a naive way ie using Python/PyTorch primitives

### Flash Attention 2

#### What is FA1?

"Don't materialize the full attention matrix"

Instead:
- Compute attention in chunks (tiles)
- Keep everything inside fast GPU Sram
- Use a trick tp compute softmax incrementally

Limitations in FA1:
1. Low GPU utilization
- Not all threads are busy
- Some cores sit idle
2. Poor parallelism
- Work is not evenly distributed
3. Not optimized for modern GPUs
- Doesn't fully use Tensor Cores
4. Backward pass inefficiencies
- Still somewhat slow

#### What changes in FA2?

FA2 is not small tweak, it's complete rethinking of how work is distributed on GPU.

Big Idea #1: Better Parallelism

FA1:
    - Parallelized over sequence dimension
FA2:
    - Parallelizes over:
        - sequence
        - heads
        - batch
        - And splits work more finely

This results in much better GPU utilization. More threads active which in turns means faster execution.

Big IDea #2: Work Partitioning (This is the CORE)

FA1:
    - One block handles a large chunk -> inefficient
FA2: 
    - Splits computation into smaller tiles
    - Multiple blocks cooperate

This removes bottlenecks where one block becomes slow

Big Idea #3: Less Shared Memory Dependency

FA1:
    - Heavy use of Shared Memory -> can limit occupancy
FA2:
    - Reduces shared memory usage
    - Uses register more efficiently

This results in more parallel threads, better scaling

Big Idea #4: Better Warp Level Optimization

GPU executes in warps (32 threads)
 
FA2:
    - Aligns computation with warp execution
    - Reduces idle threads inside warps

Less wasted compute.

Big Idea #5: Faster Backward Pass

Backward pass is often ignored but it's huge for training.

FA2:
    - Redesigns backward computation
    - Avoids recomputation inefficiencies

This results in significantly faster training speeds.

Big Idea #6: Tensor Core Friendly

Modern GPUs (A100, H100) love:
- Matrix Multiplications in specifci shapes

FA2:
    - Aligns operations to Tensor Core-Friendly sizes

This is a big deal for real-world speed.

Basically FA1 fixed the memory bottleneck, but FA2 fixed the compute utilization bottleneck.

### The `mma.m16n8k16` Thread/Register Layout

It's computing a matrix multiply: C = A x B + C, where
- A is 16 rows x 16 cols (the m16k16 part)
- B is 8 rows x 16 cols (the n8k16 part, B is transposed)
- C/D output is 16 rows x 8 cols (the m16n8 part)

Here no single thread holds an entire row or column. The 32 threads in a warp collectively hold the entire matrix, each thread holding just a few elements. This is the "distributed register" model.

### The C/D Accumulator (Output) Layout

The output is 16x8 = 128 elements, split across 32 threads. So each threads holds 128/32 = 4 elements. These are the 4 registers: `c1, c2, c3, c4`.

Every thread holds exactly 4 floats. The rule is simple, thread `T` always owns rows `T/4` and `T/4 + 8`. So `T0` own rows 0 and 8 and `T1` own rows 1 and 9 and so on. This is why online softmax tracks `rowmax[2]` per thread, one value for each of its two rows.

Here's why it matters for softmax: `c0` and `c1` are awlays on the same row as each other (they're just two consecutive columns). Same for `c2` and `c3`. So when computing row max, you do `max(c0, c1)` for one row and `max(c2, c3)` for the other.

What A gives you that C/D doesn't: A has 8 registers instead of 4, because it covers 16 columns (the K dimension) instead of 8. The left half (`a0-a3`) cover cols 0-7, the right half (`a4-a7`) covers cols 8-15. But row ownership is identical.

### cp.async

The problem: memory and compute block each other. Without pipelining, the GPU does this every KV iteration.
1. Load K tile from global -> shared memory - WAIT
2. Load V tile from global -> shared memory - WAIT
3. Run MMA on K (tensor cores active)
4. Run MMA on C (tensor cores active)

While loading (steps 1-2), tensor cores sit idle. While computing (steps 3-4), memory is idle. That's wasted time.

`cp.async` -- fire and and forget memory loads

`cp.async.cg.shared.global [dst], [src], 16;`

This instruction tells the memory unit: "start copying 16 bytes from global to shared and don't stop my thread to wait for it

The thread keeps running. The memory unut works in the background. This is the key primitive that enables overlap.

`commit_group` - seal a batch of loads

`cp.async.commit_group;`

After issuing several `cp.async` calls, you call `commit_group` to "seal" them into a named group.

Think of it like mailing a package: the individual `cp.async` calls put items in the box and `commit_group` seals and ships the box. Each groups corresponds to one pipeline stage (eg "K tile for iteration 3")

wait_group N - wait for oldest group only

`cp.async.wait_group N;`

This says: "blocks until at most N groups are still in-flight"

If you have 3 groups in-flight and call `wait_group 2`, it wait until the oldest group finises and lets the other 2 keep loading

This is how you consume a tile (use it for MMA) while simultaneously prefetching the next one.

Putting it all together:

Before the loop: prefetch K[0] -> commit_group (1 in-flight)

Each iteration i:
1. Prefetch K[i+1] -> `commit_group` (2 in-flight)
2. `wait_group 1` -> oldest group (K[i]) finishes
3. Use K[i] for 1st MMA <- memory + compute overlap
4. Prefetch V[i] -> `commit_group` (2 in flight)
5. `wait_group 1` -> V[i] finishes
6. Use V[i] for 2nd MMA

At any moment, one tile is computing while another is loading.
