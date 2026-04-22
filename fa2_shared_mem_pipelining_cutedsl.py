import torch
import torch.nn.functional as F
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda


BLOCK_Q  = 64
BLOCK_KV = 64
HEAD_DIM = 128
NUM_THREADS = 128  # 4 warps


class FlashAttnPipelined:
    def __init__(self):
        pass

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,   # [B, Sq, N, H]
        mK: cute.Tensor,   # [B, Sk, N, H]
        mV: cute.Tensor,   # [B, Sk, N, H]
        mO: cute.Tensor,   # [B, Sq, N, H]
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        smem_k_block_size = 64  # HEAD_DIM % 64 == 0
        sw = cute.make_swizzle(3, 3, 3)

        sQ_layout_atom = cute.make_composed_layout(
            sw, 0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1))
        )
        sQ_layout = cute.tile_to_shape(sQ_layout_atom, (BLOCK_Q, HEAD_DIM), (0, 1))

        sKV_layout_atom = cute.make_composed_layout(
            sw, 0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1))
        )
        sKV_layout = cute.tile_to_shape(sKV_layout_atom, (BLOCK_KV, HEAD_DIM), (0, 1))
        sV_layout = cute.make_layout((BLOCK_KV, HEAD_DIM), stride=(HEAD_DIM, 1))

        @cute.struct
        class SharedStorage:
            sQ:  cute.struct.MemRange[mQ.element_type, BLOCK_Q  * HEAD_DIM]
            sK0: cute.struct.MemRange[mQ.element_type, BLOCK_KV * HEAD_DIM]
            sK1: cute.struct.MemRange[mQ.element_type, BLOCK_KV * HEAD_DIM]
            sV0: cute.struct.MemRange[mQ.element_type, BLOCK_KV * HEAD_DIM]

        elems_per_copy = 128 // mQ.element_type.width
        cp_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            mQ.element_type,
            num_bits_per_copy=128,
        )
        tQKV_layout = cute.make_layout(
            (NUM_THREADS // (HEAD_DIM // elems_per_copy),
             HEAD_DIM // elems_per_copy),
            stride=(HEAD_DIM // elems_per_copy, 1),
        )
        vQKV_layout = cute.make_layout((1, elems_per_copy))

        gmem_tiled_copy = cute.make_tiled_copy_tv(
            cp_atom, tQKV_layout, vQKV_layout
        )

        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mQ.element_type,
            num_bits_per_copy=128,
        )
        gmem_tiled_store = cute.make_tiled_copy_tv(
            store_atom, tQKV_layout, vQKV_layout
        )

        from cutlass.cute.nvgpu import warp
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(mQ.element_type, cutlass.Float32, (16, 8, 16)),
            (NUM_THREADS // 32, 1, 1),
            permutation_mnk=(NUM_THREADS // 32 * 16, 16, 16),
        )

        grid = (
            cute.ceil_div(mQ.shape[1], BLOCK_Q),
            mQ.shape[0],
            mQ.shape[2],
        )

        self.kernel(
            mQ, mK, mV, mO,
            scale,
            sQ_layout, sKV_layout, sV_layout,
            gmem_tiled_copy,
            gmem_tiled_store,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid,
            block=[NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        scale: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sV_layout: cute.Layout,
        gmem_tiled_copy: cute.TiledCopy,
        gmem_tiled_store: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_tile, batch, head = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ  = storage.sQ.get_tensor(sQ_layout)
        sK0 = storage.sK0.get_tensor(sKV_layout)
        sK1 = storage.sK1.get_tensor(sKV_layout)
        sV0 = storage.sV0.get_tensor(sV_layout)

        gQ = cute.local_tile(
            mQ[batch, None, head, None],
            (BLOCK_Q, HEAD_DIM), (q_tile, 0),
        )
        gK = cute.local_tile(
            mK[batch, None, head, None],
            (BLOCK_KV, HEAD_DIM), (None, 0),
        )
        gV = cute.local_tile(
            mV[batch, None, head, None],
            (BLOCK_KV, HEAD_DIM), (None, 0),
        )
        gO = cute.local_tile(
            mO[batch, None, head, None],
            (BLOCK_Q, HEAD_DIM), (q_tile, 0),
        )

        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)

        tQgQ = gmem_thr_copy.partition_S(gQ)
        tQsQ = gmem_thr_copy.partition_D(sQ)

        tKgK = gmem_thr_copy.partition_S(gK)
        tKsk0 = gmem_thr_copy.partition_D(sK0)
        tKsk1 = gmem_thr_copy.partition_D(sK1)

        tVgV = gmem_thr_copy.partition_S(gV)
        tVsV0 = gmem_thr_copy.partition_D(sV0)
        tVsV1 = gmem_thr_copy.partition_D(sV0)

        # Load Q once
        cute.copy(gmem_tiled_copy, tQgQ, tQsQ)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        thr_mma = tiled_mma.get_slice(tidx)

        acc_O_shape = thr_mma.partition_shape_C((BLOCK_Q, HEAD_DIM))
        acc_O = cute.make_rmem_tensor(acc_O_shape, cutlass.Float32)
        acc_O.fill(0.0)

        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1],), cutlass.Float32
        )
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1],), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        num_kv_tiles = cute.ceil_div(mK.shape[1], BLOCK_KV)

        smem_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mQ.element_type
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma)

        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tsrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)

        m_atom = cute.size(acc_O.shape[0][0])
        acc_S_shape = thr_mma.partition_shape_C((BLOCK_Q, BLOCK_KV))
        n_atom = cute.size(acc_S_shape[0][1])
        n_tiles = cute.size(acc_S_shape[2])
        o_n_atom = cute.size(acc_O.shape[0][1])
        o_n_tiles = cute.size(acc_O.shape[2])

        # Prologue: kick off K[0]
        cute.copy(gmem_tiled_copy, tKgK[None, None, None, 0], tKsk0)
        cute.arch.cp_async_commit_group()

        for kv in range(num_kv_tiles):
            sK_cur = sK0
            sV_cur = sV0
            tVsV_cur = tVsV0
            tKsK_next = tKsk1
            tVsV_next = tVsV1
            if kv % 2 == 0:
                sK_cur = sK0
                sV_cur = sV0
                tVsV_cur = tVsV0
                tKsK_next = tKsk1
                tVsV_next = tVsV1
            else:
                sK_cur = sK1
                sV_cur = sV0
                tVsV_cur = tVsV1
                tKsK_next = tKsk0
                tVsV_next = tVsV0

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            acc_S = cute.make_rmem_tensor(acc_S_shape, cutlass.Float32)
            acc_S.fill(0.0)

            tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK_cur))
            tSsK = smem_thr_copy_K.partition_S(sK_cur)
            tSrK_copy_view = smem_thr_copy_K.retile(tSrK)

            # Overlap V[kv] load with first MMA
            cute.copy(gmem_tiled_copy, tVgV[None, None, None, kv], tVsV_cur)
            cute.arch.cp_async_commit_group()

            for k in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                cute.copy(smem_tiled_copy_Q, tSsQ[None, None, k], tsrQ_copy_view[None, None, k])
                cute.copy(smem_tiled_copy_K, tSsK[None, None, k], tSrK_copy_view[None, None, k])
                cute.gemm(tiled_mma, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)

            for i in cutlass.range_constexpr(cute.size(acc_S)):
                acc_S[i] = acc_S[i] * scale

            for row in cutlass.range_constexpr(cute.size(row_max)):
                mi = row % m_atom
                mt = row // m_atom

                tile_max = -cutlass.Float32.inf
                for nt in cutlass.range_constexpr(n_tiles):
                    for ni in cutlass.range_constexpr(n_atom):
                        tile_max = max(tile_max, acc_S[(ni, mi), mt, nt])

                tile_max = cute.arch.fmax(tile_max, cute.arch.shuffle_sync_bfly(tile_max, offset=2, mask=-1, mask_and_clamp=31))
                tile_max = cute.arch.fmax(tile_max, cute.arch.shuffle_sync_bfly(tile_max, offset=1, mask=-1, mask_and_clamp=31))

                new_rowmax = max(row_max[row], tile_max)
                rescale = cute.math.exp(row_max[row] - new_rowmax)

                for ot in cutlass.range_constexpr(o_n_tiles):
                    for oi in cutlass.range_constexpr(o_n_atom):
                        acc_O[(oi, mi), mt, ot] *= rescale
                row_sum[row] *= rescale

                for nt in cutlass.range_constexpr(n_tiles):
                    for ni in cutlass.range_constexpr(n_atom):
                        p = cute.math.exp(acc_S[(ni, mi), mt, nt] - new_rowmax)
                        acc_S[(ni, mi), mt, nt] = p
                        row_sum[row] += p

                row_max[row] = new_rowmax

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # Overlap K[kv+1] load with second MMA
            if kv + 1 < num_kv_tiles:
                cute.copy(gmem_tiled_copy, tKgK[None, None, None, kv + 1], tKsK_next)
                cute.arch.cp_async_commit_group()

            rP = cute.make_fragment_like(acc_S, mQ.element_type)
            rP.store(acc_S.load().to(mQ.element_type))

            rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
            rP_mma_view = cute.make_layout(
                (
                    (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                    rP_layout_divided.shape[1],
                    rP_layout_divided.shape[2][1],
                ),
                stride=(
                    (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                    rP_layout_divided.stride[1],
                    rP_layout_divided.stride[2][1],
                ),
            )
            tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

            sVt = cute.make_tensor(
                sV_cur.iterator,
                cute.make_layout((HEAD_DIM, BLOCK_KV), stride=(1, HEAD_DIM))
            )
            tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
            tOsVt = smem_thr_copy_V.partition_S(sVt)
            tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

            for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
                cute.copy(smem_tiled_copy_V, tOsVt[None, None, k], tOrVt_copy_view[None, None, k])
                cute.gemm(tiled_mma, acc_O, tOrS[None, None, k], tOrVt[None, None, k], acc_O)

        # Epilogue
        for row in cutlass.range_constexpr(cute.size(row_max)):
            mi = row % m_atom
            mt = row // m_atom
            rs = row_sum[row]
            rs = rs + cute.arch.shuffle_sync_bfly(rs, offset=2, mask=-1, mask_and_clamp=31)
            rs = rs + cute.arch.shuffle_sync_bfly(rs, offset=1, mask=-1, mask_and_clamp=31)
            for ot in cutlass.range_constexpr(o_n_tiles):
                for oi in cutlass.range_constexpr(o_n_atom):
                    acc_O[(oi, mi), mt, ot] /= rs

        rO = cute.make_fragment_like(acc_O, mQ.element_type)
        rO.store(acc_O.load().to(mQ.element_type))

        sO = cute.make_tensor(sQ.iterator, sQ.layout)

        smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom, tiled_mma)
        smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOrsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_tiled_copy_O, taccOrO, taccOrsO)
        cute.arch.sync_threads()

        gmem_thr_store = gmem_tiled_store.get_slice(tidx)
        tOgsO = gmem_thr_store.partition_S(sO)
        tOgO = gmem_thr_store.partition_D(gO)
        cute.copy(gmem_tiled_store, tOgsO, tOgO)


def pytorch_reference(Q, K, V, scale):
    attn = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, V.float())
    return out.to(Q.dtype)


def run_flash_attn(B, N, Sq, Sk, H, dtype=torch.float16, num_warmup=5, num_iters=20):
    assert H == HEAD_DIM, f"Head dim must be {HEAD_DIM}, got {H}"
    assert Sq % BLOCK_Q == 0, f"Sq={Sq} must be divisible by BLOCK_Q={BLOCK_Q}"
    assert Sk % BLOCK_KV == 0, f"Sk={Sk} must be divisible by BLOCK_KV={BLOCK_KV}"

    torch.manual_seed(42)
    device = torch.device("cuda:0")

    Q = torch.randn(B, Sq, N, H, dtype=dtype, device=device)
    K = torch.randn(B, Sk, N, H, dtype=dtype, device=device)
    V = torch.randn(B, Sk, N, H, dtype=dtype, device=device)
    O = torch.zeros(B, Sq, N, H, dtype=dtype, device=device)

    scale = 1.0 / (H ** 0.5)

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    cQ = from_dlpack(Q, assumed_align=16)
    cK = from_dlpack(K, assumed_align=16)
    cV = from_dlpack(V, assumed_align=16)
    cO = from_dlpack(O, assumed_align=16)

    fa = FlashAttnPipelined()

    print(f"Config: B={B}, N={N}, Sq={Sq}, Sk={Sk}, H={H}, dtype={dtype}")
    print(f"Tiles:  BLOCK_Q={BLOCK_Q}, BLOCK_KV={BLOCK_KV}, NUM_THREADS={NUM_THREADS}")
    print(f"Scale:  {scale:.6f}")
    print()

    print("Compiling kernel...")
    compiled_fa = cute.compile(fa, cQ, cK, cV, cO, scale, stream)
    print("  Compilation done.")

    print("Warming up...")
    for i in range(num_warmup):
        O.zero_()
        cO = from_dlpack(O, assumed_align=16)
        compiled_fa(cQ, cK, cV, cO, scale, stream)
        torch.cuda.synchronize()
    print(f"  {num_warmup} warmup iters done.")
    print()

    print("=" * 60)
    print("CORRECTNESS CHECK")
    print("=" * 60)

    O.zero_()
    cO = from_dlpack(O, assumed_align=16)
    compiled_fa(cQ, cK, cV, cO, scale, stream)
    torch.cuda.synchronize()

    Q_ref = Q.permute(0, 2, 1, 3)
    K_ref = K.permute(0, 2, 1, 3)
    V_ref = V.permute(0, 2, 1, 3)
    O_ref = pytorch_reference(Q_ref, K_ref, V_ref, scale)
    O_ref = O_ref.permute(0, 2, 1, 3)

    abs_diff = (O.float() - O_ref.float()).abs()
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()

    denom = O_ref.float().abs().clamp(min=1e-6)
    rel_diff = abs_diff / denom
    max_rel_err = rel_diff.max().item()
    mean_rel_err = rel_diff.mean().item()

    O_flat = O.float().reshape(-1, H)
    O_ref_flat = O_ref.float().reshape(-1, H)
    cos_sim = F.cosine_similarity(O_flat, O_ref_flat, dim=-1)
    min_cos_sim = cos_sim.min().item()
    mean_cos_sim = cos_sim.mean().item()

    print(f"  Max  absolute error: {max_abs_err:.6e}")
    print(f"  Mean absolute error: {mean_abs_err:.6e}")
    print(f"  Max  relative error: {max_rel_err:.6e}")
    print(f"  Mean relative error: {mean_rel_err:.6e}")
    print(f"  Min  cosine sim:     {min_cos_sim:.6f}")
    print(f"  Mean cosine sim:     {mean_cos_sim:.6f}")
    print()

    PASS = max_abs_err < 5e-2 and mean_cos_sim > 0.999
    print(f"  VERDICT: {'PASS ✓' if PASS else 'FAIL ✗'}")
    if not PASS:
        print()
        print("  Sample output (first row, first head):")
        print(f"    Ours: {O[0, 0, 0, :8].tolist()}")
        print(f"    Ref:  {O_ref[0, 0, 0, :8].tolist()}")
    print()

    print("=" * 60)
    print("TIMING")
    print("=" * 60)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for _ in range(num_iters):
        O.zero_()
        cO = from_dlpack(O, assumed_align=16)
        torch.cuda.synchronize()
        start_event.record()
        compiled_fa(cQ, cK, cV, cO, scale, stream)
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))

    times_ms.sort()
    trim = max(1, num_iters // 5)
    trimmed = times_ms[trim:-trim] if len(times_ms) > 2 * trim else times_ms
    avg_ms = sum(trimmed) / len(trimmed)
    min_ms = times_ms[0]
    max_ms = times_ms[-1]
    med_ms = times_ms[num_iters // 2]

    total_flops = 4 * B * N * Sq * Sk * H
    tflops = (total_flops / (avg_ms / 1000)) / 1e12

    print(f"  Iters:    {num_iters}")
    print(f"  Min:      {min_ms:.3f} ms")
    print(f"  Max:      {max_ms:.3f} ms")
    print(f"  Median:   {med_ms:.3f} ms")
    print(f"  Mean:     {avg_ms:.3f} ms  (trimmed)")
    print(f"  TFLOP/s:  {tflops:.2f}")
    print()

    print("=" * 60)
    print("PyTorch SDPA comparison")
    print("=" * 60)

    Q_sdpa = Q.permute(0, 2, 1, 3).contiguous()
    K_sdpa = K.permute(0, 2, 1, 3).contiguous()
    V_sdpa = V.permute(0, 2, 1, 3).contiguous()

    for _ in range(num_warmup):
        _ = F.scaled_dot_product_attention(Q_sdpa, K_sdpa, V_sdpa, scale=scale)
    torch.cuda.synchronize()

    sdpa_times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start_event.record()
        _ = F.scaled_dot_product_attention(Q_sdpa, K_sdpa, V_sdpa, scale=scale)
        end_event.record()
        torch.cuda.synchronize()
        sdpa_times.append(start_event.elapsed_time(end_event))

    sdpa_times.sort()
    sdpa_trimmed = sdpa_times[trim:-trim] if len(sdpa_times) > 2 * trim else sdpa_times
    sdpa_avg = sum(sdpa_trimmed) / len(sdpa_trimmed)
    sdpa_tflops = (total_flops / (sdpa_avg / 1000)) / 1e12

    print(f"  Mean:     {sdpa_avg:.3f} ms  (trimmed)")
    print(f"  TFLOP/s:  {sdpa_tflops:.2f}")
    print(f"  Ours vs SDPA: {avg_ms / sdpa_avg:.2f}x slower")
    print()

    return {
        "pass": PASS,
        "max_abs_err": max_abs_err,
        "mean_cos_sim": mean_cos_sim,
        "our_ms": avg_ms,
        "sdpa_ms": sdpa_avg,
        "our_tflops": tflops,
        "sdpa_tflops": sdpa_tflops,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Flash Attention v2: Pipelined CuteDSL")
    print("=" * 60)
    print()

    results = run_flash_attn(
        B=2, N=4, Sq=256, Sk=256, H=128,
        dtype=torch.float16, num_warmup=5, num_iters=20,
    )

    if results["pass"]:
        print()
        print("Small test passed! Running larger benchmark...")
        print()
        run_flash_attn(
            B=2, N=8, Sq=1024, Sk=1024, H=128,
            dtype=torch.float16, num_warmup=5, num_iters=50,
        )
