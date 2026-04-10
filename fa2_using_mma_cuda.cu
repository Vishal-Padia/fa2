#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int BLOCK_Q   = 64;
constexpr int BLOCK_KV  = 64;
constexpr int D         = 64;
constexpr int NUM_WARPS = 4;
constexpr int WARP_Q    = BLOCK_Q / NUM_WARPS;  // 16
constexpr int TB_SIZE   = NUM_WARPS * 32;        // 128

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

// ----------------------------------------------------------------
// PTX helpers
// ----------------------------------------------------------------
__device__ inline void ldmatrix_x4(uint32_t* reg, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
                 : "r"(addr));
}

__device__ inline void ldmatrix_x2(uint32_t* reg, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(reg[0]), "=r"(reg[1])
                 : "r"(addr));
}

__device__ inline void ldmatrix_x2_trans(uint32_t* reg, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(reg[0]), "=r"(reg[1])
                 : "r"(addr));
}

__device__ inline void mma_m16n8k16(
    uint32_t* A, uint32_t* B, float* C)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]));
}

// ----------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------
__global__ void flash_attn_v2_mma(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    float* O,
    int L, float scale)
{
    int b       = blockIdx.x;
    int q_tile  = blockIdx.y;
    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // shared memory
    extern __shared__ uint8_t smem[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* V_smem = reinterpret_cast<__nv_bfloat16*>(smem + BLOCK_KV * D * sizeof(__nv_bfloat16));

    // registers
    uint32_t Q_rmem[1][4][4] = {};
    uint32_t K_rmem[8][4][2] = {};
    uint32_t P_rmem[1][4][4] = {};
    uint32_t V_rmem[4][8][2] = {};
    float    S_rmem[1][8][4] = {};
    float    O_rmem[1][8][4] = {};

    // ----------------------------------------------------------------
    // step 1: Q global → shared
    // ----------------------------------------------------------------
    const __nv_bfloat16* q_global = Q + b * L * D + q_tile * BLOCK_Q * D;

    // each chunk = 8 bfloat16 = 16 bytes; cp.async.cg requires 16-byte alignment on both ends
    for (int chunk = tid; chunk < (BLOCK_Q * D) / 8; chunk += TB_SIZE) {
        int elem       = chunk * 8;
        int row        = elem / D;
        int col        = elem % D;           // multiple of 8 since D=64 → 16-byte-aligned source
        int global_row = q_tile * BLOCK_Q + row;
        uint32_t dst   = static_cast<uint32_t>(__cvta_generic_to_shared(&Q_smem[elem]));
        if (global_row < L) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                         :: "r"(dst), "l"(&q_global[row * D + col]));
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // ----------------------------------------------------------------
    // step 2: Q shared → registers
    // ----------------------------------------------------------------
    for (int mma_id_d = 0; mma_id_d < D / MMA_K; mma_id_d++) {
        int row      = warp_id * WARP_Q + (lane_id % 16);
        int col      = mma_id_d * MMA_K + (lane_id / 16) * 8;
        uint32_t addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Q_smem[row * D + col]));
        ldmatrix_x4(Q_rmem[0][mma_id_d], addr);
    }
    __syncthreads();  // Q_smem now safe to reuse for K

    // ----------------------------------------------------------------
    // step 3: KV loop
    // ----------------------------------------------------------------
    // online softmax state — per thread, per warp-Q row
    // each thread owns 2 rows (c0/c1 = row0, c2/c3 = row1)
    float rowmax[2]    = {-INFINITY, -INFINITY};
    float rowsumexp[2] = {0.0f, 0.0f};

    int num_kv_tiles = (L + BLOCK_KV - 1) / BLOCK_KV;

    for (int kv_idx = 0; kv_idx < num_kv_tiles; kv_idx++) {
        int kv_start = kv_idx * BLOCK_KV;

        // -- 3a: load K global → shared --
        const __nv_bfloat16* k_global = K + b * L * D;
        for (int chunk = tid; chunk < (BLOCK_KV * D) / 8; chunk += TB_SIZE) {
            int elem       = chunk * 8;
            int row        = elem / D;
            int col        = elem % D;       // multiple of 8 → 16-byte-aligned source
            int global_row = kv_start + row;
            uint32_t dst   = static_cast<uint32_t>(__cvta_generic_to_shared(&K_smem[elem]));
            if (global_row < L) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                             :: "r"(dst), "l"(&k_global[global_row * D + col]));
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        // -- 3b: load K shared → registers --
        // K acts as B operand (n8k16), use ldmatrix_x2
        // TODO: write this loop
        // hint: for each mma_id_kv in [BLOCK_KV/MMA_N] and mma_id_d in [D/MMA_K]
        //       row = mma_id_kv * MMA_N + (lane_id % 8)
        //       col = mma_id_d  * MMA_K + (lane_id / 8) * 8
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            for (int mma_id_d = 0; mma_id_d < D / MMA_K; mma_id_d++) {
                int row = mma_id_kv * MMA_N + (lane_id % 8);
                int col = mma_id_d * MMA_K + (lane_id / 8) * 8;
                uint32_t addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&K_smem[row * D + col])
                );
                ldmatrix_x2(K_rmem[mma_id_kv][mma_id_d], addr);
            }
        }

        // -- 3c: reset S accumulator --
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 4; j++)
                S_rmem[0][i][j] = 0.0f;

        // -- 3d: 1st MMA: S = Q @ K.T --
        // TODO: write the triple loop over mma_id_kv, mma_id_d
        // call mma_m16n8k16(Q_rmem[0][mma_id_d], K_rmem[mma_id_kv][mma_id_d], S_rmem[0][mma_id_kv])
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            for (int mma_id_d = 0; mma_id_d < D / MMA_K; mma_id_d++) {
                mma_m16n8k16(Q_rmem[0][mma_id_d], K_rmem[mma_id_kv][mma_id_d], S_rmem[0][mma_id_kv]);
            }
        }

        // -- 3e: scale S --
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 4; j++)
                S_rmem[0][i][j] *= scale;

        // -- 3f: online softmax --
        // TODO: write this — use rowmax[2] and rowsumexp[2]
        // remember: c0,c1 belong to row0 (rowmax[0]), c2,c3 belong to row1 (rowmax[1])
        // you need __shfl_xor_sync for the butterfly reduction

        // find row max across all mma_id_kv tiles
        float tile_max[2] = {-INFINITY, -INFINITY};
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            float* s = S_rmem[0][mma_id_kv];
            //c0, c1 -> row0
            //c2, c3 -> row 1
            tile_max[0] = fmaxf(tile_max[0], fmaxf(s[0], s[1]));
            tile_max[1] = fmaxf(tile_max[1], fmaxf(s[2], s[3]));
        }
        // butterfly reduction across 4 threads in the same row group
        tile_max[0] = fmaxf(tile_max[0], __shfl_xor_sync(0xffffffff, tile_max[0], 1));
        tile_max[0] = fmaxf(tile_max[0], __shfl_xor_sync(0xffffffff, tile_max[0], 2));
        tile_max[1] = fmaxf(tile_max[1], __shfl_xor_sync(0xffffffff, tile_max[1], 1));
        tile_max[1] = fmaxf(tile_max[1], __shfl_xor_sync(0xffffffff, tile_max[1], 2));

        // update global rowmax and compute rescale
        float new_rowmax[2] = {
            fmaxf(rowmax[0], tile_max[0]),
            fmaxf(rowmax[1], tile_max[1])
        };
        float rescale[2] = {
            expf(rowmax[0] - new_rowmax[0]),
            expf(rowmax[1] - new_rowmax[1])
        };

        // rescale O_remm and rowsumexp
        for (int mma_id_d = 0; mma_id_d < D / MMA_N; mma_id_d++) {
            float* o = O_rmem[0][mma_id_d];
            o[0] *= rescale[0]; o[1] *= rescale[0];
            o[2] *= rescale[1]; o[3] *= rescale[1];
        }
        rowsumexp[0] *= rescale[0];
        rowsumexp[1] *= rescale[1];

        // compute exp scores, pack S->P, accumulate rowsumexp
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            float* s = S_rmem[0][mma_id_kv];
            float exp_s[4] = {
                expf(s[0] - new_rowmax[0]),
                expf(s[1] - new_rowmax[0]),
                expf(s[2] - new_rowmax[1]),
                expf(s[3] - new_rowmax[1])
            };
            rowsumexp[0] += exp_s[0] + exp_s[1];
            rowsumexp[1] += exp_s[2] + exp_s[3];

            // pack S (m16n8 float) -> P (m16k16 bf16)
            // two consecutive mma_id_kv iterations fill one P tile
            __nv_bfloat162 p01 = __floats2bfloat162_rn(exp_s[0], exp_s[1]); // row0 pair
            __nv_bfloat162 p23 = __floats2bfloat162_rn(exp_s[2], exp_s[3]); // row1 pair

            // pack into P_rmem - two mma_id_kv iterations fill one P tile (m16k16)
            // first half (mma_id_kv % 2 == 0); fills positions 0,1
            // second half (mma_id_kv % 2 == 1); fills positions 2,3
            int p_tile = mma_id_kv / 2;
            int p_half = mma_id_kv % 2;
            reinterpret_cast<__nv_bfloat162*>(P_rmem[0][p_tile])[p_half * 2 + 0] = p01;
            reinterpret_cast<__nv_bfloat162*>(P_rmem[0][p_tile])[p_half * 2 + 1] = p23;
        }

        // update rowmax
        rowmax[0] = new_rowmax[0];
        rowmax[1] = new_rowmax[1];

        // -- 3g: load V global → shared --
        const __nv_bfloat16* v_global = V + b * L * D;
        for (int chunk = tid; chunk < (BLOCK_KV * D) / 8; chunk += TB_SIZE) {
            int elem       = chunk * 8;
            int row        = elem / D;
            int col        = elem % D;       // multiple of 8 → 16-byte-aligned source
            int global_row = kv_start + row;
            uint32_t dst   = static_cast<uint32_t>(__cvta_generic_to_shared(&V_smem[elem]));
            if (global_row < L) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                             :: "r"(dst), "l"(&v_global[global_row * D + col]));
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();
        
        // -- 3h: load V shared → registers --
        // V acts as B operand but TRANSPOSED, use ldmatrix_x2_trans
        // TODO: write this loop
        for (int mma_id_kv = 0; mma_id_kv < D / MMA_K; mma_id_kv++) {
            for (int mma_id_d = 0; mma_id_d < BLOCK_KV / MMA_N; mma_id_d++) {
                int row = mma_id_d * MMA_K + (lane_id % 8);
                int col = mma_id_kv * MMA_N + (lane_id / 8) * 8;
                uint32_t addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&V_smem[row * D + col])
                );
                ldmatrix_x2_trans(V_rmem[mma_id_d][mma_id_kv], addr);
            }
        }


        // -- 3i: 2nd MMA: O += P @ V --
        // TODO: write the triple loop
        // call mma_m16n8k16(P_rmem[0][mma_id_kv/2 * ...], V_rmem[...], O_rmem[0][mma_id_d])
        for (int mma_id_d = 0; mma_id_d < D / MMA_N; mma_id_d++) {
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
                mma_m16n8k16(
                    P_rmem[0][mma_id_kv],
                    V_rmem[mma_id_kv][mma_id_d],
                    O_rmem[0][mma_id_d]
                );
            }
        }

        __syncthreads();
    }

    // ----------------------------------------------------------------
    // step 4: normalize and write output
    // TODO: divide O_rmem by rowsumexp, write to O global
    // ----------------------------------------------------------------
    for (int mma_id_d = 0; mma_id_d < D / MMA_N; mma_id_d++) {
        float* o = O_rmem[0][mma_id_d];
        o[0] /= rowsumexp[0]; // c0 -> row0
        o[1] /= rowsumexp[0]; // c1 -> row0
        o[2] /= rowsumexp[1]; // c2 -> row1
        o[3] /= rowsumexp[1]; // c3 -> row1
    }
    float* o_global = O + b * L * D;
    for (int mma_id_d = 0; mma_id_d < D / MMA_N; mma_id_d++) {
        float* o = O_rmem[0][mma_id_d];
    
        // row0: warp_id*WARP_Q + lane_id/4
        // row1: row0 + 8
        int row0 = warp_id * WARP_Q + (lane_id / 4);
        int row1 = row0 + 8;
        int col  = mma_id_d * MMA_N + (lane_id % 4) * 2;
    
        int global_row0 = q_tile * BLOCK_Q + row0;
        int global_row1 = q_tile * BLOCK_Q + row1;
    
        if (global_row0 < L) {
            o_global[global_row0 * D + col + 0] = o[0];
            o_global[global_row0 * D + col + 1] = o[1];
        }
        if (global_row1 < L) {
            o_global[global_row1 * D + col + 0] = o[2];
            o_global[global_row1 * D + col + 1] = o[3];
        }
    }
}

// ----------------------------------------------------------------
// Launcher
// ----------------------------------------------------------------
void flash_attn_v2_mma_launch(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    float* O,
    int B, int L)
{
    float scale     = 1.0f / sqrtf((float)D);
    int   smem_size = max(BLOCK_Q, BLOCK_KV * 2) * D * sizeof(__nv_bfloat16);

    dim3 grid(B, (L + BLOCK_Q - 1) / BLOCK_Q);
    dim3 block(TB_SIZE);

    if (smem_size > 48 * 1024)
        cudaFuncSetAttribute(flash_attn_v2_mma,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);

    flash_attn_v2_mma<<<grid, block, smem_size>>>(Q, K, V, O, L, scale);
}

torch::Tensor flash_attn_v2_mma_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V)
{
    int B = Q.size(0);
    int L = Q.size(1);

    auto O = torch::zeros({B, L, D}, torch::dtype(torch::kFloat32).device(Q.device()));

    flash_attn_v2_mma_launch(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>()),
        O.data_ptr<float>(),
        B, L);

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_v2_mma", &flash_attn_v2_mma_cuda, "Flash Attention v2 MMA");
}