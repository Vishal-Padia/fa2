#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void flash_attn_v2_smem(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int L, int D,
    int BLOCK_Q, int BLOCK_KV,
    float scale)
{
    int b      = blockIdx.x;
    int q_tile = blockIdx.y;
    int q_row  = threadIdx.x + q_tile * blockDim.x;

    if (q_row >= L) return;

    const float* q_ptr      = Q + b * L * D + q_row * D;
    const float* k_ptr_base = K + b * L * D;
    const float* v_ptr_base = V + b * L * D;
    float* o_ptr = O + b * L * D + q_row * D;

    float m = -INFINITY;
    float l = 0.0f;
    float o[128] = {0.0f};

    extern __shared__ float smem[];
    float* K_smem = smem;
    float* V_smem = smem + BLOCK_KV * D;

    int num_kv_tiles = (L + BLOCK_KV - 1) / BLOCK_KV;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * BLOCK_KV;
        int kv_end   = min(kv_start + BLOCK_KV, L);

        // cooperative load of K tile
        for (int i = threadIdx.x; i < BLOCK_KV * D; i += blockDim.x) {
            int row        = i / D;
            int col        = i % D;
            int global_row = kv_start + row;
            K_smem[i] = (global_row < L) ? k_ptr_base[global_row * D + col] : 0.0f;
        }
        // cooperative load of V tile
        for (int i = threadIdx.x; i < BLOCK_KV * D; i += blockDim.x) {
            int row        = i / D;
            int col        = i % D;
            int global_row = kv_start + row;
            V_smem[i] = (global_row < L) ? v_ptr_base[global_row * D + col] : 0.0f;
        }

        __syncthreads();  // wait for all loads before anyone reads smem

        // compute attention scores
        float s[64];
        for (int j = 0; j < kv_end - kv_start; j++) {
            s[j] = 0.0f;
            for (int d = 0; d < D; d++)
                s[j] += q_ptr[d] * K_smem[j * D + d];
            s[j] *= scale;
        }

        // tile max
        float tile_max = -INFINITY;
        for (int j = 0; j < kv_end - kv_start; j++)
            tile_max = fmaxf(tile_max, s[j]);

        float m_new   = fmaxf(m, tile_max);
        float exp_old = expf(m - m_new);

        // rescale o BEFORE accumulating new values
        for (int d = 0; d < D; d++)
            o[d] *= exp_old;

        // update l_new starting from rescaled l
        float l_new = l * exp_old;

        // fix 3: accumulate exp_sj * V into o
        for (int j = 0; j < kv_end - kv_start; j++) {
            float exp_sj = expf(s[j] - m_new);
            l_new += exp_sj;
            for (int d = 0; d < D; d++)
                o[d] += exp_sj * V_smem[j * D + d];
        }

        // update m and l for next iteration
        m = m_new;
        l = l_new;

        __syncthreads();  // wait for all reads before next iter overwrites smem
    }

    // normalize and write output
    for (int d = 0; d < D; d++)
        o_ptr[d] = o[d] / l;
}

void flash_attn_v2_smem_launch(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B, int L, int D,
    int BLOCK_Q, int BLOCK_KV)
{
    float scale    = 1.0f / sqrtf((float)D);
    int   smem_size = 2 * BLOCK_KV * D * sizeof(float);  // K_smem + V_smem

    dim3 grid(B, (L + BLOCK_Q - 1) / BLOCK_Q);
    dim3 block(BLOCK_Q);

    flash_attn_v2_smem<<<grid, block, smem_size>>>(
        Q, K, V, O, L, D, BLOCK_Q, BLOCK_KV, scale);
}

torch::Tensor flash_attn_v2_smem_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int BLOCK_Q,
    int BLOCK_KV)
{
    int B = Q.size(0);
    int L = Q.size(1);
    int D = Q.size(2);

    auto O = torch::zeros_like(Q);

    flash_attn_v2_smem_launch(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        B, L, D, BLOCK_Q, BLOCK_KV);

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_v2_smem", &flash_attn_v2_smem_cuda, "Flash Attention v2 shared memory");
}