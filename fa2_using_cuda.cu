#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void flash_attn_v2_naive(
    const float* Q,  // [B, L, D]
    const float* K,  // [B, L, D]
    const float* V,  // [B, L, D]
    float* O,        // [B, L, D]
    int L, int D,
    int BLOCK_Q, int BLOCK_KV,
    float scale)
{
    // step 1: figure out which row of Q this thread owns
    // hint: you need batch index, q_tile index, and thread index
    int b = blockIdx.x;
    int q_tile = blockIdx.y;
    int q_row = threadIdx.x + q_tile * blockDim.x;       // actual row in [0, L)

    // step 2: bounds check — what if q_row >= L?
    if (q_row >= L) return;
    // step 3: get pointer to this thread's Q row
    const float* q_ptr = Q + b * L * D + q_row * D; // should point to Q[b, q_row, 0]

    // pointers to K, V, O for this batch
    const float* k_ptr_base = K + b * L * D;    // K[b, 0, 0]
    const float* v_ptr_base = V + b * L * D;   // V[b, 0, 0]
    float* o_ptr = O + b * L * D + q_row * D;   // O[b, q_row, 0]

    // online softmax state — one value per element of D for O,
    // one scalar per thread for m and l
    float m = -INFINITY;
    float l = 0.0f;
    float o[128] = {0.0f};   // assume D <= 128 for now, we'll fix later

    // KV loop
    int num_kv_tiles = (L + BLOCK_KV - 1) / BLOCK_KV;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * BLOCK_KV;
        int kv_end = min(kv_start + BLOCK_KV, L);

        // step 1: compute S[j] = dot(q_ptr, k_ptr) * scale
        //         for each row j in this KV tile
        //         store in a local array float s[BLOCK_KV]
        float s[64];
        for (int j = 0; j < kv_end - kv_start; j++) {
            const float* k_row = k_ptr_base + (kv_start + j) * D;
            s[j] = 0.0f;
            for (int d = 0; d < D; d++) {
                s[j] += q_ptr[d] * k_row[d];
            }
            s[j] *= scale;
        }

        // step 2: find tile max, update m
        //         m_new = max(m, max(s))
        float tile_max = -INFINITY;
        for (int j = 0; j < kv_end - kv_start; j++) {
            tile_max = fmaxf(tile_max, s[j]);
        }
        float m_new = fmaxf(m, tile_max);

        // step 3: compute exp_old = exp(m - m_new)
        //         rescale l and o
        float exp_old = expf(m - m_new);

        // step 4: for each j, accumulate exp(s[j] - m_new) into l
        //         and exp(s[j] - m_new) * v_row_j into o
        float l_new = l * exp_old;
        for (int d = 0; d < D; d++) {
            o[d] *= exp_old;
        }
        for (int j = 0; j < kv_end - kv_start; j++) {
            float exp_sj = expf(s[j] - m_new);
            l_new += exp_sj;
            const float* v_row = v_ptr_base + (kv_start + j) * D;
            for (int d = 0; d < D; d++) {
                o[d] += exp_sj * v_row[d];
            }
        }

        // step 5: update m = m_new
        m = m_new;
        l = l_new;
    }

    // normalize and write output
    for (int d = 0; d < D; d++)
        o_ptr[d] = o[d] / l;
}


void flash_attn_v2_naive_launch(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B, int L, int D,
    int BLOCK_Q, int BLOCK_KV)
{
    float scale = 1.0f / sqrtf((float)D);

    // grid: (B, num_q_tiles)
    // block: (BLOCK_Q,)
    dim3 grid(B, (L + BLOCK_Q - 1) / BLOCK_Q);
    dim3 block(BLOCK_Q);

    flash_attn_v2_naive<<<grid, block>>>(
        Q, K, V, O, L, D, BLOCK_Q, BLOCK_KV, scale);
}

torch::Tensor flash_attn_v2_naive_cuda(
    torch::Tensor Q,  // [B, L, D]
    torch::Tensor K,
    torch::Tensor V,
    int BLOCK_Q,
    int BLOCK_KV)
{
    int B = Q.size(0);
    int L = Q.size(1);
    int D = Q.size(2);

    auto O = torch::zeros_like(Q);

    flash_attn_v2_naive_launch(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        B, L, D, BLOCK_Q, BLOCK_KV);

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_v2_naive", &flash_attn_v2_naive_cuda, "Flash Attention v2 naive");
}