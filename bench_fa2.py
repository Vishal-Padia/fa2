import torch
import torch.nn.functional as F
import time

from examples.python.CuTeDSL.ampere.flash_attention_v2 import FlashAttentionForwardAmpere
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

B, Sq, Sk, N, H = 1, 4096, 4096, 16, 128
dtype = torch.float16

# must be contiguous and H (innermost dim) must be 16-byte aligned
# H=128 * 2 bytes = 256 bytes — fine
# layout: (B, Sq, N, H) contiguous
Q = torch.randn(B, Sq, N, H, dtype=dtype, device="cuda").contiguous()
K = torch.randn(B, Sk, N, H, dtype=dtype, device="cuda").contiguous()
V = torch.randn(B, Sk, N, H, dtype=dtype, device="cuda").contiguous()
O = torch.zeros(B, Sq, N, H, dtype=dtype, device="cuda").contiguous()

fa = FlashAttentionForwardAmpere(
    head_dim=H,
    m_block_size=128,
    n_block_size=128,
    num_threads=128
)

stream = torch.cuda.current_stream().cuda_stream
cu_stream = cuda.CUstream(stream)

scale = float(H ** -0.5)

# warmup — first call also compiles
print("Compiling...")
mQ = from_dlpack(Q)
mK = from_dlpack(K)
mV = from_dlpack(V)
mO = from_dlpack(O)
fa(mQ, mK, mV, mO, scale, cu_stream)
torch.cuda.synchronize()
print("Done. Benchmarking...")

N_ITER = 50
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    fa(mQ, mK, mV, mO, scale, cu_stream)
torch.cuda.synchronize()
t1 = time.perf_counter()

ms = (t1 - t0) / N_ITER * 1000
flops = 4 * B * N * Sq * Sk * H
tflops = flops / (ms / 1000) / 1e12
print(f"NVIDIA CuteDSL FA2: {ms:.3f} ms  |  {tflops:.2f} TFLOPS")

# pytorch sdpa — needs (B, N, Sq, H)
Qp = Q.permute(0,2,1,3).contiguous()
Kp = K.permute(0,2,1,3).contiguous()
Vp = V.permute(0,2,1,3).contiguous()

for _ in range(5):
    F.scaled_dot_product_attention(Qp, Kp, Vp)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    F.scaled_dot_product_attention(Qp, Kp, Vp)
torch.cuda.synchronize()
t1 = time.perf_counter()

ms_ref = (t1 - t0) / N_ITER * 1000
tflops_ref = flops / (ms_ref / 1000) / 1e12
print(f"PyTorch SDPA:       {ms_ref:.3f} ms  |  {tflops_ref:.2f} TFLOPS")

peak = 125.0  # A10G fp16 tensor core peak
print(f"\nNVIDIA CuteDSL: {tflops/peak*100:.1f}% of A10G peak")
print(f"PyTorch SDPA:   {tflops_ref/peak*100:.1f}% of A10G peak")