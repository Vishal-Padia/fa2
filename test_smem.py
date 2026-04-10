import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# this compiles your .cu file on the fly
attn = load(
    name="flash_attn_v2_smem",
    sources=["fa2_using_smem_cuda.cu"],
    extra_cuda_cflags=["-O2"],
    verbose=True
)

# correctness test
B, L, D = 2, 256, 64
Q = torch.randn(B, L, D).cuda()
K = torch.randn(B, L, D).cuda()
V = torch.randn(B, L, D).cuda()

# ref = F.scaled_dot_product_attention(Q, K, V)
# out = attn.flash_attn_v2_naive(Q, K, V, 64, 64)

# print("max diff:", (ref - out).abs().max().item())
# print("allclose:", torch.allclose(ref, out, atol=1e-5))


# warmup
for _ in range(5):
    out = attn.flash_attn_v2_smem(Q, K, V, 64, 64)

# benchmark
import time
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    out = attn.flash_attn_v2_smem(Q, K, V, 64, 64)
torch.cuda.synchronize()
elapsed = (time.time() - start) / 100 * 1000

ref_out = F.scaled_dot_product_attention(Q, K, V)

# warmup ref
for _ in range(5):
    ref_out = F.scaled_dot_product_attention(Q, K, V)

torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    ref_out = F.scaled_dot_product_attention(Q, K, V)
torch.cuda.synchronize()
elapsed_ref = (time.time() - start) / 100 * 1000

print(f"smem kernel: {elapsed:.3f} ms")
print(f"pytorch sdpa: {elapsed_ref:.3f} ms")