import torch
import torch.nn.functional as F

def flash_attn_v2(Q, K, V, BLOCK_Q=64, BLOCK_KV=64):
    # Q, K, V: [B, L, D]
    # for each batch, for each Q tile:
    #   load Q tile, keep it fixed
    #   loop over all KV tiles, accumulate O with online softmax
    B, L, D = Q.shape
    scale = 1.0 / (D ** 0.5)

    # output tensor
    O = torch.zeros_like(Q)

    # num of tiles
    num_q_tiles = (L + BLOCK_Q - 1) // BLOCK_Q
    num_kv_tiles = (L + BLOCK_KV - 1) // BLOCK_KV

    for b in range(B):
        for q_tile_idx in range(num_q_tiles):
            # load q tile
            q_start = q_tile_idx * BLOCK_Q # [BLOCK_Q, D]
            q_end = min(q_start + BLOCK_Q, L)
            Q_tile = Q[b, q_start:q_end, :] # [actual_q_size, D]

            # initialize running statistics
            m = torch.full((q_end - q_start,), -torch.inf, device=Q.device)
            l = torch.zeros(q_end - q_start, device=Q.device)
            O_tile = torch.zeros((q_end - q_start, D), device=Q.device) # [actual_q_size, D]

            # loop over all KV tiles
            for kv_tile_idx in range(num_kv_tiles):
                # load KV tile
                kv_start = kv_tile_idx * BLOCK_KV # [BLOCK_KV, D]
                kv_end = min(kv_start + BLOCK_KV, L)
                K_tile = K[b, kv_start:kv_end, :] # [actual_kv_size, D]
                V_tile = V[b, kv_start:kv_end, :] # [actual_kv_size, D]

                # compute the attn scores for this tile pair
                S_tile = (Q_tile @ K_tile.T) * scale

                # online softmax update
                m_new = torch.maximum(m, S_tile.max(dim=1).values)

                # compute exp of scores with old and new max
                exp_old = torch.exp(m - m_new)
                exp_scores = torch.exp(S_tile - m_new.unsqueeze(1))

                # update the running sum
                l_new = l * exp_old + exp_scores.sum(dim=1)
                
                # update accumlated output
                O_tile = (O_tile * exp_old.unsqueeze(1)) + (exp_scores @ V_tile)

                # udpate running stats
                m = m_new
                l = l_new
            
            # write O_tile back to output
            O_tile = O_tile / l.unsqueeze(1)
            O[b, q_start:q_end, :] = O_tile
        
    return O
         

# correctness check
B, L, D = 2, 256, 64
Q = torch.randn(B, L, D).cuda()
K = torch.randn(B, L, D).cuda()
V = torch.randn(B, L, D).cuda()

out_ref = F.scaled_dot_product_attention(Q, K, V)
# print("out_ref", out_ref)
out_mine = flash_attn_v2(Q, K, V)
# print("out_mine", out_mine)
print(torch.allclose(out_ref, out_mine, atol=1e-5))

# test 1: different sequence lengths
B, L, D = 1, 512, 128
Q = torch.randn(B, L, D).cuda()
K = torch.randn(B, L, D).cuda()
V = torch.randn(B, L, D).cuda()
out_ref = F.scaled_dot_product_attention(Q, K, V)
out_mine = flash_attn_v2(Q, K, V, BLOCK_Q=64, BLOCK_KV=32)
print("different block sizes:", torch.allclose(out_ref, out_mine, atol=1e-5))

# test 2: L not divisible by block size
B, L, D = 2, 200, 64
Q = torch.randn(B, L, D).cuda()
K = torch.randn(B, L, D).cuda()
V = torch.randn(B, L, D).cuda()
out_ref = F.scaled_dot_product_attention(Q, K, V)
out_mine = flash_attn_v2(Q, K, V, BLOCK_Q=64, BLOCK_KV=64)
print("non-divisible L:", torch.allclose(out_ref, out_mine, atol=1e-5))

# test 3: batch size > 1
B, L, D = 4, 256, 64
Q = torch.randn(B, L, D).cuda()
K = torch.randn(B, L, D).cuda()
V = torch.randn(B, L, D).cuda()
out_ref = F.scaled_dot_product_attention(Q, K, V)
out_mine = flash_attn_v2(Q, K, V)
print("batch size 4:", torch.allclose(out_ref, out_mine, atol=1e-5))