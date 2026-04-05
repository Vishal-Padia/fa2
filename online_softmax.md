# Online Softmax

This is me trying to understand the online softmax algorithm.

## The Problem

We have a row of attention scores: `[s0, s1, s2, .....]` - one score per KV token. 

We want softmax:
```
p_i = exp(s_i) / sum(exp(s_j) for all j)
```

But in FA, we don't have all the scores at once. We see them in tiles. So we need to compute them incrementally.

## The Naive Incremental Attempt (Broken)

```
after tile 1: sum1: = exp(s0) + exp(s1)
after tile 2: sum2 = sum1 + exp(s2) + exp(s3)
```
This works mathematically, but `exp(s_i)` explodes to infinity if `s_i` is large (eg 100). So we subtract the max for numerical stability.

```
p_i = exp(s_i - m) / sum(exp(s_j - m))
```

where m = max of all scores. But you don't know the global max until you've seen all tiles. That's the problem.

## Online Softmax: The Fix

You maintain 3 running values per query row, updating them as each tile arrives.

```
m = max score seen so far
l = sum of exp(s_i - m) seen so far (the normalizer)
O = sum of exp(s_i - m) * v_i seen so far (unnormalized output)
```

When a new tile arrives with scores `s_new` and values `v_new`:
#### Step 1 - update the max
```
m_new = max(m, max(s_new))
```
#### Step 2 - compute rescale factor
```
exp_old = exp(m - m_new) # how much to shrink previous accumulations
```

This is always <= 1. If `m_new > m`, old values were computed with a smaller max, so they're too large and need to be rescaled down.

#### Step 3 - udpate l:
```
l_new = l * exp_old + sum(exp(s_new - m_new))
```
Rescale the old sum, add the new sum.

#### Step 4 - update O:
```
O_new = O * exp_old + exp(s_new - m_new) @ v_new
```
Rescale the old output, add new contribution.

After all tiles, we normalize once:
```
output = O / 1

At any point 'O / 1' gives you the correct softmax-weighted sum for all tiles seen so far. When 'm_new == m' (max didn't change), 'exp_old = 1' and it reduces to simple accumulation. When 'm_new > m' everything gets rescaled consistently.


Think of this as a running correction:
Tile 1:   m=3,  l=2.1,  O=...
             ↓ new tile arrives, max jumps to 5
             ↓ exp_old = exp(3-5) = 0.135  ← "old values were too big, scale them down"
Tile 2:   m=5,  l=0.28+1.8=2.08,  O=rescaled+new
             ↓ new tile arrives, max stays at 5
             ↓ exp_old = exp(5-5) = 1.0  ← "max didn't change, no correction needed"
Tile 3:   m=5,  l=2.08+0.9=2.98,  O=same+new
```

Final Output: `O / 1`, one division, done.