# KV Cache, Eviction Policies & Sliding Window Attention — Deep Notes
## From "Why Cache?" to H2O, StreamingLLM, and SWA

> Covers everything from first principles: what KV cache is and why it exists,
> the exact memory formula, why the cache becomes a bottleneck, and every strategy
> developed to manage or eliminate it — including H2O, StreamingLLM, and SWA.
> Includes backstory, math, worked examples, real-world usage, and research papers.

---

## Table of Contents

**Part 1 — Why KV Cache Exists**
1. [Autoregressive Generation — The Core Loop](#1-autoregressive-generation--the-core-loop)
2. [The Recomputation Problem](#2-the-recomputation-problem)
3. [KV Cache — The Fix](#3-kv-cache--the-fix)
4. [What Gets Cached and What Does Not](#4-what-gets-cached-and-what-does-not)
5. [Prefill vs Decode — Two Phases of Inference](#5-prefill-vs-decode--two-phases-of-inference)

**Part 2 — KV Cache Memory Formula**
6. [The Exact Memory Formula](#6-the-exact-memory-formula)
7. [Worked Example — LLaMA 2 7B](#7-worked-example--llama-2-7b)
8. [Worked Example — LLaMA 2 70B](#8-worked-example--llama-2-70b)
9. [How Each Variable Scales KV Cache Size](#9-how-each-variable-scales-kv-cache-size)
10. [KV Cache vs Model Weights — The Memory Race](#10-kv-cache-vs-model-weights--the-memory-race)
11. [Why KV Cache Makes Inference Memory-Bound](#11-why-kv-cache-makes-inference-memory-bound)

**Part 3 — Cache Eviction Policies**
12. [Why Eviction Is Needed — The Root Problem](#12-why-eviction-is-needed--the-root-problem)
13. [Naive Eviction Baselines](#13-naive-eviction-baselines)
14. [Attention Score Distribution — The Key Observation](#14-attention-score-distribution--the-key-observation)
15. [H2O — Heavy Hitter Oracle](#15-h2o--heavy-hitter-oracle)
16. [H2O Algorithm — How It Works](#16-h2o-algorithm--how-it-works)
17. [H2O Memory Math and Tradeoffs](#17-h2o-memory-math-and-tradeoffs)
18. [Attention Sinks — The Unexpected Discovery](#18-attention-sinks--the-unexpected-discovery)
19. [StreamingLLM — Sink Tokens + Recency Window](#19-streamingllm--sink-tokens--recency-window)
20. [StreamingLLM Algorithm and Fixed Cache Layout](#20-streamingllm-algorithm-and-fixed-cache-layout)
21. [StreamingLLM Memory Math and Use Cases](#21-streamingllm-memory-math-and-use-cases)
22. [H2O vs StreamingLLM — Comparison](#22-h2o-vs-streamingllm--comparison)
23. [Other Eviction Strategies](#23-other-eviction-strategies)

**Part 4 — Sliding Window Attention**
24. [The O(N²) Problem That Eviction Doesn't Solve](#24-the-on-problem-that-eviction-doesnt-solve)
25. [Sliding Window Attention — Core Idea](#25-sliding-window-attention--core-idea)
26. [SWA Memory and Compute Complexity](#26-swa-memory-and-compute-complexity)
27. [What SWA Can and Cannot Attend To](#27-what-swa-can-and-cannot-attend-to)
28. [Receptive Field Growth Through Layers](#28-receptive-field-growth-through-layers)
29. [Global Tokens — Mixing Local and Full Attention](#29-global-tokens--mixing-local-and-full-attention)
30. [SWA KV Cache — Fixed Memory Regardless of Context](#30-swa-kv-cache--fixed-memory-regardless-of-context)
31. [SWA vs Eviction — Different Problems, Different Solutions](#31-swa-vs-eviction--different-problems-different-solutions)

**Part 5 — Real-World Usage**
32. [Who Uses What — Modern Models](#32-who-uses-what--modern-models)
33. [Production Serving — vLLM and PagedAttention](#33-production-serving--vllm-and-pagedattention)
34. [Quantizing the KV Cache](#34-quantizing-the-kv-cache)

**Part 6 — Research Papers and Further Reading**
35. [Essential Papers with Summaries](#35-essential-papers-with-summaries)

**Part 7 — Reference**
36. [Formula Sheet](#36-formula-sheet)
37. [Glossary](#37-glossary)

---

# Part 1 — Why KV Cache Exists

---

## 1. Autoregressive Generation — The Core Loop

Language models generate text one token at a time. This is called **autoregressive generation** — each new token is conditioned on everything generated before it.

### The Generation Loop

```
Prompt: "The Eiffel Tower is located in"

Step 1: Input = ["The", "Eiffel", "Tower", "is", "located", "in"]
        Model runs full forward pass over 6 tokens
        Outputs probability over vocabulary
        Picks "Paris"

Step 2: Input = ["The", "Eiffel", "Tower", "is", "located", "in", "Paris"]
        Model runs full forward pass over 7 tokens
        Picks ","

Step 3: Input = ["The", ..., "Paris", ","]
        Full forward pass over 8 tokens
        Picks "France"

... continues until <EOS> token or max length reached
```

At each step, the model sees the entire sequence so far and produces one new token.

### Where Time Goes

During the forward pass of a Transformer, each attention layer computes:

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √d) × V
```

To compute this, the model needs Q, K, V for **every token in the current sequence**.

At step T (generating the T-th output token):
- Sequence has T + prompt_length tokens total
- Every attention layer must compute K and V for all T + prompt_length tokens
- Most of that computation is **identical to what was done at step T-1**

---

## 2. The Recomputation Problem

Let's make the redundancy concrete.

### Step-by-Step Redundancy

Say the prompt is 10 tokens, and we've generated 5 output tokens so far (step 5).

```
Step 5: Sequence = [p1, p2, ..., p10, g1, g2, g3, g4, g5]
        Compute K and V for all 15 tokens.

Step 6: Sequence = [p1, p2, ..., p10, g1, g2, g3, g4, g5, g6]
        Compute K and V for all 16 tokens.

The K, V for [p1, ..., p10, g1, ..., g5] are IDENTICAL to step 5.
We just computed them. We threw them away. We recompute them.
```

At step T, we recompute K, V for T-1 tokens that haven't changed.

### Cost Without Caching

```
Total K, V computations across L generation steps:
  Step 1:  compute K, V for  N_p + 1  tokens
  Step 2:  compute K, V for  N_p + 2  tokens
  ...
  Step L:  compute K, V for  N_p + L  tokens

Total = Σ_{t=1}^{L} (N_p + t) = L × N_p + L(L+1)/2

For N_p=1000, L=500:
  = 500 × 1000 + 500 × 501 / 2 = 500,000 + 125,250 = 625,250 computations

With caching:
  Prefill: N_p = 1000 computations (once)
  Decode:  L = 500 computations (one per step)
  Total:   1,500 computations

Speedup: 625,250 / 1,500 ≈ 417×
```

This is why KV cache exists. Without it, generation is quadratically expensive in generation length.

---

## 3. KV Cache — The Fix

The fix is simple: **compute K and V once for each token, store them, and reuse them at every future step.**

### How It Works

```
At each decoding step t:

  Compute Q, K, V for the ONE new token only
  Append new K, V to cache:
      cache_K[:, :, t, :] = new_k
      cache_V[:, :, t, :] = new_v

  Attend: Q_new over ALL cached K and V
      scores = Q_new × cache_K[:, :, :t+1, :]ᵀ   (attends to all past)
      output = softmax(scores) × cache_V[:, :, :t+1, :]
```

At each step:
- **Compute:** K, V for 1 new token → O(1) work per step
- **Load from cache:** K, V for all previous tokens → O(t) memory reads

The computation goes from O(t²) total to O(t) total for the decode phase.

### Memory Layout in GPU VRAM

```
KV Cache tensor shape:

  [num_layers, 2, batch_size, num_kv_heads, max_seq_len, head_dim]
   │           │   │           │              │             │
   │           │   │           │              │             per-head feature dimension
   │           │   │           │              allocated once for max length
   │           │   │           KV heads (< query heads if GQA)
   │           │   simultaneous sequences
   │           K=0 or V=1
   one entry per Transformer layer

Each layer maintains its own K and V cache.
Total memory = sum over layers.
```

---

## 4. What Gets Cached and What Does Not

This is a common point of confusion.

### Cached: Keys (K) and Values (V)

K and V vectors represent **what each token offers to others**:
- K: "what I can be found by" — used to match against future queries
- V: "what information I carry" — aggregated by future queries

These are **static per token** — once a token is processed, its K and V never change (assuming no re-encoding). They can be cached indefinitely.

### NOT Cached: Queries (Q)

Q represents "what the current token is looking for." At each decoding step, you're generating a new token, and only that new token's Q matters:

```
At step t, generating token t+1:
  Need Q for token t+1  → compute fresh (tiny: just one row)
  Need K, V for tokens 0...t → load from cache (large: all past tokens)

There is no reason to cache Q:
  - Old Q vectors were for old tokens looking at even older context
  - They are irrelevant when generating the next token
  - Each step needs only ONE query: the current new token's query
```

### NOT Cached: Attention Weights

The attention weight matrix P = softmax(Q × Kᵀ) is also not cached:
- At each step, P changes because Q is new and K grows by one row
- P must be recomputed at every step (but cheaply, since Q is just one row)
- FlashAttention computes P in SRAM tiles, never writing it to VRAM

### Summary Table

```
Tensor         Cache it?   Why
────────────────────────────────────────────────────────────────
K              YES         Fixed per token, needed by all future queries
V              YES         Fixed per token, aggregated by all future queries
Q              NO          Only the current new token's Q is needed
P (attn wts)   NO          Recomputed each step from Q (cheap: one row)
Output O       NO          Final output for each token is written to next layer
```

---

## 5. Prefill vs Decode — Two Phases of Inference

Inference has two very different phases with different bottlenecks.

### Phase 1: Prefill

**What it is:** Processing the input prompt — all tokens in parallel.

```
Prompt: 1000 tokens

Prefill:
  Process all 1000 tokens simultaneously (parallel like training)
  All Q, K, V computed at once via matrix operations
  Full N×N attention over the prompt
  Populate KV cache with K, V for all 1000 prompt tokens
```

**Characteristics:**
- Large batch of work → compute-bound (GPU is busy)
- Fast: uses full GPU parallelism
- Output: populated KV cache + first output token

### Phase 2: Decode

**What it is:** Generating new tokens one at a time.

```
Decode (one step):
  1. Take new token (one vector)
  2. Compute Q, K, V for just that one token
  3. Append K, V to cache
  4. Compute attention: Q (one row) × cache_K (all rows) → small matmul
  5. Get output token
  6. Repeat
```

**Characteristics:**
- One token at a time → tiny matmuls → GPU underutilized
- Memory-bound: dominant cost is LOADING the KV cache from VRAM
- Slow: lots of time waiting for VRAM reads, not computing

```
Decode bottleneck:

  Each step reads the entire KV cache from VRAM.
  For a 7B model, cache grows ~32 MB per token.
  At 1000 tokens, each step reads 32 GB of data (weights + cache).
  GPU compute could do this in 0.01 ms; memory latency takes 10 ms.

  Memory bandwidth is 99% of the problem during decode.
```

### Why Batch Size Helps Decode

With batch_size=B, you generate B tokens per step, but the KV cache read cost is amortized:

```
Batch=1:   Read 32 GB → produce 1 token  → 32 GB/token
Batch=32:  Read 32 GB → produce 32 tokens → 1 GB/token

→ 32× more memory-efficient use of bandwidth
→ 32× higher throughput (tokens/sec)
```

This is why serving systems try to maximize batch size — it directly determines inference efficiency.

---

# Part 2 — KV Cache Memory Formula

---

## 6. The Exact Memory Formula

```
KV_cache_bytes =
    2                ← K and V
  × n_layers         ← one cache per Transformer layer
  × n_kv_heads       ← number of KV heads (= n_heads for MHA, < n_heads for GQA)
  × seq_len          ← number of tokens currently cached
  × d_head           ← head dimension (= d_model / n_heads)
  × dtype_bytes      ← 4 for FP32, 2 for FP16/BF16, 1 for INT8
```

Written as a formula:

```
M_kv = 2 × L × H_kv × S × d × B_dtype

Where:
  L      = num_layers
  H_kv   = num_kv_heads
  S      = sequence length (tokens in cache)
  d      = head_dim = d_model / n_heads
  B_dtype = bytes per element (dtype-dependent)
```

### Breaking Down Each Term

**Factor of 2:** One tensor for K, one for V. Both have the same shape. Simple multiplicative factor.

**n_layers:** Every Transformer layer maintains its own independent KV cache. Layer 1's K, V are different projections from Layer 2's K, V. They cannot be shared. So total memory scales linearly with depth.

**n_kv_heads:** In MHA (Multi-Head Attention), n_kv_heads = n_heads. In GQA (Grouped Query Attention), n_kv_heads < n_heads, directly reducing cache. In MQA (Multi-Query Attention), n_kv_heads = 1.

**seq_len:** The number of tokens whose K and V vectors are stored. Grows as generation continues. This is the dynamic dimension — memory increases every step.

**d_head:** The dimension of each K or V vector. Each token contributes one d_head-dimensional vector to each KV head at each layer. Standard choice: d_head = d_model / n_heads.

**dtype_bytes:** The numerical precision of stored K, V tensors. FP16 is standard (2 bytes). Quantizing to INT8 (1 byte) halves cache size.

---

## 7. Worked Example — LLaMA 2 7B

LLaMA 2 7B model configuration:

```
d_model    = 4096
n_layers   = 32
n_heads    = 32   (query heads)
n_kv_heads = 32   (MHA — same as n_heads)
d_head     = d_model / n_heads = 4096 / 32 = 128
dtype      = FP16 → 2 bytes
```

### Cache at Various Sequence Lengths

```
Formula: M_kv = 2 × 32 × 32 × S × 128 × 2  bytes
              = 2 × 32 × 32 × S × 128 × 2
              = 524,288 × S  bytes
              = 0.5 MB × S  (approximately, S in tokens)

Actually:
  2 × 32 × 32 × 128 × 2 = 524,288 bytes per token = 512 KB per token

Sequence length   KV cache size   GPU with 80 GB (after weights ~14 GB)
───────────────────────────────────────────────────────────────────────
128 tokens        65.5 MB         trivial
1,024 tokens      512 MB          ~0.8% of GPU memory
4,096 tokens      2.05 GB         ~3% of GPU memory
16,384 tokens     8.19 GB         ~12% of GPU memory
32,768 tokens     16.4 GB         ~24% of GPU memory
131,072 tokens    65.5 GB         ~97% of GPU memory — leaves nothing!
```

At 128K context, the KV cache consumes nearly all VRAM on an 80 GB H100. The model weights (~14 GB) barely fit alongside it.

### Batch Size Impact

```
For batch_size = B (serving B users simultaneously):
  M_kv_total = B × M_kv_per_sequence

For batch=32, seq=4096:
  = 32 × 2.05 GB = 65.6 GB

That leaves only 80 - 14 (weights) - 65.6 (cache) = 0.4 GB for activations.
You need a second H100 just for the KV cache at this configuration.
```

---

## 8. Worked Example — LLaMA 2 70B

LLaMA 2 70B uses GQA with 8 KV heads instead of 64 (a 8× reduction):

```
d_model    = 8192
n_layers   = 80
n_heads    = 64   (query heads)
n_kv_heads = 8    (GQA — 8× fewer than query heads)
d_head     = 8192 / 64 = 128
dtype      = FP16 → 2 bytes
```

### Cache Calculation

```
M_kv = 2 × 80 × 8 × S × 128 × 2  bytes
     = 327,680 × S  bytes
     ≈ 0.32 MB per token

Sequence length   KV cache size   (with model weights ~140 GB, needs 2× H100)
───────────────────────────────────────────────────────────────────────────────
1,024 tokens      320 MB          manageable
4,096 tokens      1.28 GB         small compared to weights
16,384 tokens     5.12 GB         still manageable
65,536 tokens     20.5 GB         significant
131,072 tokens    40.9 GB         half a H100 just for cache
```

### MHA vs GQA Comparison (70B scale)

What would the KV cache look like if LLaMA 2 70B used MHA (64 KV heads) instead of GQA (8 KV heads)?

```
MHA (n_kv_heads=64): M_kv = 2 × 80 × 64 × S × 128 × 2 = 2,621,440 × S bytes ≈ 2.5 MB/token
GQA (n_kv_heads=8):  M_kv = 2 × 80 × 8  × S × 128 × 2 = 327,680  × S bytes ≈ 0.32 MB/token

GQA reduction: 2.5 MB / 0.32 MB = 8× smaller cache
```

At seq=4096:
```
MHA 70B: 2.5 × 4096 = 10.2 GB of KV cache
GQA 70B: 0.32 × 4096 = 1.28 GB of KV cache

8× reduction enables 8× larger batch sizes or 8× longer contexts.
```

This is exactly why Meta chose GQA for LLaMA 2 70B — the KV cache at 70B scale with full MHA would be unmanageable at any reasonable batch size.

---

## 9. How Each Variable Scales KV Cache Size

```
Variable          If you double it...    Cache size
───────────────────────────────────────────────────────
n_layers          2× more layers         2× larger
n_kv_heads        2× more KV heads       2× larger
seq_len           2× longer context      2× larger
d_head            2× larger heads        2× larger
batch_size        2× more users          2× larger (each user has own cache)
dtype_bytes       FP16→FP32 (2→4 bytes)  2× larger
dtype_bytes       FP16→INT8 (2→1 byte)   2× smaller

n_heads (query)   Doesn't affect cache   No change (if GQA already used)
d_model           Affects d_head if      Proportional change
                  n_heads fixed
```

The variables you can control to reduce cache:
1. **Reduce n_kv_heads** → use GQA or MQA (architecture choice, may affect quality)
2. **Reduce seq_len** → eviction policies or SWA (covered in Parts 3 and 4)
3. **Reduce dtype_bytes** → quantize the KV cache to INT8 or INT4
4. **Reduce batch_size** → fewer simultaneous users (hurts throughput)
5. **Reduce n_layers** → use a smaller model (hurts quality)

---

## 10. KV Cache vs Model Weights — The Memory Race

At small sequence lengths, model weights dominate memory. As context grows, KV cache overtakes them.

### Crossover Point

Model weights (once, fixed):

```
LLaMA 2 7B  → 14 GB  (FP16)
LLaMA 2 70B → 140 GB (FP16)
```

KV cache (grows with sequence × batch):

```
LLaMA 2 7B: 0.5 MB/token/user

Cache equals weights at:
  14 GB / (0.5 MB/token) = 28,000 tokens (for batch=1)
  
For batch=8:
  14 GB / (8 × 0.5 MB/token) = 3,500 tokens
  → At 3.5K context with 8 users, cache equals weights
```

```
LLaMA 2 70B: 0.32 MB/token/user

Cache equals weights at:
  140 GB / (0.32 MB/token) = 437,500 tokens (batch=1)
  140 GB / (8 × 0.32 MB/token) = 54,687 tokens (batch=8)
```

### The Practical Consequence

For interactive chatbots (short conversations, many users):
- Cache is small per user (few hundred tokens)
- Weights dominate memory
- Can serve many users per GPU

For long-form applications (code generation, document analysis):
- Cache grows to thousands of tokens per user
- Cache dominates weights at larger batches
- GPU memory exhausted; must limit batch or context

---

## 11. Why KV Cache Makes Inference Memory-Bound

Even though GPU has enormous compute (H100: 989 TFLOPs FP16), inference decoding is typically only 10–30% GPU utilization.

### The Bandwidth Constraint

At each decode step, the GPU must:

```
1. Load all model weights from VRAM          ~14 GB  (for 7B model)
2. Load entire KV cache from VRAM            grows with context
3. Compute Q for 1 new token                 trivially fast
4. Compute attention over cached K, V        fast — tiny matmul
5. Run FFN layers                            dominated by weight loads

Total VRAM reads per step ≈ model_weights + kv_cache_size
```

For a 7B model at 1000-token context:

```
Reads per step: 14 GB (weights) + 0.5 GB (cache) = 14.5 GB
H100 bandwidth: 3.35 TB/s

Time to read:   14.5 GB / 3350 GB/s ≈ 4.3 ms

Compute for one token:  ~0.1 ms (trivially fast)

→ GPU is idle 97% of the time, waiting for memory.
→ Increasing FLOPs won't help. Only bandwidth or less memory movement helps.
```

This is why KV cache management is the central problem of LLM inference at scale — not arithmetic, not model quality, but **where K and V vectors live and how many of them you have to read**.

---

# Part 3 — Cache Eviction Policies

---

## 12. Why Eviction Is Needed — The Root Problem

KV cache grows unboundedly as sequences get longer. Two fundamental problems:

### Problem 1: Fixed GPU Memory

```
80 GB H100:
  Model weights:   14 GB  (7B, FP16)
  Available for KV: 66 GB
  Cache per token: 0.5 MB
  Max tokens:      66 GB / 0.5 MB = 132,000 tokens (batch=1)
  
  With batch=32: 66 GB / (32 × 0.5 MB) = 4,125 tokens per user

At batch=32, every user is limited to ~4K tokens.
Longer contexts? Out of memory.
```

### Problem 2: Bandwidth Grows with Cache

Even if memory is sufficient, loading a larger cache takes longer:

```
Step 1000 with full cache:
  Load 1000 × 0.5 MB = 500 MB cache per step
  At 3.35 TB/s: 0.15 ms just for cache reads

Step 10000 with full cache:
  Load 10000 × 0.5 MB = 5 GB cache per step
  At 3.35 TB/s: 1.5 ms just for cache reads

→ Decode latency grows linearly with context length.
→ For a 100K context, each token takes ~15 ms just for cache reads.
```

**Eviction:** Instead of keeping all past K, V vectors, keep only the most important ones and discard the rest. The key question is: **which tokens' K, V are safe to throw away?**

---

## 13. Naive Eviction Baselines

Before discussing sophisticated methods, understand the obvious baselines and why they fail.

### FIFO (First In, First Out) — The Most Obvious

Discard the oldest tokens when the cache is full.

```
Budget: 1000 tokens
At token 1001: discard token 1

Cache always holds: [t-999, t-998, ..., t-1, t]  (most recent 1000)
```

**Problem:** Destroys long-range context.

```
Document: "Alice founded the company in 1998. [900 more tokens...] Alice retired in 2023."

Processing "Alice retired in 2023":
  FIFO has evicted token 0–100 (including "Alice founded the company in 1998")
  Model has no memory of who Alice is or what company
  → Incoherent output for long documents
```

### Random Eviction

Randomly select tokens to evict.

**Problem:** Equal chance of discarding a critical token vs. a filler word. Quality is unpredictable and poorly correlates with content importance.

### LRU (Least Recently Used) — Computer Science Classic

Evict the token that was "used" (attended to) least recently.

**Problem:** Attention patterns in LLMs don't match LRU assumptions:
- Early tokens (especially the very first token) are attended to at every step, so they'd never be evicted — but late-middle tokens might be critical for the next few steps even if not attended to for a while

The fundamental issue: **which tokens are important depends on the future query, which you don't know yet.**

---

## 14. Attention Score Distribution — The Key Observation

All sophisticated eviction methods exploit the same empirical observation about how attention weights are distributed.

### The Heavy Hitter Phenomenon

Researchers analyzing attention weights across many LLM layers found:

```
For a sequence of N tokens, the attention weights at step t are NOT uniform.

Typical distribution:
  ~5–10% of tokens receive ~80–90% of the total attention weight.
  The remaining ~90–95% of tokens receive only ~10–20% of attention.
```

This means:

```
Most tokens contribute almost nothing to the output.
A small subset of "heavy hitter" tokens dominate.

If you could keep only those heavy hitters:
  → Same output (approximately)
  → Much smaller cache

The question is: which tokens will be heavy hitters for future queries?
```

### The Key Pattern

Heavy hitters tend to be:
1. **The very first token(s)** — nearly every layer attends to them heavily (explained in Section 18 — "attention sinks")
2. **Recently generated tokens** — last few tokens provide immediate context
3. **Semantically important tokens** — subject nouns, rare words, key entities (task-dependent)

This observation motivated both H2O and StreamingLLM.

---

## 15. H2O — Heavy Hitter Oracle

### Backstory

In 2023, a team at UT Austin and other institutions published "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models." The paper asked a simple question:

**Can we identify which tokens are the "heavy hitters" (receive the most attention) and keep only those in the cache?**

The insight: across multiple generation steps, **the same tokens tend to receive high attention repeatedly.** Token importance is somewhat persistent across time. If a token was a heavy hitter at step 5, it's likely still important at step 10.

This means you don't need to know future queries — you can use **past attention scores** as a proxy for future importance.

---

## 16. H2O Algorithm — How It Works

### Core Algorithm

```
Parameters:
  budget = K    (max tokens to keep in cache)
  heavy_hitters = K/2  (allocate half budget to heavy hitters)
  recent = K/2         (allocate other half to recent tokens)

For each token t during generation:

  Step 1: Compute Q_t, K_t, V_t for the new token
  Step 2: Compute attention scores: s_t = Q_t × cache_K^T  [1 × cache_size]
  Step 3: Add s_t to accumulated scores:
          acc_score[i] += s_t[i]  for all i in cache
          (accumulate WHICH tokens received attention over all past steps)
  Step 4: Add new token to cache
  Step 5: If cache_size > budget:
          → Keep top-(K/2) tokens by accumulated score (heavy hitters)
          → Keep the K/2 most recent tokens (recency)
          → Evict everything else
```

### The Greedy Heavy Hitter Score

```
accumulated_score[i] = Σ_{t=1}^{T} attention_weight[t][i]

This is the total attention a token has received across all past steps.

High score → attended to many times → likely important for future
Low score  → rarely attended to    → safe to evict
```

### Worked Example

```
Budget: 6 tokens (3 heavy hitters + 3 recent)

Step 10, sequence: [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]

Accumulated attention scores:
  t1: 0.45  ← very high (attention sink)
  t2: 0.02  ← low
  t3: 0.18  ← moderate
  t4: 0.05  ← low
  t5: 0.31  ← high (key entity was here)
  t6: 0.03  ← low
  t7: 0.08  ← low
  t8: 0.15  ← moderate
  t9: 0.12  ← recency
  t10: —    ← just generated (always kept as part of recency)

Cache after eviction:
  Heavy hitters (top 3 by score): t1 (0.45), t5 (0.31), t3 (0.18)
  Recent tokens (last 3):         t8, t9, t10

Evicted: t2, t4, t6, t7
```

### The Score Accumulation Insight

Why accumulate scores instead of using the current step's scores only?

```
Current step scores might be noisy:
  At step 100, current Q asks about "when"
  → date tokens score high, entity tokens score low
  
  At step 101, current Q asks about "who"
  → entity tokens score high, date tokens score low

Using only current scores: evict entity tokens at step 100, date tokens at step 101.
Thrashing — evicting something important then regretting it.

Accumulated scores:
  Shows which tokens are important ON AVERAGE across all past queries
  More stable: important tokens stay important across different query types
```

---

## 17. H2O Memory Math and Tradeoffs

### Memory with H2O

```
Without H2O:
  Cache at token T = 2 × L × H_kv × T × d × bytes  (grows with T)

With H2O (budget K):
  Cache is FIXED at K tokens regardless of T
  Memory = 2 × L × H_kv × K × d × bytes  (constant!)

For 7B model, K=1000:
  Fixed: 2 × 32 × 32 × 1000 × 128 × 2 = 512 MB always
  No matter if sequence is 1000 or 100,000 tokens long.
```

### The Quality Tradeoff

```
Budget K = N (full cache):         Perfect quality, full memory
Budget K = N/2:                    ~97% quality, half memory
Budget K = N/5:                    ~90–95% quality, 5× memory savings
Budget K = N/10:                   ~85–90% quality, 10× memory savings
Budget K = N/20:                   Noticeable degradation

(Quality percentages are rough benchmarks from the paper; task-dependent)
```

### What H2O Doesn't Handle Well

**Positional encoding issues:** When tokens are evicted from the middle of a sequence, the remaining tokens have non-contiguous positions. KV cache stores K, V vectors that were computed with specific position embeddings.

```
Original:  pos [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
After evicting pos 3, 4, 7:
           pos [0, 1, 2, 5, 6, 8, 9]    ← gaps in position indices
```

For absolute position encodings (old style): the model sees odd position patterns it wasn't trained on.

For RoPE (Rotary Position Embedding — used in LLaMA):
- KV vectors are rotated by their position during computation
- The rotation is "baked in" when the vector is created
- Evicting and rearranging doesn't change the baked-in position
- So RoPE is relatively robust to eviction (a good property)

**Cannot recover evicted tokens:** Once evicted, that K, V is gone. If the model later needs information from that token, it can't get it. H2O is a gamble that important tokens will have high accumulated attention.

---

## 18. Attention Sinks — The Unexpected Discovery

### The Discovery

While building StreamingLLM, researchers at MIT (Xiao et al., 2023) made a surprising finding when studying attention patterns in LLMs:

```
The first 1–4 tokens in ANY sequence receive disproportionately high
attention weight — often 20–50% of total weight — at EVERY layer.

This happens regardless of:
  - What those first tokens are (they could be "The", "I", punctuation)
  - What the current query is about
  - How long the sequence is

These tokens act as "attention sinks" — places where attention drains
even when the content is not relevant.
```

### Why Attention Sinks Exist

The explanation is subtle but important:

Softmax is an all-or-nothing normalization — attention weights must sum to 1. Sometimes, no key in the sequence is particularly relevant to the current query. But softmax can't express "no strong match" — it must distribute weight somewhere.

The model learns to dump excess attention weight onto "safe" tokens — typically the first token (often a BOS token like `<s>`) or common initial tokens. These become the "garbage can" where unneeded attention weight is deposited.

```
Query at step t: very specific factual question
If no token has high similarity:
  softmax([−0.1, −0.2, −0.15, ...]) ≈ [0.15, 0.13, 0.14, ...]
  All attention is spread, but the model learned to push surplus to position 0

→ Position 0 gets 0.4 weight even when its content is irrelevant.
```

### Why This Matters for Eviction

```
FIFO: Would evict the first tokens (oldest = first)
       → Destroys attention sinks
       → Sudden performance collapse (model can't dump attention anywhere "safe")
       → NOT just a content loss — it's a structural breakdown

H2O: Keeps top-K by accumulated score
       → First tokens have highest accumulated scores (always attended to)
       → H2O automatically protects attention sinks
       → More principled than FIFO, but doesn't explicitly model sink structure
```

### Empirical Confirmation

The StreamingLLM paper showed a striking result:

```
Evict all tokens EXCEPT the first 4 (sinks) + last W (recent):
  → Performance nearly as good as full attention!

Evict the first 4 (sinks) + keep everything else:
  → Sudden catastrophic performance collapse!

The first few tokens are structurally critical.
They are not important for their content — they are important as
attention "anchors" that the softmax mechanism relies on.
```

---

## 19. StreamingLLM — Sink Tokens + Recency Window

### Backstory

"Efficient Streaming Language Models with Attention Sinks" (Xiao et al., MIT/Meta, 2023) was motivated by a specific use case: **streaming inference on potentially infinite sequences.**

The challenge they set:

> Can we run an LLM on arbitrarily long sequences with a fixed memory footprint — without restarting or losing coherence — while maintaining close to full-attention quality?

The answer they found: **yes, by keeping sink tokens + a recency window.**

### The StreamingLLM Policy

```
StreamingLLM cache = [sink tokens] + [recent tokens]

Parameters:
  n_sink    = 4   (number of initial tokens to ALWAYS keep)
  window    = W   (number of most recent tokens to keep)

Total cache size = n_sink + W  (FIXED, regardless of total sequence length)

At each new token t:
  - If t is within first n_sink: always keep (it's a sink)
  - Otherwise:
    - Add to recency window
    - If window is full: evict the OLDEST token in the window (FIFO within window)
    - Sinks are never evicted
```

### Visual Representation

```
Full sequence:     [s1, s2, s3, s4, ..., tokens 5..10000..., t-W, t-W+1, ..., t-1, t]
StreamingLLM sees: [s1, s2, s3, s4,                          t-W, t-W+1, ..., t-1, t]
                    ── sinks ──                                ────── recency window ──

Middle tokens (5 to t-W-1) are EVICTED.
Sinks are ALWAYS kept (never evicted).
Recent W tokens are kept as a sliding window.
```

### Why This Works

```
The two components serve different purposes:

Sink tokens (first 4):
  Preserve the structural "garbage collector" role
  Prevent attention weight from having nowhere safe to go
  Without them: softmax becomes unstable, output degrades catastrophically

Recent window (last W tokens):
  Provides local context (the most recent W tokens)
  For most language tasks, the immediately preceding text
  is the most relevant context
  Window of 1000–4000 usually captures enough local coherence
```

### What StreamingLLM Cannot Do

It **cannot** answer questions that require long-range retrieval beyond the window:

```
Token 0:   "The code repository uses MIT license."
Tokens 1-5000: (other content)
Token 5001: "What license does this use?"

If window=1000:
  Token 0 is a sink → kept
  Tokens 1-4001 → evicted (outside window)
  Tokens 4002-5000 → in window → kept

In this case, it works! Token 0 (the relevant fact) is a sink and is kept.

But:
Token 0:   "The code repository uses MIT license."
Token 500: "The company was founded by Sarah Chen."
Token 5001: "Who founded the company?"

Token 500 ("founded by Sarah Chen") is NOT a sink.
If window=1000: tokens 4001-5000 in window.
Token 500 is EVICTED.
Model cannot answer correctly.

StreamingLLM has no explicit notion of "semantic importance" —
it only keeps sinks + recency, nothing in between.
```

---

## 20. StreamingLLM Algorithm and Fixed Cache Layout

### The Position Embedding Challenge

StreamingLLM has a subtle but critical implementation detail around position embeddings.

**The problem:**

```
Original sequence positions:
  s1(pos=0), s2(pos=1), s3(pos=2), s4(pos=3), [evicted...], t-W(pos=9500), ..., t(pos=10000)

After eviction and packing:
  s1, s2, s3, s4, t-W, ..., t   (in cache, contiguous)

What position do we assign to t-W when computing Q×K^T?
```

If we use the original position (9500), then the relative distance between s4 (pos=3) and t-W (pos=9500) is 9497. The model was trained with continuous position sequences, not with gaps like this.

**StreamingLLM's fix: Re-index positions**

Don't use original positions. Assign fresh contiguous positions to whatever is in the cache:

```
Cache:      [s1, s2, s3, s4, t-W, t-W+1, ..., t]
Re-indexed: [pos=0, pos=1, pos=2, pos=3, pos=4, pos=5, ..., pos=n_sink+W]
```

This only works cleanly with **RoPE (Rotary Position Embedding)** because RoPE encodes *relative* distances, not absolute positions. Re-indexing is equivalent to saying "pretend the evicted tokens don't exist."

For other position encodings (ALiBi, absolute sinusoidal), the fix is more complex.

### Pseudo-code

```
Initialize:
  sink_cache_K, sink_cache_V = [None] × n_sink   (sink token KVs)
  window_K = deque(maxlen=W)   (recent window, FIFO eviction)
  window_V = deque(maxlen=W)

For each new token t (content x_t):

  Compute:
    k_t, v_t = compute_kv(x_t)

  Update cache:
    if t < n_sink:
      sink_cache_K[t] = k_t
      sink_cache_V[t] = v_t
    else:
      window_K.append(k_t)   # if full, automatically evicts oldest
      window_V.append(v_t)   # deque(maxlen=W) handles this

  Build attention inputs:
    full_K = concat(sink_cache_K, list(window_K))  # [n_sink + min(t, W)] tokens
    full_V = concat(sink_cache_V, list(window_V))

  Re-index positions:
    positions = [0, 1, ..., n_sink-1,             # sink positions (fixed)
                 n_sink, n_sink+1, ..., n_sink+len(window_K)-1]  # window positions

  Attend:
    Q_t with re-indexed full_K, full_V
    → get output token
```

---

## 21. StreamingLLM Memory Math and Use Cases

### Fixed Memory Budget

```
StreamingLLM cache = n_sink + W  tokens (always, regardless of seq length)

For 7B model, n_sink=4, W=1000:
  = 2 × 32 × 32 × (4 + 1000) × 128 × 2 bytes
  = 2 × 32 × 32 × 1004 × 128 × 2
  ≈ 515 MB

For 70B model, n_sink=4, W=1000:
  = 2 × 80 × 8 × 1004 × 128 × 2
  ≈ 328 MB

This is CONSTANT — the same at token 1 and token 1,000,000.
```

### Enabled Use Cases

**Infinite streaming:** Process a never-ending stream of text (live transcription, continuous monitoring) without memory growing unboundedly.

**Very long documents:** Process a 100K-word document token by token with fixed VRAM.

**Perpetual conversation:** A chatbot that runs forever without hitting memory limits (though it forgets old conversation content outside the window).

**Comparison:**

```
Strategy        Memory              Quality (long seq)
──────────────────────────────────────────────────────
Full KV cache   O(N) — grows       Perfect
H2O             O(K) — fixed       Good (importance-based)
StreamingLLM    O(sink + W) — fixed OK (local + sink only)
SWA             O(W) — fixed       OK (local only)
```

---

## 22. H2O vs StreamingLLM — Comparison

| Property | H2O | StreamingLLM |
|---|---|---|
| **Core policy** | Keep top-K by accumulated attention score | Keep n_sink + W most recent |
| **Memory** | O(K) fixed | O(n_sink + W) fixed |
| **Long-range retrieval** | Yes (if important tokens scored high) | Only if in sink or window |
| **Implementation complexity** | Higher (must track accumulated scores) | Low (just a deque) |
| **Position embedding handling** | Difficult (gaps in positions) | Clean (re-index sinks + window) |
| **Understands attention sinks** | Implicitly (they score highest) | Explicitly (hardcoded n_sink) |
| **Suitable for streaming** | Partially (positions tricky) | Yes (designed for this) |
| **Task dependency** | High (important tokens vary by task) | Low (sinks + recency are universal) |
| **Quality guarantee** | Better for semantic tasks | Better for fluency/local coherence |

### When to Use Which

```
Use H2O when:
  - You need to answer questions about content anywhere in a long document
  - You can accept higher implementation complexity
  - RoPE-based model (handles position gaps better)
  - You want importance-based selection

Use StreamingLLM when:
  - Infinite streaming is the goal
  - Simple, predictable memory budget is critical
  - Local coherence is sufficient (conversation, transcription)
  - Minimal implementation complexity is valued
```

---

## 23. Other Eviction Strategies

### ScissorHands

Uses the observation that attention patterns are locally consistent: the attention weights for nearby queries are similar to the current query's weights. Instead of accumulating all past scores, uses a **pivot token** strategy to find heavy hitters via clustering.

Key advantage over H2O: lower overhead (doesn't need to track accumulated scores for all tokens).

### SnapKV

Evicts at the **prefill** stage rather than during decode. Analyzes attention patterns during prompt processing to decide which prompt tokens' KV to evict before decoding even begins.

```
Prefill 10,000 token document:
  SnapKV analyzes attention during prefill
  Keeps top-K important KV vectors
  Decode starts with already-compressed cache

Advantage: no eviction overhead during decode (already done)
Disadvantage: can't adapt to what the model focuses on during decode
```

### PyramidKV

Empirical finding: different layers have different optimal KV cache sizes. Lower layers need more tokens (they handle local syntactic patterns); upper layers need fewer (they handle high-level semantics, which is already sparse).

```
Layer distribution:
  Layers 1-10 (bottom): full cache or large K
  Layers 11-25 (middle): medium K
  Layers 26-32 (top): small K

Total memory reduced without uniform eviction across all layers.
```

### Quest

A training-free approximate attention method that selects which KV "pages" to load based on query-key similarity. Doesn't evict (still stores everything) but only LOADS the relevant pages per step — reducing bandwidth even when memory is available.

```
Cache: store all K, V in VRAM
Step t: instead of loading ALL cached K, V:
         estimate which pages are relevant using cheap approximation
         load only relevant pages (~10–20% of cache)
         compute exact attention over those pages

→ Approximate: may miss some relevant tokens
→ Faster: 5–10× less bandwidth per step
```

---

# Part 4 — Sliding Window Attention

---

## 24. The O(N²) Problem That Eviction Doesn't Solve

Eviction policies (H2O, StreamingLLM) fix **memory** for the KV cache but don't change the **attention computation architecture**. Even with eviction:

```
At decode step: attention between 1 new query and K cached tokens = O(K) per step
This is fine — decode is already O(K) per step with a full cache.

The problem eviction solves: memory cost of storing K tokens.
The problem eviction does NOT solve: prefill cost.

Prefill over N tokens: O(N²) attention — all N tokens attend to all previous.
Eviction doesn't help prefill because eviction only applies during decode.

For N=1,000,000 tokens:
  Prefill FLOPs: 4 × 10^12 × d_model × N²  (astronomical)
  Cannot evict during prefill — you haven't generated the cache yet.
```

Sliding Window Attention is an **architectural change** that makes the attention computation itself O(N × W) instead of O(N²) — not a cache management trick, but a different way of computing attention.

---

## 25. Sliding Window Attention — Core Idea

### Standard Attention Recap

```
Token i can attend to: tokens 0, 1, 2, ..., i-1, i   (all past tokens)

Attention matrix (causal):
             token 0   1   2   3   4   5
token 0    [  1    0   0   0   0   0  ]
token 1    [  1    1   0   0   0   0  ]
token 2    [  1    1   1   0   0   0  ]
token 3    [  1    1   1   1   0   0  ]
token 4    [  1    1   1   1   1   0  ]
token 5    [  1    1   1   1   1   1  ]

N×N lower-triangular matrix.  Memory: O(N²).
```

### Sliding Window Attention

```
Token i can attend to: tokens max(0, i-W), ..., i-1, i  (only the last W tokens)

With W = 3:
             token 0   1   2   3   4   5
token 0    [  1    0   0   0   0   0  ]
token 1    [  1    1   0   0   0   0  ]
token 2    [  1    1   1   0   0   0  ]
token 3    [  0    1   1   1   0   0  ]  ← token 0 no longer in window
token 4    [  0    0   1   1   1   0  ]  ← only 3 tokens
token 5    [  0    0   0   1   1   1  ]

Each row has at most W non-zero entries.
Total non-zero entries: O(N × W).
Memory: O(N × W).
```

For N=1,000,000, W=4096:

```
Standard attention: N² = 10^12 values   (1 trillion — impossible)
SWA:               N × W = 4 × 10^9    (4 billion — manageable)

Reduction: N/W = 1,000,000 / 4096 ≈ 244×
```

---

## 26. SWA Memory and Compute Complexity

### Memory (Training / Prefill)

```
Standard attention:    O(N²)      — attention matrix size
SWA:                  O(N × W)   — N rows × W entries each

For N=100K, W=4096:
  Standard: 10^10 values × 2 bytes = 20 GB  (per head, per layer!)
  SWA:      100K × 4096 × 2 bytes = 800 MB  (per head — 25× smaller)
```

### Compute (FLOPs)

```
Standard attention FLOPs per layer: 4 × N² × d_model
SWA FLOPs per layer:                4 × N × W × d_model

Compute reduction: N/W times fewer FLOPs
```

### KV Cache During Decode (Fixed Size)

```
Standard KV cache during decode: grows as O(N) with each new token
SWA KV cache during decode:      FIXED at W tokens

At each decode step:
  Standard: Q attends to all N past tokens
  SWA:      Q attends only to the last W past tokens → evict oldest, add newest
  Cache size: always exactly W (or n_sink + W with StreamingLLM-style sinks)
```

SWA naturally limits KV cache size during decode — you never need to store more than W keys and values.

---

## 27. What SWA Can and Cannot Attend To

### The Locality Assumption

SWA works well when:

```
The information needed to process token i is usually within W tokens of i.

Examples where this holds:
  Sentence parsing:     "The cat [that chased] the mouse was tired."
                        → local syntactic context
  Code completion:      Current function body is usually < 4096 tokens
  Conversation turn:    The immediate exchange provides context
  Named entity:         "Paris is..." → "Paris" is nearby
```

SWA fails when:

```
The information needed is far back in the sequence.

Examples:
  Legal document: "Section 1 defines X. [50 pages later] As defined above, X..."
  → "defined above" refers to Section 1, outside any reasonable window

  Novel summary: "The protagonist [from chapter 1] returns [in chapter 20]"
  → Without global tokens or full attention, model has forgotten chapter 1

  Very long-range dependency:
  "The author of Hamlet is Shakespeare." [100K tokens of play text]
  "Who wrote this play?" → "Shakespeare" is 100K tokens ago, outside window.
```

---

## 28. Receptive Field Growth Through Layers

A key property: even though each SWA layer only sees W tokens, information from farther away reaches the model through **stacking layers**.

### How Information Propagates

```
Each layer: each token aggregates information from its W-token window.

After layer 1: each token contains information from tokens ±W away.
After layer 2: each token contains information from tokens ±2W away.
After layer k: each token contains information from tokens ±k×W away.

For L layers of SWA with window W:
  Effective receptive field = L × W tokens
```

### Worked Example

```
Mistral 7B:
  W = 4096 tokens (sliding window)
  L = 32 layers

  Effective receptive field = 32 × 4096 = 131,072 tokens ≈ 128K

So even though each layer only sees 4096 tokens at a time,
information from 128K tokens back can theoretically reach the output
through 32 hops of attention.
```

**The caveat:** Information degrades across hops. Something 128K tokens away contributes weakly through 32 layers of aggregation. It's not the same as attending directly to that token. But for many tasks, this indirect path is sufficient.

---

## 29. Global Tokens — Mixing Local and Full Attention

Pure SWA has the hard limitation: no direct long-range access. Many practical systems add **global tokens** that attend to (and are attended to by) the entire sequence.

### Two Types of Global Tokens

**Designated global tokens:** Specific positions in the input are marked as "global" — they participate in full attention, while others use the window.

```
[GLOBAL] Doc: The contract expires on December 31st. [regular tokens...] When does it expire?

The [GLOBAL] token:
  → attends to ALL tokens in the document
  → is attended to by ALL tokens
  → acts as a "memory bus" carrying long-range information

Longformer (Allen AI):
  Selects certain tokens (task-specific, like [CLS]) as global.
  Others use local window attention.
  Cost: O(N × W + N × G) where G = number of global tokens.
```

**Interleaved full attention layers:** Some layers use SWA, others use full attention.

```
Mistral 7B approach:
  Odd layers:  Sliding Window Attention (W=4096)
  Even layers: Full attention
  
  Full attention layers provide long-range connections.
  SWA layers provide efficient local processing.
  Alternation gives a balance of efficiency and expressiveness.
```

### BigBird Strategy

BigBird (Google, 2020) uses three types of attention simultaneously:

```
1. Local window (W=3): Each token attends to 3 nearest neighbors
2. Global tokens (G=some special tokens): Full attention to/from these
3. Random tokens (R=2): Each token attends to 2 random other tokens

Total per token: W + G + R  (constant regardless of N)
This is provably expressive enough for many NLP tasks (theoretically equivalent
to full attention under certain conditions).
```

---

## 30. SWA KV Cache — Fixed Memory Regardless of Context

### During Decode

At each decode step with SWA:

```
Current token t: only needs to attend to tokens [t-W, t-W+1, ..., t-1, t]
                 → only needs K, V for those W tokens in cache

When token t+1 arrives:
  → Remove K, V of token t-W from cache (outside window now)
  → Add K, V of token t to cache (just computed)
  → Cache still holds exactly W tokens

Cache size NEVER grows beyond W tokens.
```

### KV Cache Formula for SWA

```
SWA KV cache = 2 × n_layers × n_kv_heads × W × d_head × dtype_bytes

Where W is the window size (fixed, not growing).

For Mistral 7B (W=4096, n_layers=32, n_kv_heads=8, d_head=128, BF16):
  = 2 × 32 × 8 × 4096 × 128 × 2 bytes
  = 536,870,912 bytes
  ≈ 512 MB  (FIXED, regardless of total context length)

This 512 MB is the same whether the total sequence is:
  4,097 tokens → 1 token outside window, cache holds 4096
  100,000 tokens → 95,904 outside window, cache holds 4096
  1,000,000 tokens → cache holds 4096
```

### Comparison: Memory at 100K Tokens

```
Strategy                 KV Cache at 100K tokens   Comment
─────────────────────────────────────────────────────────────────────
Full KV (Mistral 7B)     ~25.6 GB                  Grows with seq len
SWA (Mistral, W=4096)    ~512 MB (FIXED)            50× less than full
StreamingLLM (W=1000)    ~128 MB (FIXED)            200× less than full
H2O (K=4096)             ~512 MB (FIXED)            Same as SWA in bytes
                                                     but importance-selected
```

---

## 31. SWA vs Eviction — Different Problems, Different Solutions

These are often confused. They are fundamentally different approaches.

### KV Cache Eviction (H2O, StreamingLLM)

```
Architecture: UNCHANGED — full attention mathematically
Where applied: During decode, at cache management level
What it changes: Which cached K, V vectors to STORE
Full attention over cached tokens: YES (over whatever is in cache)
Prefill cost: UNCHANGED — still O(N²) for the prompt

Good for: Post-training deployment, no architecture change needed
          Adapting existing models without retraining
Bad for:  Prefill over very long prompts (still O(N²))
```

### Sliding Window Attention

```
Architecture: CHANGED — different attention pattern
Where applied: Inside the attention kernel
What it changes: The COMPUTATION of attention (not just storage)
Full attention over window: YES, but only the window
Prefill cost: O(N × W) — much cheaper

Good for: Training new models for long contexts
          Very long sequence efficiency at both prefill and decode
Bad for:  Retrofitting existing models (requires retraining)
          Tasks requiring long-range attention beyond the window
```

### When to Use What

```
You have an existing model and want longer context → Eviction (H2O, StreamingLLM)
You are training a new model for long contexts → SWA (Mistral-style)
You need both training efficiency and long context → SWA + global tokens (Longformer)
You have infinite streaming use case → StreamingLLM
You need semantic importance selection → H2O
You just want simplicity → StreamingLLM
```

---

# Part 5 — Real-World Usage

---

## 32. Who Uses What — Modern Models

### Attention Type and KV Cache Strategy

| Model | Attention | n_kv_heads | Window W | Eviction | Notes |
|---|---|---|---|---|---|
| BERT-base | MHA (Encoder) | 12 | Full | None | No decode KV cache (encoder) |
| GPT-2 | MHA | 25 | Full | None | |
| GPT-3 175B | MHA | 96 | Full | None | |
| PaLM 540B | MQA | 1 | Full | None | MQA = tiny KV cache |
| LLaMA 1 7B | MHA | 32 | Full | None | |
| LLaMA 2 7B | GQA | 8 | Full | None | 4× KV reduction vs MHA |
| LLaMA 2 70B | GQA | 8 | Full | None | 8× KV reduction vs MHA |
| LLaMA 3 8B | GQA | 8 | Full | None | |
| LLaMA 3 70B | GQA | 8 | Full | None | |
| LLaMA 3.1 405B | GQA | 8 | Full | None | Supports 128K context |
| Mistral 7B | GQA | 8 | 4096 (SWA) | None | Alternating SWA/full layers |
| Mixtral 8×7B | GQA | 8 | 4096 (SWA) | None | MoE + SWA |
| Mistral Large | GQA | — | Partial SWA | None | |
| Gemma 7B | MQA | 1 | Full | None | |
| Gemma 2 9B | GQA | 4 | Partial SWA | None | Alternating local/global |
| Phi-3 | GQA | 8 | Full | None | |
| Falcon 40B | MQA | 8 | Full | None | Uses multi-query |
| DeepSeek-V2 | MLA | — | Full | None | Latent KV compression |
| DeepSeek-V3 | MLA | — | Full | None | |
| Qwen 2.5 72B | GQA | 8 | Full | None | |
| Command R+ | GQA | — | Full | None | |
| Longformer | MHA | Full | Local+Global | None | Research/encoders |
| BigBird | MHA | Full | Local+Random+Global | None | Research |

**Key trends:**
- **GQA with 8 KV heads** is the dominant choice for production decoders — reduces KV cache by 4–8× vs MHA with minimal quality loss
- **SWA** used in Mistral family but not mainstream for all models — training with SWA required, can't simply add at inference
- **Eviction** applied at serving time by inference frameworks, transparent to the model itself

---

## 33. Production Serving — vLLM and PagedAttention

### The Fragmentation Problem Without PagedAttention

Traditional KV cache allocation:

```
Request A: might generate up to 2000 tokens → allocate 2000 × cache_per_token in VRAM (contiguous)
Request B: might generate up to 2000 tokens → allocate another 2000 block (contiguous)

Result:
  If A generates only 300 tokens: 1700 tokens worth of VRAM wasted
  If B generates 2000 tokens: fine
  If C arrives: no contiguous block available even though combined free memory exists
  → External fragmentation: lots of free memory, can't use it
```

### PagedAttention Solution

```
KV cache divided into "pages" (blocks) of fixed size (e.g., 16 tokens)

Request A generates 300 tokens:
  Allocates 19 pages of 16 = 304 tokens (only 4 wasted)
  Pages don't need to be contiguous in VRAM

Block table maps logical token positions to physical pages:
  Token 0-15   → Page 7 (physical)
  Token 16-31  → Page 23 (physical)
  Token 32-47  → Page 3 (physical)
  ...

New requests can use any free pages, regardless of memory layout.
```

### Impact on Eviction

PagedAttention enables efficient eviction at the page level:

```
H2O or other eviction:
  Evict a set of tokens → mark their pages as free → reuse for other requests

Without PagedAttention:
  Evicting tokens from the middle of a contiguous cache requires expensive copy
  Not practical for dynamic eviction

With PagedAttention:
  Eviction = freeing page references
  No data movement required
  Enables dynamic, fine-grained eviction in production
```

### vLLM Throughput Numbers

```
HuggingFace naive serving:
  Batch=1 only (no dynamic batching)
  ~50 tokens/sec on A100 (7B model)

vLLM (PagedAttention + continuous batching):
  Dynamic batch sizes up to GPU memory
  ~500–1500 tokens/sec on A100 (7B model)
  10–30× throughput improvement

With H2O or eviction integrated:
  Enables larger effective batch sizes
  KV cache is managed more efficiently
```

---

## 34. Quantizing the KV Cache

Quantization reduces KV cache memory by storing K, V at lower precision.

### The Case for KV Quantization

```
Standard KV cache:  FP16 (2 bytes per element)
INT8 KV cache:      1 byte per element → 2× smaller
INT4 KV cache:      0.5 bytes per element → 4× smaller

For 7B model, seq=4096, batch=8:
  FP16:  16.4 GB
  INT8:   8.2 GB
  INT4:   4.1 GB

At batch=8, INT4 saves 12.3 GB vs FP16.
This enables larger batches or longer contexts.
```

### Why KV Quantization Is Safer Than Weight Quantization

KV vectors are activations, not weights. Their distribution matters:

```
KV vectors have:
  - Similar scale within a head (easier to quantize)
  - Some outlier dimensions (more complex)

Research finding:
  K vectors have higher variance than V vectors
  K quantization causes more quality degradation than V
  → Some systems quantize V to INT4 and K to INT8 asymmetrically
```

### Methods

**Per-token quantization:** Compute a scale factor per token, quantize each K and V row independently.

**Per-channel quantization:** Compute scale per feature dimension across all tokens.

**Grouped quantization (like GPTQ for KV):** Groups of elements share a scale — balance of memory and accuracy.

### Real-World Usage

```
vLLM: FP8 KV cache option (recently added)
llama.cpp: INT8 KV cache by default in some configurations
TensorRT-LLM: INT8 KV with custom calibration
FlexGen: INT4 KV cache for offloading scenarios

Quality impact:
  FP8 KV:  Nearly identical to FP16, negligible degradation
  INT8 KV: Minor degradation (<1% on most benchmarks)
  INT4 KV: Noticeable on complex reasoning; OK for simple tasks
```

---

# Part 6 — Research Papers and Further Reading

---

## 35. Essential Papers with Summaries

### Foundational

**"Attention Is All You Need"**
Vaswani et al., Google, NeurIPS 2017
[arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

Introduced the Transformer with the attention mechanism that makes KV cache necessary. The Q, K, V decomposition directly leads to the KV cache optimization in autoregressive generation.

---

**"Fast Transformer Decoding: One Write-Head is All You Need"**
Noam Shazeer, Google, 2019
[arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)

MQA — the first paper to explicitly reduce KV cache by sharing K, V heads across query heads. Reduced KV cache by H× (number of heads). Motivated by the observation that KV cache bandwidth dominates decode latency.

---

**"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"**
Ainslie et al., Google Research, 2023
[arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)

Introduced Grouped Query Attention as a middle ground between MHA (full KV cache) and MQA (minimal KV cache). G=8 KV heads for 64 query heads is now the dominant production choice. Also introduced uptraining procedure.

---

### KV Cache Eviction

**"H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"**
Zhang, Sheng, et al., UT Austin / UW / Shanghai AI Lab, NeurIPS 2023
[arxiv.org/abs/2306.14048](https://arxiv.org/abs/2306.14048)

Introduced the observation that a small subset of "heavy hitter" tokens receive the majority of attention. Proposes keeping top-K by accumulated attention score + K most recent tokens. Showed 20×+ memory reduction with <5% quality loss on many tasks. Foundational paper for the eviction approach.

---

**"Efficient Streaming Language Models with Attention Sinks"**
Xiao et al., MIT / Meta AI, ICLR 2024
[arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453)

Discovered the "attention sink" phenomenon — the first few tokens receive disproportionate attention weight regardless of content. Showed that removing sinks causes catastrophic quality collapse. Proposed StreamingLLM: keep n_sink + W tokens for O(1) memory with infinite streaming capability. Also showed that re-indexing positions (not using original positions) is critical for RoPE-based models.

---

**"ScissorHands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time"**
Liu et al., 2023
[arxiv.org/abs/2305.17118](https://arxiv.org/abs/2305.17118)

Exploits the observation that attention importance is persistent across nearby queries. Uses a pivot-based approach to approximate heavy hitters with lower overhead than H2O's full accumulated scoring. Efficient alternative to H2O.

---

**"SnapKV: LLM Knows What You are Looking for Before Generation"**
Li et al., 2024
[arxiv.org/abs/2404.14469](https://arxiv.org/abs/2404.14469)

Proposes eviction during prefill rather than decode. Analyzes attention patterns over the prompt to decide which KV pairs to evict before generation begins. Avoids overhead of per-step eviction during decode. Works well for RAG (retrieval-augmented generation) where the document is fixed and queries vary.

---

**"PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling"**
Cai et al., 2024
[arxiv.org/abs/2406.02069](https://arxiv.org/abs/2406.02069)

Empirically shows that optimal KV cache size varies by layer — lower layers need more tokens, upper layers need fewer. Proposes a pyramidal allocation: large cache at bottom layers, small cache at top layers. Achieves same quality as uniform-budget eviction with less total memory.

---

**"Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference"**
Tang et al., 2024
[arxiv.org/abs/2406.10774](https://arxiv.org/abs/2406.10774)

Instead of evicting (deleting) KV pairs, Quest stores all KVs but only loads the relevant ones per step. Uses a lightweight page-level index to estimate relevance. Reduces bandwidth per decode step without permanent information loss.

---

### Sliding Window Attention

**"Longformer: The Long-Document Transformer"**
Beltagy et al., Allen Institute for AI, 2020
[arxiv.org/abs/2004.05150](https://arxiv.org/abs/2004.05150)

Introduced sliding window attention combined with global tokens for O(N) attention on long documents. Designed for encoders (BERT-style). Shows that local + global attention is sufficient for most long-document tasks. The template for mixing local and global attention.

---

**"Big Bird: Transformers for Longer Sequences"**
Zaheer et al., Google Research, NeurIPS 2020
[arxiv.org/abs/2007.14062](https://arxiv.org/abs/2007.14062)

Local + global + random attention. Theoretically proved that this combination is a universal approximator of the full attention matrix. Provided theoretical justification for why sparse attention works. Extended to very long sequences (up to 4096 in original, later extended).

---

**"Mistral 7B"**
Jiang et al., Mistral AI, 2023
[arxiv.org/abs/2310.06825](https://arxiv.org/abs/2310.06825)

The production model that popularized SWA in an open-weight decoder model. Uses alternating SWA (W=4096) and full attention layers. Also uses GQA. Demonstrated that SWA + GQA together give excellent quality/efficiency tradeoff. Widely deployed and fine-tuned by the community.

---

### Production Serving

**"Efficient Memory Management for Large Language Model Serving with PagedAttention"**
Kwon et al., UC Berkeley, SOSP 2023
[arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

Introduced vLLM and PagedAttention. Solved the memory fragmentation problem in KV cache allocation using OS virtual memory principles. 10–24× throughput improvement. Now standard in production LLM serving. Enables efficient implementation of eviction policies.

---

**"Continuous Batching of LLM Inference Requests"**
Yu et al., OSDI 2022 (Orca paper)
[arxiv.org/abs/2306.00008](https://arxiv.org/abs/2306.02539)

Introduced continuous (iteration-level) batching — dynamically inserting/removing requests mid-generation. Complements PagedAttention. Together these two papers form the foundation of modern LLM serving.

---

**"FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU"**
Sheng et al., Stanford, ICML 2023
[arxiv.org/abs/2303.06865](https://arxiv.org/abs/2303.06865)

Tackles running large models on limited hardware by offloading KV cache (and weights) to CPU or disk. Uses aggressive quantization (INT4) for both weights and KV. Enables running 70B+ models on consumer hardware at the cost of latency.

---

---

# Part 7 — Reference

---

## 36. Formula Sheet

### KV Cache Memory

```
M_kv = 2 × L × H_kv × S × d × B_dtype

L      = num_layers
H_kv   = num_kv_heads (= n_heads for MHA, G for GQA, 1 for MQA)
S      = seq_len (tokens in cache)
d      = head_dim = d_model / n_heads
B_dtype = bytes per element: 4 (FP32), 2 (FP16/BF16), 1 (INT8), 0.5 (INT4)

For multiple users: M_total = batch_size × M_kv (each user has own cache)
```

### GQA KV Reduction

```
GQA cache vs MHA cache = H_kv / n_heads = G / H

For H=32, G=8:
  GQA / MHA = 8/32 = 0.25  → 4× smaller cache
```

### SWA Cache Size (Fixed)

```
M_swa = 2 × L × H_kv × W × d × B_dtype   (W = window size, constant)

SWA receptive field = L × W   (information from L layers ago via multi-hop)
```

### StreamingLLM Cache Size (Fixed)

```
M_streaming = 2 × L × H_kv × (n_sink + W) × d × B_dtype
```

### H2O Cache Size (Fixed)

```
M_h2o = 2 × L × H_kv × K × d × B_dtype   (K = total budget = heavy_hitters + recent)
```

### Compute Comparison

```
Operation             FLOPs              Memory
──────────────────────────────────────────────────────────────────────────
Standard attention    O(N² × d × H)     O(N²) (attention matrix)
SWA                   O(N × W × d × H)  O(N × W) (banded matrix)
Decode with full KV   O(N × d × H)      O(N) KV reads per step
Decode with SWA KV    O(W × d × H)      O(W) KV reads per step (fixed!)
```

---

## 37. Glossary

| Term | Definition |
|------|-----------|
| **KV Cache** | Stored Key and Value tensors from past tokens, reused to avoid recomputation in decode |
| **Prefill** | Processing the input prompt in parallel — fast, compute-bound, populates KV cache |
| **Decode** | Generating new tokens one at a time — slow, memory-bound, reads KV cache each step |
| **TTFT** | Time To First Token — latency of the prefill phase |
| **TPOT** | Time Per Output Token — latency per decode step (depends on KV cache size) |
| **n_kv_heads** | Number of KV heads — reduced from n_heads in GQA/MQA to shrink KV cache |
| **head_dim (d)** | Dimension of each K or V vector = d_model / n_heads |
| **d_model** | Total model embedding dimension |
| **MHA** | Multi-Head Attention — H_kv = n_heads, largest cache |
| **GQA** | Grouped Query Attention — H_kv = G < n_heads, G× smaller cache |
| **MQA** | Multi-Query Attention — H_kv = 1, smallest cache possible |
| **MLA** | Multi-Head Latent Attention (DeepSeek) — low-rank K, V projection, only latent cached |
| **Eviction** | Removing some K, V vectors from cache to stay within memory budget |
| **Heavy Hitter** | Token that receives high attention weight across many steps — should be kept in cache |
| **Attention Sink** | First few tokens that receive disproportionate attention regardless of content |
| **H2O** | Heavy-Hitter Oracle — eviction policy keeping top-K by accumulated attention + recent |
| **StreamingLLM** | Eviction policy keeping n_sink sink tokens + W recent tokens — enables infinite streaming |
| **FIFO** | First-In-First-Out eviction — evict oldest tokens; fails for long-range tasks |
| **SWA** | Sliding Window Attention — each token attends to only the W most recent tokens |
| **Window size (W)** | SWA parameter: how many past tokens each position attends to |
| **Global token** | Token that participates in full attention even within SWA model (e.g., CLS, BOS) |
| **PagedAttention** | Non-contiguous KV cache allocation using fixed-size pages (vLLM) |
| **Continuous batching** | Dynamically adding/removing requests during generation to maximize batch size |
| **KV quantization** | Storing K, V in lower precision (INT8, INT4) to reduce cache memory |
| **Receptive field** | Effective range of tokens that can influence a given position (SWA: L × W) |
| **Memory-bound** | GPU bottlenecked by VRAM bandwidth, not FLOPs — typical during decode |
| **Compute-bound** | GPU bottlenecked by arithmetic operations — typical during prefill |
| **Budget (K)** | H2O parameter: maximum number of tokens to keep in cache |
| **n_sink** | StreamingLLM parameter: number of initial sink tokens to always preserve |
| **Accumulated attention** | H2O scoring: sum of attention weights a token received over all past steps |
| **RoPE** | Rotary Position Embedding — position encoding that allows re-indexing after eviction |
| **FlexGen** | LLM inference system that offloads KV cache to CPU/disk for very long contexts |
| **vLLM** | Production LLM serving system using PagedAttention + continuous batching |
| **SnapKV** | Eviction at prefill time — analyze prompt attention to select which KV to keep |
| **PyramidKV** | Layer-adaptive eviction: larger cache at bottom layers, smaller at top |
| **Quest** | Query-aware selective loading: stores all KV but loads only relevant pages per step |

---

*Notes covering KV cache mechanics, memory formulas, eviction policies (H2O and StreamingLLM), and sliding window attention — from first principles through production systems and research literature.*
