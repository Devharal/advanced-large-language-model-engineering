# Attention Variants & KV Cache — Deep Notes
## MHA → MQA → GQA: The Full Story from Basics to Research Papers

> Covers everything from scratch: what attention is, why KV cache exists, why it becomes a problem,
> and how MHA → MQA → GQA evolved to solve it. Includes backstory, intuition, math, memory
> calculations, real-world usage, and research paper references.

---

## Table of Contents

**Part 1 — Foundation**
1. [What Problem Does Attention Solve?](#1-what-problem-does-attention-solve)
2. [Self-Attention from Scratch](#2-self-attention-from-scratch)
3. [Query, Key, Value — The Retrieval Analogy](#3-query-key-value--the-retrieval-analogy)
4. [Scaled Dot-Product Attention — The Math](#4-scaled-dot-product-attention--the-math)
5. [Why the √d Scaling Factor?](#5-why-the-d-scaling-factor)
6. [Causal Masking — Why Autoregressive Models Need It](#6-causal-masking--why-autoregressive-models-need-it)

**Part 2 — Multi-Head Attention (MHA)**
7. [Why One Attention Head Is Not Enough](#7-why-one-attention-head-is-not-enough)
8. [Multi-Head Attention — Full Mechanics](#8-multi-head-attention--full-mechanics)
9. [What Each Head Actually Learns](#9-what-each-head-actually-learns)
10. [MHA Memory and Compute Cost](#10-mha-memory-and-compute-cost)

**Part 3 — KV Cache**
11. [Autoregressive Generation — The Token-by-Token Problem](#11-autoregressive-generation--the-token-by-token-problem)
12. [KV Cache — What It Is and Why It Exists](#12-kv-cache--what-it-is-and-why-it-exists)
13. [KV Cache Memory Math — How Big Does It Get?](#13-kv-cache-memory-math--how-big-does-it-get)
14. [KV Cache as the Inference Bottleneck](#14-kv-cache-as-the-inference-bottleneck)

**Part 4 — Multi-Query Attention (MQA)**
15. [The Backstory — Why MQA Was Invented](#15-the-backstory--why-mqa-was-invented)
16. [Multi-Query Attention — Full Mechanics](#16-multi-query-attention--full-mechanics)
17. [MQA Memory Savings — The Math](#17-mqa-memory-savings--the-math)
18. [MQA Quality Tradeoff](#18-mqa-quality-tradeoff)

**Part 5 — Grouped Query Attention (GQA)**
19. [The Problem MQA Left Unsolved](#19-the-problem-mqa-left-unsolved)
20. [Grouped Query Attention — Full Mechanics](#20-grouped-query-attention--full-mechanics)
21. [GQA Memory Math and Group Size Tuning](#21-gqa-memory-math-and-group-size-tuning)
22. [MHA vs MQA vs GQA — Side-by-Side Comparison](#22-mha-vs-mqa-vs-gqa--side-by-side-comparison)

**Part 6 — Related Techniques**
23. [Multi-Head Latent Attention (MLA) — DeepSeek's Approach](#23-multi-head-latent-attention-mla--deepseeks-approach)
24. [Sliding Window Attention — Handling Long Contexts](#24-sliding-window-attention--handling-long-contexts)
25. [PagedAttention — KV Cache Memory Management](#25-pagedattention--kv-cache-memory-management)
26. [FlashAttention — IO Efficiency for All Variants](#26-flashattention--io-efficiency-for-all-variants)

**Part 7 — Real-World Usage**
27. [Who Uses What — Modern LLMs](#27-who-uses-what--modern-llms)
28. [Converting MHA Models to GQA (Uptraining)](#28-converting-mha-models-to-gqa-uptraining)

**Part 8 — Research Papers and Further Reading**
29. [Essential Papers with Summaries](#29-essential-papers-with-summaries)
30. [Recommended Blogs and Tutorials](#30-recommended-blogs-and-tutorials)

**Part 9 — Quick Reference**
31. [Formula Sheet](#31-formula-sheet)
32. [Glossary](#32-glossary)

---

# Part 1 — Foundation

---

## 1. What Problem Does Attention Solve?

Before attention mechanisms existed, sequence models used **RNNs (Recurrent Neural Networks)** and **LSTMs**.

### The RNN Problem

An RNN processes tokens one by one, left to right. Each step passes a **hidden state** forward:

```
Token 1 → h1 → Token 2 → h2 → Token 3 → h3 → ... → Token N → hN
```

The final hidden state `hN` is supposed to encode everything about the entire sequence.

**Problem 1: The Bottleneck**

```
"The cat, which had been sitting on a mat in the corner of the old farmhouse
kitchen for most of the afternoon, was hungry."

To predict the verb after "was", the RNN must compress 25+ tokens
into a single vector hN.

Information from early tokens gets "forgotten".
```

**Problem 2: Sequential Processing**

RNNs cannot parallelize. Token 3 can't be processed until token 2 is done. This is catastrophically slow on GPUs (which are built for parallel work).

**Problem 3: Long-Range Dependencies**

"The keys to the cabinet **are** on the table."

The verb "are" agrees with "keys", not "cabinet". The RNN must carry that information across multiple tokens without losing it. This becomes harder as distance grows.

### What Attention Does Differently

Instead of a single bottleneck vector, attention lets every token **directly look at every other token** in the sequence:

```
Token 5 wants to know:
  "Which other tokens in this sequence are most relevant to me right now?"

It queries all other tokens.
Gets back a weighted sum of their information.
No information bottleneck.
No sequential dependency.
Full parallelism on GPU.
```

This is the core insight of the 2017 paper "Attention Is All You Need" — you don't need recurrence at all. Attention alone is sufficient to model sequences.

---

## 2. Self-Attention from Scratch

"Self-attention" means tokens attend to **other tokens in the same sequence** (as opposed to cross-attention, where a decoder attends to encoder outputs).

### A Simple Example

Input: `["The", "cat", "sat"]`

After embedding: each token becomes a vector (say, 4-dimensional):

```
"The" → [0.1, 0.2, 0.8, 0.3]
"cat" → [0.5, 0.1, 0.2, 0.9]
"sat" → [0.3, 0.7, 0.1, 0.4]
```

Self-attention asks: for each token, how much should it "attend to" every other token?

**"sat" attending to "cat":** The verb "sat" and the subject "cat" should be strongly connected — high attention weight.

**"sat" attending to "The":** Articles are less relevant to the verb — lower attention weight.

The output for "sat" will be a weighted combination of all token representations, with "cat" contributing more than "The".

This is how syntactic and semantic relationships emerge from training.

---

## 3. Query, Key, Value — The Retrieval Analogy

The Q, K, V framework is the most important thing to understand deeply.

### The Library Analogy

Imagine a **library**:

```
You walk in with a search query:   "books about French cooking"
                                          ↑
                                        Query (Q)

Every book has a label on the spine:   "French recipes", "Italian cuisine", ...
                                                ↑
                                              Keys (K)

The actual content inside each book:   pages of recipes
                                               ↑
                                             Values (V)
```

Process:
1. Your **Query** is compared to all **Keys**
2. Keys that match your query get high relevance scores
3. You retrieve a **weighted combination of Values** based on those scores

The result: you get information proportional to how relevant each source was to your question.

### In Transformers

Each token generates three vectors from its embedding:

```
Token embedding x  (dimension: d_model)
        │
        ├─── × W_Q  →  Query  q  (dimension: d_head)
        ├─── × W_K  →  Key    k  (dimension: d_head)
        └─── × W_V  →  Value  v  (dimension: d_head)
```

`W_Q`, `W_K`, `W_V` are learned weight matrices. They're learned during training to create useful projections.

**Query:** "What information am I looking for?"
**Key:** "What information do I have to offer?"
**Value:** "What information do I actually pass along if selected?"

### Why Separate Q from K?

Intuitively: a token's "question" (what it needs) and its "answer" (what it offers others) can be different things.

A pronoun "it" might:
- Query: "find my referent — who or what do I refer to?"
- Key: "I am a pronoun — pick me if you need anaphora resolution"
- Value: pass along features of its referent once found

---

## 4. Scaled Dot-Product Attention — The Math

### Step 1: Compute Scores

For a sequence of N tokens, stack all queries into matrix Q, all keys into K:

```
Q  [N × d_head]   — one query row per token
K  [N × d_head]   — one key row per token
V  [N × d_head]   — one value row per token
```

Scores matrix:

```
S = Q × Kᵀ        [N × N]

S[i][j] = dot product of query_i and key_j
         = "how much should token i attend to token j?"
```

### Step 2: Scale

```
S = S / √d_head
```

(Explained in next section)

### Step 3: Mask (for causal models)

```
S[i][j] = -∞   for all j > i
```

Tokens can't look at the future.

### Step 4: Softmax

```
A = softmax(S)   [N × N]

Each row of A sums to 1.
A[i] = attention distribution for token i over all other tokens.
```

### Step 5: Weighted Sum of Values

```
Output = A × V   [N × d_head]

Output[i] = Σ_j  A[i][j] × V[j]
           = weighted sum of all value vectors
             weighted by how much token i attended to token j
```

### Full Formula

```
Attention(Q, K, V) = softmax( Q × Kᵀ / √d_head ) × V
```

This single equation is the heart of the Transformer.

---

## 5. Why the √d Scaling Factor?

This is a detail most tutorials skip but it matters.

### The Problem Without Scaling

Suppose `d_head = 64`. Each query and key is a 64-dimensional vector.

The dot product `q · k` is the sum of 64 multiplications. As `d_head` grows, the magnitude of this dot product grows roughly like `√d_head`.

For large `d_head`, dot products become very large in magnitude:

```
Large dot products → extreme softmax inputs
                   → softmax output approaches one-hot
                   → almost all attention on one token
                   → gradients near zero (saturated softmax)
                   → training is very difficult
```

### The Fix

Divide by `√d_head` to normalize the dot products back to unit variance:

```
S = Q × Kᵀ / √d_head
```

After scaling, dot products have variance ~1 regardless of `d_head`. Softmax operates in a reasonable range. Gradients flow properly.

This is why the paper is called "Scaled Dot-Product Attention" — the scaling is a critical detail.

---

## 6. Causal Masking — Why Autoregressive Models Need It

### The Two Types of Transformers

**Encoder (BERT-style):** Bidirectional. Every token can attend to all other tokens. Used for understanding tasks (classification, NER).

**Decoder (GPT-style):** Causal / autoregressive. Each token can only attend to previous tokens. Used for generation.

### Why Causal Masking?

If during training, token 5 can see token 6 (a future token), the model can "cheat" — it already has the answer it's supposed to predict. It would learn to copy rather than learn real language structure.

```
During training (teacher forcing):
  Input:  ["The", "cat", "sat", "on", "the"]
  Target: ["cat", "sat", "on",  "the", "mat"]

If "sat" (position 3) can see "on" (position 4):
  → Model learns to copy forward instead of learning syntax
  → Model fails at real inference where future is unknown
```

Causal mask solution:

```
         The   cat   sat   on    the
The   [  1     0     0     0     0  ]
cat   [  1     1     0     0     0  ]
sat   [  1     1     1     0     0  ]
on    [  1     1     1     1     0  ]
the   [  1     1     1     1     1  ]

1 = can attend, 0 = masked to -∞
```

**Implementation:**

```python
mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
attention_weights = torch.softmax(scores, dim=-1)
```

---

# Part 2 — Multi-Head Attention (MHA)

---

## 7. Why One Attention Head Is Not Enough

Suppose you have one attention head. The model computes one set of Q, K, V projections and one attention distribution per token.

**Problem:** Language requires attending to multiple things simultaneously.

Consider: `"The animal didn't cross the street because it was too tired."`

Token "it" needs to:
- Attend to "animal" (for coreference — "it" = animal)
- Attend to "tired" (for the predicate)
- Attend to "cross" (for verb relation)
- Potentially track syntactic position

One attention pattern can't capture all these relationships simultaneously. It would have to compromise.

**Solution:** Run H independent attention computations in parallel, each learning to focus on different relationships.

This is **Multi-Head Attention**.

---

## 8. Multi-Head Attention — Full Mechanics

### The Architecture

Instead of one set of W_Q, W_K, W_V, we have H sets — one per head:

```
For each head h = 1...H:
    Q_h = X × W_Q_h    [N × d_head]
    K_h = X × W_K_h    [N × d_head]
    V_h = X × W_V_h    [N × d_head]
    
    head_h = Attention(Q_h, K_h, V_h)   [N × d_head]
```

Then concatenate all heads and project:

```
MultiHead(X) = Concat(head_1, ..., head_H) × W_O

Concat output: [N × (H × d_head)] = [N × d_model]
After W_O:     [N × d_model]
```

### Dimension Relationship

Standard choice:

```
d_head = d_model / H

Example (GPT-3 style):
    d_model = 4096
    H = 32 heads
    d_head = 4096 / 32 = 128
```

This keeps total computation similar to one big attention.

### Parameter Count for MHA

```
W_Q_h, W_K_h, W_V_h: each [d_model × d_head]

Per head: 3 × d_model × d_head parameters
All H heads: 3 × H × d_model × d_head
           = 3 × d_model × d_model   (since H × d_head = d_model)
           = 3 × d_model²

Output projection W_O: [d_model × d_model]

Total per layer: 4 × d_model²
```

For d_model=4096: 4 × 4096² = 67M parameters per attention layer.

---

## 9. What Each Head Actually Learns

Research has examined what individual attention heads specialize in. Common findings:

```
Head type          What it attends to
─────────────────────────────────────────────────────
Positional heads   Previous or next token (local context)
Syntactic heads    Subject-verb agreement
Coreference heads  Pronoun → antecedent resolution
Semantic heads     Semantically related words
Copy heads         Repeating recent tokens (important for in-context learning)
Rare word heads    Paying special attention to uncommon tokens
BOS heads          Large fraction of attention on [BOS] token (a "no-op" pattern)
```

Different heads in different layers learn qualitatively different patterns. This diversity is the primary justification for multi-head attention.

Papers that study this: "Are Sixteen Heads Really Better Than One?" (Michel et al., 2019) showed that many heads can be pruned without much quality loss — but during training, the diversity helps convergence.

---

## 10. MHA Memory and Compute Cost

### Training Memory (per layer)

```
Activations stored for backward pass:
  Q, K, V matrices:    3 × N × d_model  (per layer)
  Attention matrix:    H × N × N         (H heads, N×N each)
  Output:              N × d_model

For N=2048, d_model=4096, H=32, batch=16:
  Q+K+V: 3 × 2048 × 4096 × 16 × 2 bytes ≈ 768 MB
  Attn:  32 × 2048 × 2048 × 16 × 2 bytes ≈ 8.6 GB (!!)

Attention matrix dominates for long sequences.
This is the O(N²) memory problem.
```

### Inference Memory (with KV cache)

During inference, the attention matrix itself isn't the main problem — KV cache is. Covered in Part 3.

### Compute Cost

```
Q × Kᵀ:     2 × N² × d_head × H FLOPs = 2 × N² × d_model
Softmax + × V:  similar

Total attention FLOPs ≈ 4 × N² × d_model per layer
```

For long sequences (large N), this grows quadratically. At N=1M tokens, this is astronomically expensive.

---

# Part 3 — KV Cache

---

## 11. Autoregressive Generation — The Token-by-Token Problem

When an LLM generates text, it works **one token at a time**:

```
Prompt: "The capital of France is"

Step 1: Model sees ["The", "capital", "of", "France", "is"]
        → Outputs probability distribution over vocabulary
        → Picks "Paris" (highest probability)

Step 2: Model sees ["The", "capital", "of", "France", "is", "Paris"]
        → Outputs next token → "."

Step 3: Model sees all previous + "."
        → Outputs next token → "<EOS>"
        → Generation complete
```

At each step, the model runs a full forward pass over ALL tokens seen so far.

### The Naive Problem

At step 100 (100 tokens generated), the model must:
- Compute Q, K, V for all 100 tokens
- Compute attention between all 100 tokens
- Do this for every layer

At step 101, it does this again for 101 tokens. Including all the same K, V vectors it already computed at step 100.

**K and V for the first 99 tokens are computed identically at every step.** Pure redundant computation.

This is the problem KV cache solves.

---

## 12. KV Cache — What It Is and Why It Exists

### Core Idea

**Cache (store) the Key and Value tensors from all previous steps. Reuse them instead of recomputing.**

```
Step 1:
  Input: token_1
  Compute K_1, V_1
  Store K_1, V_1 in cache
  Attend: token_1 to {K_1}

Step 2:
  Input: new token token_2
  Compute K_2, V_2 for token_2 only
  Append to cache: {K_1, K_2}, {V_1, V_2}
  Attend: token_2 to {K_1, K_2}

Step 3:
  Input: new token token_3
  Compute K_3, V_3 for token_3 only
  Attend: token_3 to {K_1, K_2, K_3}
```

At every step, you only compute K and V for the **new token**, then append to the cache. Old K, V vectors are loaded from cache — fast, no recomputation.

### What You Do NOT Cache

**Queries (Q) are not cached** because:
- You only need the query for the current token (what is this token looking for?)
- The query for token 5 is irrelevant at step 6 — you're generating token 6, so you only need token 6's query
- Queries are computed fresh for each new token — cheap (only one row)

### Visual Summary

```
Without KV cache:
  Step N: compute K,V for all N tokens      = O(N) work per step
  Total for L steps: O(L²) work             = quadratic!

With KV cache:
  Step N: compute K,V for 1 new token       = O(1) work per step
  Total for L steps: O(L) work              = linear!

Memory cost: Store K,V for all previous tokens in GPU VRAM.
```

### Key Insight

KV cache is a **memory for compute tradeoff**:
- **Pay:** VRAM to store all past K, V vectors
- **Gain:** Don't recompute them — faster generation

For long contexts or large batches, the memory cost becomes enormous.

---

## 13. KV Cache Memory Math — How Big Does It Get?

### Formula

```
KV cache size (bytes) =
    2              (K and V)
  × num_layers     (one cache per layer)
  × num_heads      (one per head in MHA)
  × head_dim       (size of each K/V vector)
  × seq_len        (number of tokens cached)
  × batch_size     (simultaneous sequences)
  × bytes_per_elem (2 for FP16, 4 for FP32)
```

### Example: LLaMA 2 7B

```
Config:
  num_layers = 32
  num_heads  = 32
  head_dim   = 128
  FP16       → 2 bytes

For seq_len = 4096, batch = 1:
  = 2 × 32 × 32 × 128 × 4096 × 1 × 2
  = 2,147,483,648 bytes
  ≈ 2 GB

For seq_len = 32768 (32K context), batch = 1:
  ≈ 16 GB

For seq_len = 4096, batch = 32 (32 parallel users):
  ≈ 64 GB
```

A single H100 (80GB VRAM) would be entirely consumed by KV cache for 32 users at 4K context — before even loading the model weights.

### LLaMA 2 70B Example

```
Config:
  num_layers = 80
  num_heads  = 64
  head_dim   = 128

For seq_len = 4096, batch = 1:
  = 2 × 80 × 64 × 128 × 4096 × 1 × 2
  ≈ 13.4 GB

For seq_len = 100K (long context), batch = 1:
  ≈ 327 GB  (needs multiple H100s just for KV cache!)
```

### KV Cache vs Model Weights

```
Model              Weights (FP16)    KV cache (4K ctx, bs=1)
──────────────────────────────────────────────────────────────
LLaMA 2 7B         14 GB             2 GB
LLaMA 2 70B        140 GB            13.4 GB
GPT-3 175B         350 GB            ~50 GB
```

For serving at large batch sizes or long contexts, KV cache dominates.

---

## 14. KV Cache as the Inference Bottleneck

### The Memory Bound Problem

During inference:
1. Model weights are loaded once and stay in VRAM
2. For each new token, KV cache is read from VRAM
3. New K, V vectors are written to VRAM
4. Very little computation (just one new token's worth)

**Result: inference is heavily memory-bound**, not compute-bound.

```
For 7B model, generating 1 token:
  Read model weights:  ~14 GB from VRAM
  Read KV cache:       grows with context length
  Actual computation:  very small (1 token's worth)

GPU compute utilization during inference: often < 10%
GPU memory bandwidth: saturated
```

This is why:
- **Throughput scales with memory bandwidth** (how fast you can read weights and cache)
- **Batch size helps** (amortizes weight reads across many requests)
- **The number of heads × head_dim is a key tuning knob** — it directly controls KV cache size

Reducing KV cache size without hurting quality too much = the research problem that motivated MQA and GQA.

---

# Part 4 — Multi-Query Attention (MQA)

---

## 15. The Backstory — Why MQA Was Invented

### Context: 2019

The year is 2019. The "Attention Is All You Need" paper is two years old. BERT and GPT-2 are established. The field is scaling up models. And a researcher at Google Brain named **Noam Shazeer** (who also co-authored the original Transformer paper) is working on improving autoregressive generation speed.

The problem he observes:

```
During autoregressive decoding:
  - Queries: computed for 1 new token = tiny
  - Keys/Values: loaded from cache for ALL past tokens = huge

For each generated token:
  Weight loads:   ~14 GB (model weights)
  KV cache reads: grows with sequence × heads

With H=32 heads: you read 32× as much KV data as you need
if you only had 1 head.

The multiple heads of K and V are a bandwidth bottleneck
at inference time.
```

### The Insight

Q is computed fresh for each token — no caching needed. So having H independent Q projections is cheap.

But K and V are cached across all steps. Having H independent K and V caches means reading H × more data from VRAM at every step.

**What if K and V were shared across all query heads?**

You'd reduce KV cache size by H× (from 32 sets down to 1 set), while keeping all H query heads for expressiveness.

This is the Multi-Query Attention idea, published as "Fast Transformer Decoding: One Write-Head is All You Need" (Shazeer, 2019).

---

## 16. Multi-Query Attention — Full Mechanics

### Architecture

In MHA, each head has its own W_Q_h, W_K_h, W_V_h.

In MQA:
- Still H independent Query projections (W_Q_h for h=1...H)
- **Only ONE shared Key projection (W_K)**
- **Only ONE shared Value projection (W_V)**

```
MHA:
  Head h: Q_h = X × W_Q_h,  K_h = X × W_K_h,  V_h = X × W_V_h
  → H sets of Q, K, V

MQA:
  Head h: Q_h = X × W_Q_h,  K   = X × W_K,     V   = X × W_V
  → H sets of Q, but only 1 set of K and 1 set of V (shared)
```

### Computation

```
For each head h:
    Q_h = X × W_Q_h          [N × d_head]
    K   = X × W_K             [N × d_head]   ← same for all heads
    V   = X × W_V             [N × d_head]   ← same for all heads
    
    scores_h = Q_h × Kᵀ / √d_head   [N × N]
    attn_h   = softmax(scores_h)     [N × N]
    head_h   = attn_h × V            [N × d_head]

Output = Concat(head_1, ..., head_H) × W_O
```

**K and V are broadcast to all H heads.** Each head still produces a different output because each head has a different Q projection and thus a different attention distribution — but they all attend over the same keys and aggregate the same values.

### Forward Pass Equivalence

During training / prefill (processing the prompt), the compute is similar to MHA — you still run H attention computations. The difference is:

```
MHA: Each attention head reads its own K_h, V_h
MQA: All attention heads read the same K, V
```

For the forward pass, this is almost the same FLOP count (since Q × Kᵀ dominates).

### KV Cache in MQA

During generation:
```
MHA cache: H × (K vectors + V vectors) per layer
MQA cache: 1 × (K vectors + V vectors) per layer

MQA KV cache is H× smaller.
For H=32: MQA cache is 32× smaller than MHA cache.
```

---

## 17. MQA Memory Savings — The Math

### MHA KV Cache

```
= 2 × num_layers × H × d_head × seq_len × batch × bytes
= 2 × L × H × d_head × S × B × 2
```

### MQA KV Cache

```
= 2 × num_layers × 1 × d_head × seq_len × batch × bytes
= 2 × L × d_head × S × B × 2
```

### Savings

```
MQA cache = MHA cache / H

For H = 32:
MQA cache = MHA cache / 32

LLaMA 7B (MHA):  2 GB at 4K context
If it used MQA:  2 GB / 32 ≈ 62 MB at 4K context
```

This is a dramatic reduction — the KV cache almost disappears as a concern. More VRAM can go to larger batch sizes or longer contexts.

### Bandwidth Impact

```
Each generated token requires reading full KV cache:

MHA: Read H × (K + V) from VRAM per layer
MQA: Read 1 × (K + V) from VRAM per layer

For H=32, 32-layer model, 4K sequence:
MHA: read 2 GB per token generated
MQA: read 62 MB per token generated

MQA token generation: ~32× less memory bandwidth per step
→ ~32× more tokens/second at the same memory bandwidth
(approximation — weights still dominate at small contexts)
```

---

## 18. MQA Quality Tradeoff

MQA saves enormous memory but at a cost: quality.

### What's Lost

In MHA, each head has its own W_K and W_V. This means:
- Head 1's keys project to "syntactic" features
- Head 2's keys project to "semantic" features
- Head 7's keys project to "positional" features
- etc.

Each head can decide independently what to "offer" to queries and what to "pass along" as values.

In MQA, all heads share the same K and V projections:
- The single W_K must serve as a universal key space for all 32 query types
- The single W_V must encode everything all 32 heads might want to extract

This reduces the expressiveness of what different heads can learn. The model compensates somewhat with different Q projections (each head still asks different questions), but the answer space (K, V) is constrained.

### Empirical Findings

From the paper and subsequent work:

```
Task                      MHA quality    MQA quality
──────────────────────────────────────────────────────
Short generation (< 512)  Baseline       Within 1–2%
Long generation (> 2K)    Baseline       1–5% degradation
Complex reasoning         Baseline       Noticeable gap
Summarization             Baseline       Minor gap
```

The quality gap is real but often acceptable, especially when it enables:
- 10× larger batch sizes
- 10× longer context windows
- Faster inference that reaches users sooner

### Key Paper Finding

Shazeer's original paper showed MQA was "close in quality" to MHA while being significantly faster at inference. But many practitioners found the quality gap was non-trivial for production use cases, especially in larger models doing complex reasoning.

This set the stage for a middle ground: GQA.

---

# Part 5 — Grouped Query Attention (GQA)

---

## 19. The Problem MQA Left Unsolved

After MQA became known, teams started using it in production models. Results were mixed:

- **Smaller models (7B, 13B):** Quality gap acceptable
- **Larger models (70B+):** Quality gap noticeable, especially on benchmarks
- **Long-context tasks:** Quality degraded more than expected

The community needed something between:

```
MHA: H independent K/V heads → maximum quality, maximum KV cache
MQA: 1 shared K/V head → minimum KV cache, quality tradeoff

Need: G groups, each with 1 shared K/V head
      G groups × H/G queries per group = H queries total
      G < H but G > 1

This is GQA.
```

GQA was formalized in the paper "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al., Google Research, 2023).

---

## 20. Grouped Query Attention — Full Mechanics

### The Concept

Divide the H query heads into G groups. Each group shares one K head and one V head.

```
Example: H=8 query heads, G=2 groups

Group 1: Queries 1,2,3,4 → share K_1, V_1
Group 2: Queries 5,6,7,8 → share K_2, V_2
```

Visually:

```
MHA (H=8, G=8):                    MQA (H=8, G=1):
Q1 → K1,V1                         Q1 ─┐
Q2 → K2,V2                         Q2  │
Q3 → K3,V3                         Q3  │
Q4 → K4,V4                         Q4  ├→ K1,V1
Q5 → K5,V5                         Q5  │
Q6 → K6,V6                         Q6  │
Q7 → K7,V7                         Q7  │
Q8 → K8,V8                         Q8 ─┘

GQA (H=8, G=2):
Q1 ─┐                              
Q2  ├→ K1,V1                    
Q3  │                           
Q4 ─┘                           
Q5 ─┐                              
Q6  ├→ K2,V2                   
Q7  │                           
Q8 ─┘                           
```

GQA smoothly interpolates between MHA and MQA by varying G.

### Computation

```
queries_per_group = H / G

For each group g = 1...G:
    K_g = X × W_K_g             [N × d_head]
    V_g = X × W_V_g             [N × d_head]
    
    For each query head h in group g:
        Q_h = X × W_Q_h         [N × d_head]
        scores_h = Q_h × K_gᵀ / √d_head
        attn_h   = softmax(scores_h)
        head_h   = attn_h × V_g

Output = Concat(head_1, ..., head_H) × W_O
```

### Special Cases

```
G = H  →  MHA   (each query head has its own K, V)
G = 1  →  MQA   (all query heads share one K, V)
1 < G < H  →  GQA  (the general case)
```

---

## 21. GQA Memory Math and Group Size Tuning

### KV Cache Formula

```
GQA KV cache = MHA KV cache × (G / H)

= 2 × num_layers × G × d_head × seq_len × batch × bytes
```

### Examples (LLaMA 2 70B: H=64 heads)

| Setup | KV Cache (4K ctx, bs=1) |
|-------|------------------------|
| MHA (G=64) | 13.4 GB |
| GQA (G=8) | 1.67 GB |
| GQA (G=4) | 0.84 GB |
| MQA (G=1) | 0.21 GB |

LLaMA 2 70B uses **G=8** — roughly 8× memory reduction from MHA while preserving most of the quality.

### Quality vs Memory Tradeoff by Group Size

```
G=H (MHA):   Best quality,  Maximum memory
G=H/2:       Very close,    Half memory
G=H/4:       Minor drop,    Quarter memory  ← Often sweet spot
G=H/8:       Noticeable,    1/8 memory
G=1 (MQA):   Visible drop,  Minimum memory
```

The paper showed that quality degrades roughly logarithmically with G — halving G costs less than you might expect.

### Choosing G in Practice

The most common choices in production models:

```
Total heads H = 32 (7B models):
  G = 8   → 4 queries per group (LLaMA 2 7B uses this)

Total heads H = 64 (70B models):
  G = 8   → 8 queries per group (LLaMA 2 70B, LLaMA 3 70B)

Total heads H = 32 (Mistral 7B):
  G = 8   → 4 queries per group
```

G=8 seems to be the empirical sweet spot across model sizes — significant memory savings without quality loss.

---

## 22. MHA vs MQA vs GQA — Side-by-Side Comparison

### Architecture Comparison

| Property | MHA | MQA | GQA |
|----------|-----|-----|-----|
| Query heads | H | H | H |
| Key heads | H | 1 | G |
| Value heads | H | 1 | G |
| KV cache size | H × d_head | 1 × d_head | G × d_head |
| KV cache ratio | 1× (baseline) | 1/H | G/H |
| Expressive power | Maximum | Minimum | Tunable |
| Inference speed | Slowest | Fastest | Tunable |

### Memory Comparison (LLaMA 2 style, H=32, d_head=128, FP16)

```
Per token, per layer:
  MHA: 32 × 128 × 2 × 2 bytes = 16,384 bytes = 16 KB/token/layer
  GQA: 8  × 128 × 2 × 2 bytes =  4,096 bytes =  4 KB/token/layer
  MQA: 1  × 128 × 2 × 2 bytes =    512 bytes = 512 B/token/layer

For 32 layers, 100K tokens:
  MHA: 16 KB × 32 × 100K = 51.2 GB
  GQA: 4 KB  × 32 × 100K = 12.8 GB
  MQA: 512 B × 32 × 100K =  1.6 GB
```

### When to Use Each

```
MHA:  Research models, accuracy critical, short contexts, smaller models
      → BERT-base, original GPT-2, many academic models

MQA:  Extreme inference optimization, memory very tight
      → PaLM 2 (partially), early production decoder models

GQA:  Production LLMs — best balance of quality + efficiency
      → LLaMA 2/3, Mistral, Mixtral, Gemma, Falcon, most modern models
```

---

# Part 6 — Related Techniques

---

## 23. Multi-Head Latent Attention (MLA) — DeepSeek's Approach

### The Context

In 2024, DeepSeek (a Chinese AI lab) introduced a different approach to the KV cache problem in their **DeepSeek-V2** and **DeepSeek-V3** models.

The insight: instead of reducing the *number* of K/V heads (like GQA does), what if we reduce the *dimensionality* of what gets cached?

### The Core Idea

MLA introduces a **latent bottleneck** for K and V:

```
Standard:
  K = X × W_K    [N × d_head × H]  ← full dimensionality cached

MLA:
  c = X × W_down   [N × d_latent]   ← low-rank projection (SMALL, cached)
  K = c × W_up_K   [N × d_head × H] ← expanded at attention time (NOT cached)
  V = c × W_up_V   [N × d_head × H]
```

You only cache `c` (the compressed latent), which is much smaller than full K/V.

At attention time, you expand `c` back to full K/V on the fly.

### Memory Comparison

```
Standard MHA cache:  H × d_head × 2 (K+V) per token per layer
MLA cache:           d_latent per token per layer

If d_latent << H × d_head × 2:
  MLA gives MHA-quality with much smaller cache

DeepSeek-V2:
  H=128 heads, d_head=128
  Standard: 128 × 128 × 2 = 32,768 dims cached
  MLA:      512 dims cached
  Ratio: ~64× compression
```

### Trade-off

The W_up_K and W_up_V expansions happen at inference time — extra compute per token. But for long contexts where memory is the bottleneck, this trade is very favorable.

MLA enables DeepSeek-V2/V3 to run very long contexts with dramatically smaller KV caches, which was a key enabler of their efficient inference approach.

---

## 24. Sliding Window Attention — Handling Long Contexts

### The Problem

For very long sequences, even GQA's reduced KV cache can become enormous. And attending over 100K+ tokens is computationally expensive (O(N²)).

### The Solution

**Sliding Window Attention (SWA)** limits each token to attending only within a local window:

```
Window size W = 4096

Token 10,000 attends to tokens: 6,000 to 10,000
                                 └──────────────┘
                                    4,000 tokens (the window)

Token 10,001 attends to tokens: 6,001 to 10,001
```

### Memory Impact

```
Standard attention KV cache: grows with full seq_len
SWA KV cache:                fixed at W (window size) per token

For W=4096:
  KV cache is fixed regardless of total sequence length
  1 million token context → same cache size as 4K context
```

### Global Attention (Mixed)

Most implementations mix local + global:

```
Some layers: Sliding window (fast, local context)
Some layers: Full attention (slow but captures long-range)

Mistral 7B: alternates every layer
```

### Used In

- **Mistral 7B:** Window size 4096, interleaved with full attention
- **Mixtral 8×7B:** Same pattern
- **Longformer (Allen AI):** Window + global tokens

---

## 25. PagedAttention — KV Cache Memory Management

### The Memory Fragmentation Problem

Traditional KV cache implementation:

```
For each request, pre-allocate contiguous VRAM:
  Request 1 might use 2K tokens  → allocate 4K tokens worth (max possible)
  Request 2 might use 1K tokens  → allocate 4K tokens worth
  ...

Result:
  Fragmented, wasted VRAM
  Cannot mix long and short requests efficiently
  50–80% VRAM wasted on average
```

### PagedAttention (vLLM, 2023)

Inspired by OS virtual memory paging. Instead of contiguous allocation:

```
KV cache is split into fixed-size "blocks" (like memory pages)
  Block size: e.g., 16 tokens

For a request using 100 tokens:
  Allocate 7 blocks of 16 = 112 tokens (minimal waste)
  
Blocks for different requests can be interleaved in VRAM
  Request 1: blocks [2, 7, 15, 22, ...]
  Request 2: blocks [1, 4, 9, 11, ...]
```

A **block table** maps logical positions to physical blocks.

### Benefits

```
Without PagedAttention (continuous batching):
  ~60% VRAM utilization (rest wasted on fragmentation)

With PagedAttention:
  ~95%+ VRAM utilization
  Supports 2–4× larger batch sizes
  Enables dynamic request insertion
```

### Impact on Inference Serving

PagedAttention was the core innovation behind **vLLM**, which became the de facto standard for production LLM serving. Combined with continuous batching, it dramatically improved throughput:

```
vLLM (PagedAttention) vs HuggingFace transformers naive:
  Throughput: 10–24× improvement depending on model and workload
```

---

## 26. FlashAttention — IO Efficiency for All Variants

FlashAttention (Dao et al., 2022) is orthogonal to MHA/MQA/GQA — it speeds up the attention computation itself by reducing memory bandwidth usage, regardless of which variant you use.

It applies equally to MHA, MQA, and GQA:

```
MHA + FlashAttention: faster, less memory than naive MHA
GQA + FlashAttention: faster, less memory than naive GQA
```

### Interaction with GQA

FlashAttention-2 (Dao, 2023) added explicit support for GQA/MQA, broadcasting the shared K/V heads correctly across query heads without materializing the full expanded matrices.

```python
# PyTorch 2.0+ — handles GQA natively
output = F.scaled_dot_product_attention(
    query,          # [batch, H, seq, d_head]
    key,            # [batch, G, seq, d_head]   G < H
    value,          # [batch, G, seq, d_head]
    is_causal=True  # causal mask
)
# PyTorch automatically broadcasts K, V from G heads to H heads
```

FlashAttention handles the broadcasting inside the SRAM tile loop, so you never materialize the expanded K/V in HBM.

---

# Part 7 — Real-World Usage

---

## 27. Who Uses What — Modern LLMs

### Attention Variants in Production Models

| Model | Released | Attention Type | Heads (Q) | KV Heads (G) | Notes |
|-------|----------|----------------|-----------|--------------|-------|
| BERT-base | 2018 | MHA | 12 | 12 | Encoder-only |
| GPT-2 (1.5B) | 2019 | MHA | 25 | 25 | |
| GPT-3 (175B) | 2020 | MHA | 96 | 96 | |
| PaLM (540B) | 2022 | MQA | 48 | 1 | First major MQA LLM |
| Falcon 7B | 2023 | MHA | 71 | 71 | |
| Falcon 40B | 2023 | MQA | 64 | 8 | Actually multiquery |
| LLaMA 2 7B | 2023 | GQA | 32 | 8 | |
| LLaMA 2 13B | 2023 | GQA | 40 | 8 | |
| LLaMA 2 70B | 2023 | GQA | 64 | 8 | |
| Mistral 7B | 2023 | GQA | 32 | 8 | |
| Mixtral 8×7B | 2023 | GQA | 32 | 8 | MoE + GQA |
| Gemma 7B | 2024 | MQA | 16 | 1 | |
| Gemma 2 9B | 2024 | GQA | 8 | 4 | |
| Phi-3 | 2024 | GQA | 32 | 8 | |
| LLaMA 3 8B | 2024 | GQA | 32 | 8 | |
| LLaMA 3 70B | 2024 | GQA | 64 | 8 | |
| LLaMA 3 405B | 2024 | GQA | 128 | 8 | |
| Mistral Nemo | 2024 | GQA | 32 | 8 | |
| DeepSeek-V2 | 2024 | MLA | 128 | — | Custom latent compression |
| DeepSeek-V3 | 2024 | MLA | 128 | — | |
| Qwen 2.5 7B | 2024 | GQA | 28 | 4 | |
| Qwen 2.5 72B | 2024 | GQA | 64 | 8 | |

**Pattern:** The industry has largely converged on GQA with G=8, providing ~4–8× KV cache reduction vs MHA. G=8 appears to be the quality/efficiency sweet spot across all major model families.

---

## 28. Converting MHA Models to GQA (Uptraining)

The GQA paper introduced an important technique: converting an existing MHA checkpoint to GQA without training from scratch.

### Why This Matters

If you already have a well-trained MHA model (like GPT-3 or LLaMA-1), you don't want to train a new GQA model from scratch. Uptraining lets you convert and then fine-tune for a fraction of the original training cost.

### The Conversion Process

**Step 1: Group the H existing K/V heads into G groups**

```
Original MHA: H K heads [W_K_1, W_K_2, ..., W_K_H]

For GQA with G groups:
  Group 1: heads 1 to H/G
  Group 2: heads (H/G + 1) to 2H/G
  ...
  Group G: heads (H - H/G + 1) to H
```

**Step 2: Create one K/V head per group by mean pooling**

```
W_K_group_g = mean(W_K_{g*step} ... W_K_{(g+1)*step - 1})
```

Mean pooling (averaging) the existing K/V projection matrices within each group.

**Step 3: Continue training (uptraining)**

Resume training on ~5% of original training tokens. The model adapts to using the reduced K/V heads while the Q heads stay independent.

### Results from the Paper

```
Original MHA 7B:       Baseline quality
GQA 7B (G=8) trained from scratch:  Slight gap from MHA
GQA 7B (G=8) uptrained from MHA:    Essentially matches MHA quality

Uptraining cost: ~5% of original training compute
Result: Near-MHA quality with H/8 KV cache
```

This technique was used by Meta to create LLaMA 2 70B (which was uptrained from LLaMA 1 using GQA).

---

# Part 8 — Research Papers and Further Reading

---

## 29. Essential Papers with Summaries

### Foundational

**"Attention Is All You Need"**
Vaswani et al., Google Brain/Research, 2017
[arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

The paper that introduced the Transformer architecture. Defined Q, K, V, multi-head attention, positional encoding, and the encoder-decoder structure. Replaced RNNs for sequence transduction (translation). Required reading — everything else builds on this.

---

**"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"**
Devlin et al., Google, 2018
[arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

Showed that a bidirectional Transformer (encoder-only) pre-trained on masked language modeling achieves SOTA on 11 NLP tasks. Established the pre-train → fine-tune paradigm. Uses standard MHA.

---

### KV Cache and Attention Variants

**"Fast Transformer Decoding: One Write-Head is All You Need"**
Noam Shazeer, Google, 2019
[arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)

The original MQA paper. Proposes sharing a single K/V head across all query heads. Shows significant inference speedup (1.7–4× on autoregressive decoding tasks) with modest quality loss. Short paper, very readable. Introduced the core idea 4 years before it became mainstream.

---

**"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"**
Ainslie, Lee-Thorp, et al., Google Research, 2023
[arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)

Introduced Grouped Query Attention. Shows that G groups of K/V heads interpolates smoothly between MHA and MQA. Proposes the uptraining procedure (mean pooling → continue training). Empirically validates G=8 as the quality/efficiency sweet spot. Directly led to LLaMA 2's architecture choice.

---

**"Efficient Streaming Language Models with Attention Sinks"**
Xiao et al., MIT/Meta, 2023
[arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453)

Discovered that certain early tokens (especially the first token) receive unexpectedly high attention weights — called "attention sinks." Shows how to use sliding window attention while keeping these sink tokens in the window to preserve performance. Key insight for long-context inference.

---

### FlashAttention

**"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"**
Dao et al., Stanford, 2022
[arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)

Showed that standard attention is IO-bound (not compute-bound) and proposed tiling to keep computation in SRAM. 2–4× speedup on attention kernels, O(N) memory instead of O(N²). Now the standard implementation in PyTorch and HuggingFace.

---

**"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"**
Dao, 2023
[arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)

Improved FA-1 by parallelizing across sequence length, reducing non-matmul FLOPs, and better utilizing Tensor Cores. ~2× speedup over FA-1. Added native GQA/MQA support.

---

**"FlashAttention-3: Fast and Accurate Attention for GPUs from H100s to Blackwells"**
Shah et al., 2024
[arxiv.org/abs/2407.08608](https://arxiv.org/abs/2407.08608)

Targets H100/Hopper architecture specifically. Uses asynchronous memory copies, pipeline overlapping of GEMM and softmax, and FP8 Tensor Cores. ~2× over FA-2 on H100.

---

### Inference & KV Cache Optimization

**"Efficient Memory Management for Large Language Model Serving with PagedAttention"**
Kwon et al., UC Berkeley, 2023
[arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

Introduced PagedAttention and the vLLM serving system. Applies OS virtual memory concepts to KV cache management. 10–24× throughput improvement over naive serving. One of the most practically impactful inference papers.

---

**"Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation"**
Leviathan et al., Google, 2023
[arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192)

Formalized speculative decoding: use small draft model to propose tokens, verify with large model in parallel. 2–3× latency reduction. Combined with KV cache management for modern inference pipelines.

---

**"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"**
DeepSeek, 2024
[arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434)

Introduced Multi-head Latent Attention (MLA): low-rank compression of K/V into a latent space, only caching the latent. 5–13× KV cache reduction vs GQA with no quality loss. Also uses DeepSeekMoE. Strong commercial and research impact.

---

### Long Context

**"Mistral 7B"**
Jiang et al., Mistral AI, 2023
[arxiv.org/abs/2310.06825](https://arxiv.org/abs/2310.06825)

Popularized GQA (with G=8) in a widely-adopted open model. Also uses Sliding Window Attention for long context handling. The model became a community standard for fine-tuning and deployment.

---

**"LLaMA 2: Open Foundation and Fine-Tuned Chat Models"**
Touvron et al., Meta, 2023
[arxiv.org/abs/2307.09288](https://arxiv.org/abs/2307.09288)

Formally adopted GQA for the 34B and 70B models. Documents the uptraining procedure from LLaMA 1 MHA checkpoints. Most widely used open-weight model family through 2024.

---

**"Extending Context Window of Large Language Models via Positional Interpolation"**
Chen et al., Meta, 2023
[arxiv.org/abs/2306.15595](https://arxiv.org/abs/2306.15595)

Shows how to extend RoPE-based models to longer context windows via interpolation. KV cache size is a key concern here — GQA makes long-context extension far more feasible.

---

## 30. Recommended Blogs and Tutorials

**"The Illustrated Transformer" — Jay Alammar**
[jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer)
The single best visual introduction to attention and the Transformer. Highly recommended as a first read.

---

**"The Illustrated GPT-2" — Jay Alammar**
[jalammar.github.io/illustrated-gpt2](https://jalammar.github.io/illustrated-gpt2)
Excellent deep dive into decoder-only Transformers and autoregressive generation. Covers KV cache mechanics visually.

---

**"Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA"**
[huggingface.co/blog/4bit-transformers-bitsandbytes](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
Good practical coverage of memory constraints and how quantization + GQA work together in production.

---

**"FlashAttention: Fast and Memory-Efficient Exact Attention" — Tri Dao's blog**
[tridao.me/publications/flash2/flash2.pdf](https://tridao.me/publications/flash2/flash2.pdf)
The author's explanation of FlashAttention internals. Worth reading after the paper.

---

**"Dissecting Batching Effects in GPT Inference" — Cursor blog**
Excellent practical exploration of how batch size, KV cache, and memory bandwidth interact.

---

**"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"**
[arxiv.org/abs/2006.16236](https://arxiv.org/abs/2006.16236)
Shows an alternative formulation of attention as an RNN — useful for understanding the relationship between sequential and parallel views of attention.

---

**Andrej Karpathy — "Let's build GPT"**
[youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)
50K-token GPT built from scratch in Python. Best hands-on introduction to Transformer implementation.

---

**"A Visual Guide to Mamba and State Space Models" — Maarten Grootendorst**
[newsletter.maartengrootendorst.com](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)
Covers the attention alternative. Understanding Mamba's linear-time KV handling gives a different lens on why KV cache is a problem worth solving.

---

# Part 9 — Quick Reference

---

## 31. Formula Sheet

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax( Q × Kᵀ / √d_head ) × V

Q: [N × d_head]     (queries)
K: [N × d_head]     (keys)
V: [N × d_head]     (values)
N: sequence length
d_head: head dimension = d_model / H
```

### MHA

```
Q_h = X × W_Q_h       for h = 1...H   each [N × d_head]
K_h = X × W_K_h       for h = 1...H
V_h = X × W_V_h       for h = 1...H

head_h = Attention(Q_h, K_h, V_h)
Output = Concat(head_1, ..., head_H) × W_O
```

### MQA

```
Q_h = X × W_Q_h       for h = 1...H   (H different Q projections)
K   = X × W_K                          (ONE shared K projection)
V   = X × W_V                          (ONE shared V projection)

head_h = Attention(Q_h, K, V)
Output = Concat(head_1, ..., head_H) × W_O
```

### GQA

```
Group size: queries_per_group = H / G

For group g = 1...G:
    K_g = X × W_K_g
    V_g = X × W_V_g
    For head h in group g:
        Q_h = X × W_Q_h
        head_h = Attention(Q_h, K_g, V_g)

Output = Concat(all heads) × W_O
```

### KV Cache Size

```
KV cache (bytes) = 2 × L × G × d_head × S × B × bytes_per_elem

L = num_layers
G = num_KV_heads (= H for MHA, = 1 for MQA, = G for GQA)
d_head = d_model / H
S = sequence length (tokens cached)
B = batch size
bytes_per_elem = 2 (FP16) or 4 (FP32)
```

### Attention Compute Cost

```
Per layer, per attention head:
  Q × Kᵀ:  2 × N² × d_head FLOPs
  A × V:   2 × N² × d_head FLOPs
  Total:   4 × N² × d_head × H FLOPs = 4 × N² × d_model

For entire model (L layers):
  Total attention FLOPs ≈ 4 × L × N² × d_model
```

### GQA Special Cases

```
GQA(G = H) = MHA
GQA(G = 1) = MQA
```

---

## 32. Glossary

| Term | Definition |
|------|-----------|
| **Attention** | Mechanism for computing a weighted combination of values, weighted by query-key similarity |
| **Query (Q)** | "What am I looking for?" — computed for every token |
| **Key (K)** | "What do I offer to queries?" — compared against queries |
| **Value (V)** | "What information do I pass along if selected?" — aggregated by attention weights |
| **d_model** | Total embedding dimension of the model (e.g., 4096) |
| **d_head** | Dimension of each attention head = d_model / H |
| **H** | Total number of query heads |
| **G** | Number of KV head groups (G=H: MHA, G=1: MQA) |
| **Softmax** | Normalizes a vector to sum to 1 — turns raw scores into probabilities |
| **Causal Mask** | Prevents tokens from attending to future tokens during autoregressive generation |
| **Self-Attention** | Attention where tokens attend to other tokens in the same sequence |
| **Cross-Attention** | Attention where tokens in one sequence attend to another (e.g., decoder attending to encoder) |
| **MHA** | Multi-Head Attention — H independent heads, each with own K/V |
| **MQA** | Multi-Query Attention — H query heads, single shared K/V |
| **GQA** | Grouped Query Attention — H query heads, G shared K/V groups |
| **MLA** | Multi-Head Latent Attention — low-rank K/V compression (DeepSeek) |
| **KV Cache** | Cached Key/Value tensors from previous tokens, avoiding recomputation during generation |
| **Autoregressive** | Generating one token at a time, each conditioned on all previous tokens |
| **Prefill** | Processing the input prompt in parallel (fast) |
| **Decode** | Generating new tokens one at a time (slow, KV cache helps) |
| **Attention Sink** | First token(s) receiving disproportionately high attention weight |
| **Sliding Window Attention** | Each token attends only to the last W tokens, fixed memory |
| **PagedAttention** | OS-inspired non-contiguous KV cache allocation for efficient VRAM use |
| **vLLM** | LLM serving system based on PagedAttention |
| **Uptraining** | Converting an existing MHA model to GQA by mean-pooling heads and continuing training |
| **FlashAttention** | IO-aware attention algorithm that tiles computation in SRAM, avoids HBM materialization |
| **RoPE** | Rotary Position Embedding — encodes position in Q and K via rotation |
| **GQA group size** | G — the key hyperparameter: G=H is MHA, G=1 is MQA, G=8 is common production choice |
| **TTFT** | Time To First Token — latency metric for prefill phase |
| **TPOT** | Time Per Output Token — latency metric for decode phase |
| **Throughput** | Tokens generated per second across all active requests |

---

*Notes compiled from original research papers, production model documentation, and systems engineering literature. Covers the full evolution from basic attention to modern GQA-based production LLMs.*