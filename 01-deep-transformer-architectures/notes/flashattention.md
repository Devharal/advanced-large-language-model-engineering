# FlashAttention — Complete Deep Notes
## IO-Aware Kernels, Online Softmax, FA-1 → FA-2 → FA-3

> Covers everything from scratch: why naive attention is slow, what "IO-aware" means,
> the online softmax trick, tiling in SRAM, and every improvement from FA-1 through FA-3.
> Includes backstory, math, pseudocode, benchmarks, and research paper references.

---

## Table of Contents

**Part 1 — Why Attention Is Slow (The Real Reason)**
1. [The Standard Attention Algorithm](#1-the-standard-attention-algorithm)
2. [GPU Memory Hierarchy Recap](#2-gpu-memory-hierarchy-recap)
3. [Where Naive Attention Spends Its Time](#3-where-naive-attention-spends-its-time)
4. [The Roofline Model — Compute Bound vs Memory Bound](#4-the-roofline-model--compute-bound-vs-memory-bound)
5. [Memory Bandwidth Is the Real Bottleneck](#5-memory-bandwidth-is-the-real-bottleneck)

**Part 2 — The IO-Aware Insight**
6. [What "IO-Aware" Actually Means](#6-what-io-aware-actually-means)
7. [Tiling — The Core Strategy](#7-tiling--the-core-strategy)
8. [The Problem Tiling Creates — Softmax](#8-the-problem-tiling-creates--softmax)

**Part 3 — Online Softmax**
9. [Standard Softmax and Why It Needs the Full Row](#9-standard-softmax-and-why-it-needs-the-full-row)
10. [Numerically Stable Softmax — The First Fix](#10-numerically-stable-softmax--the-first-fix)
11. [Online Softmax — One Pass, Any Number of Tiles](#11-online-softmax--one-pass-any-number-of-tiles)
12. [The Running Statistics Trick — Step by Step](#12-the-running-statistics-trick--step-by-step)
13. [Why Online Softmax Is Mathematically Exact](#13-why-online-softmax-is-mathematically-exact)

**Part 4 — FlashAttention-1**
14. [Backstory — The 2022 Stanford Paper](#14-backstory--the-2022-stanford-paper)
15. [The FlashAttention-1 Algorithm](#15-the-flashattention-1-algorithm)
16. [FlashAttention-1 Forward Pass — Full Pseudocode](#16-flashattention-1-forward-pass--full-pseudocode)
17. [The Backward Pass Problem — Recomputation](#17-the-backward-pass-problem--recomputation)
18. [FlashAttention-1 Memory Analysis](#18-flashattention-1-memory-analysis)
19. [FlashAttention-1 IO Complexity Analysis](#19-flashattention-1-io-complexity-analysis)
20. [FlashAttention-1 Benchmarks and Results](#20-flashattention-1-benchmarks-and-results)

**Part 5 — FlashAttention-2**
21. [What FA-1 Left on the Table](#21-what-fa-1-left-on-the-table)
22. [FA-2 Improvement 1 — Fewer Non-Matmul FLOPs](#22-fa-2-improvement-1--fewer-non-matmul-flops)
23. [FA-2 Improvement 2 — Parallelism Across Sequence Dimension](#23-fa-2-improvement-2--parallelism-across-sequence-dimension)
24. [FA-2 Improvement 3 — Better Work Partitioning Within an SM](#24-fa-2-improvement-3--better-work-partitioning-within-an-sm)
25. [FA-2 Improvement 4 — Native GQA/MQA Support](#25-fa-2-improvement-4--native-gqamqa-support)
26. [FlashAttention-2 Algorithm — What Changed](#26-flashattention-2-algorithm--what-changed)
27. [FlashAttention-2 Benchmarks and Results](#27-flashattention-2-benchmarks-and-results)

**Part 6 — FlashAttention-3**
28. [The H100 Hopper Architecture — New Hardware Features](#28-the-h100-hopper-architecture--new-hardware-features)
29. [FA-3 Improvement 1 — Asynchronous TMA Copies](#29-fa-3-improvement-1--asynchronous-tma-copies)
30. [FA-3 Improvement 2 — Warp Specialization](#30-fa-3-improvement-2--warp-specialization)
31. [FA-3 Improvement 3 — GEMM and Softmax Overlap](#31-fa-3-improvement-3--gemm-and-softmax-overlap)
32. [FA-3 Improvement 4 — FP8 Tensor Core Support](#32-fa-3-improvement-4--fp8-tensor-core-support)
33. [FA-3 Improvement 5 — Incoherent Processing](#33-fa-3-improvement-5--incoherent-processing)
34. [FlashAttention-3 Benchmarks and Results](#34-flashattention-3-benchmarks-and-results)

**Part 7 — Putting It All Together**
35. [FA-1 vs FA-2 vs FA-3 — Full Comparison](#35-fa-1-vs-fa-2-vs-fa-3--full-comparison)
36. [How to Use FlashAttention in Practice](#36-how-to-use-flashattention-in-practice)
37. [What FlashAttention Does Not Solve](#37-what-flashattention-does-not-solve)
38. [Alternatives and Related Work](#38-alternatives-and-related-work)

**Part 8 — Research Papers and Further Reading**
39. [Essential Papers with Summaries](#39-essential-papers-with-summaries)
40. [Formula Sheet and Quick Reference](#40-formula-sheet-and-quick-reference)
41. [Glossary](#41-glossary)

---

# Part 1 — Why Attention Is Slow (The Real Reason)

---

## 1. The Standard Attention Algorithm

Before understanding why FlashAttention is fast, you need to know exactly what the standard (naive) attention algorithm does and where its time goes.

### Standard Attention — Step by Step

Given:
```
Q: [N × d]    (query matrix, N = sequence length, d = head dimension)
K: [N × d]    (key matrix)
V: [N × d]    (value matrix)
```

**Step 1: Compute raw scores**
```
S = Q × Kᵀ            [N × N]    (matrix multiply)
```

**Step 2: Scale**
```
S = S / √d             [N × N]    (elementwise divide)
```

**Step 3: Apply causal mask (for decoder models)**
```
S[i][j] = -∞  for j > i           (elementwise mask)
```

**Step 4: Softmax row by row**
```
P = softmax(S, dim=-1)  [N × N]   (row-wise softmax)
```

**Step 5: Weighted sum of values**
```
O = P × V              [N × d]    (matrix multiply)
```

**Output:** O, the attention output for this head.

### The Memory Footprint

For every step, data must live somewhere. In the naive implementation:

```
S = Q × Kᵀ            → S written to HBM: N × N floats
P = softmax(S)         → S read from HBM, P written to HBM: N × N floats
O = P × V             → P read from HBM: N × N floats

Total HBM traffic from the N×N matrix alone:
   Write S:     N² × 4 bytes
   Read S:      N² × 4 bytes  (for softmax)
   Write P:     N² × 4 bytes  (softmax output)
   Read P:      N² × 4 bytes  (for P × V)
   = 4 × N² × 4 bytes = 16 N² bytes of HBM traffic

For N=2048, one head, FP32:
   16 × 2048² × 4 = 268 MB of HBM reads/writes
   Just for the intermediate attention matrix.
   Per layer. Per head.
```

This is the core problem.

---

## 2. GPU Memory Hierarchy Recap

To understand why this matters, revisit the GPU memory hierarchy:

```
Memory Level    Size        Bandwidth      Latency
──────────────────────────────────────────────────────
Registers       ~256 KB/SM  ~20 TB/s       ~1 cycle
SRAM (Shared)   ~96 KB/SM   ~10 TB/s       ~5 cycles
L2 Cache        ~50 MB      ~5 TB/s        ~100 cycles
HBM (VRAM)      24–80 GB    ~2–3.5 TB/s    ~300–800 cycles
```

The key numbers:

```
SRAM bandwidth:  ~10 TB/s
HBM bandwidth:   ~3.35 TB/s  (H100)

SRAM is ~3× faster in bandwidth.
But more critically: SRAM latency is ~1–5 cycles, HBM is ~300–800 cycles.

Any data that lives in HBM and gets read/written multiple times
wastes enormous amounts of time on latency and bandwidth.
```

### The Critical Insight

The attention matrix `S` and `P` of shape `[N × N]` are:
- Too large to fit in SRAM (96 KB = ~24K float32 values; N=2048 means 4M values)
- Written to HBM after Q × Kᵀ
- Read back from HBM for softmax
- Written again after softmax
- Read again for P × V

**Four HBM round trips for data that is never needed after the final P × V.**

---

## 3. Where Naive Attention Spends Its Time

Here is what actually takes time in naive attention on modern GPUs:

### FLOPs Count

```
Q × Kᵀ:    2 × N² × d  FLOPs
P × V:     2 × N² × d  FLOPs
Softmax:   ~5 × N²      FLOPs  (exp, sum, divide — cheap)

Total:     4 × N² × d + 5 × N²  FLOPs
         ≈ 4 × N² × d FLOPs     (d usually dominates)

For N=2048, d=64:
  4 × 2048² × 64 ≈ 1.07 × 10⁹ FLOPs = ~1 GFLOP per head
```

### HBM Bytes Transferred

```
Read Q, K, V:       3 × N × d × 4 bytes = 3 × 2048 × 64 × 4 = 1.57 MB
Write S:            N² × 4 bytes = 2048² × 4 = 16.8 MB
Read S (softmax):   N² × 4 bytes = 16.8 MB
Write P:            N² × 4 bytes = 16.8 MB
Read P (P×V):       N² × 4 bytes = 16.8 MB
Write O:            N × d × 4 bytes = 0.52 MB

Total HBM bytes:    3 × 0.52 MB + 4 × 16.8 MB ≈ 1.6 MB + 67.2 MB ≈ 69 MB
```

### Compute vs Memory Time

```
H100 compute:  989 TFLOPS (FP16)
H100 HBM BW:   3.35 TB/s

Time for 1 GFLOP: 1×10⁹ / 989×10¹² ≈ 0.001 ms
Time for 69 MB:   69×10⁶ / 3.35×10¹² ≈ 0.021 ms

Memory time is ~20× the compute time.
The GPU spends most of its attention time waiting for HBM,
not computing.
```

This is what "memory bound" means in practice — even though the GPU is extraordinarily powerful, it spends most attention time idle, waiting for data from HBM.

---

## 4. The Roofline Model — Compute Bound vs Memory Bound

The **Roofline Model** is a standard framework for understanding kernel performance.

```
        Peak Compute (TFLOPS)
        │
        │          /‾‾‾‾‾‾‾‾‾‾‾‾‾‾  ← Compute roof (peak FLOPs)
        │         /
        │        /
FLOPs/s │       /   ← Achievable performance
        │      /
        │     /
        │    /   ← Memory roof slope = bandwidth × arithmetic intensity
        │   /
        └──────────────────────────────────────
             Arithmetic Intensity (FLOPs/byte)
```

**Arithmetic Intensity** = FLOPs / bytes of HBM traffic

```
Naive attention arithmetic intensity:
  FLOPs:  4 × N² × d
  Bytes:  ~4 × N² × 4  (dominated by N×N matrix reads/writes)
  
  AI = (4 × N² × d) / (16 × N²) = d / 4

For d=64:  AI = 16 FLOPs/byte
For d=128: AI = 32 FLOPs/byte

H100 "ridge point" (where compute and memory roofs meet):
  ~3.35 TB/s memory BW, ~989 TFLOPs compute
  Ridge point ≈ 989×10¹² / 3.35×10¹² ≈ 295 FLOPs/byte

Naive attention AI (~32) << Ridge point (~295)
→ Naive attention is deeply memory bound
→ Most GPU compute units are idle during attention
```

FlashAttention's goal: increase arithmetic intensity by reusing data in SRAM instead of going to HBM repeatedly.

---

## 5. Memory Bandwidth Is the Real Bottleneck

This is counterintuitive for most people. The GPU has nearly 1000 TFLOPs of compute. Surely compute is the bottleneck?

For attention: no.

```
The softmax kernel that runs between Q×Kᵀ and P×V:
  - Has almost zero FLOPs relative to the matmuls
  - But forces materialization of the N×N matrix in HBM
  - Every element written, then read back
  - At N=4096: 16M elements × 4 bytes = 64 MB per pass

That 64 MB × multiple passes dominates the time budget.

The GPU could compute 10× more FLOPs in the time it takes
to move that 64 MB through HBM bandwidth.
```

**The fix is not to compute faster. The fix is to avoid writing to HBM at all.**

---

# Part 2 — The IO-Aware Insight

---

## 6. What "IO-Aware" Actually Means

The FlashAttention paper's title says "IO-Awareness." This term means the algorithm is designed with explicit knowledge of the memory hierarchy — specifically, minimizing reads and writes to slow HBM.

### Traditional Algorithm Design

Most algorithms are analyzed by counting **FLOPs** — floating point operations. The assumption is that FLOPs are the bottleneck.

```
Traditional thinking:
  Fewer FLOPs = faster algorithm
  
This works for compute-bound kernels.
It fails completely for memory-bound kernels.
```

### IO-Aware Algorithm Design

Instead of counting FLOPs, count **HBM accesses**:

```
IO-aware thinking:
  HBM reads + writes = the actual bottleneck
  Minimize data movement between HBM and SRAM
  Accept MORE FLOPs if it means FEWER HBM accesses
```

FlashAttention does exactly this — it performs slightly more FLOPs than naive attention (due to recomputation in the backward pass) but dramatically fewer HBM reads/writes.

```
Naive attention:   Few FLOPs, many HBM accesses  → slow
FlashAttention:    More FLOPs, few HBM accesses  → fast

Counter-intuitive but correct.
```

### The Three Operations FlashAttention Fuses

Naive attention has three separate GPU kernels:
```
Kernel 1: Q × Kᵀ → S  (matmul)
Kernel 2: softmax(S) → P  (elementwise)
Kernel 3: P × V → O  (matmul)
```

Each kernel launch reads from HBM and writes to HBM. Between kernels, the N×N matrix sits in HBM.

FlashAttention fuses all three into **one kernel**:
```
Single fused kernel: Q, K, V → O
Never materialize S or P in HBM.
```

The N×N intermediate matrix exists only in registers/SRAM and is discarded immediately after use.

---

## 7. Tiling — The Core Strategy

Fusing the three operations requires one key technique: **tiling**.

### The Idea

Instead of computing the full [N × N] attention matrix at once, process it in small tiles that fit in SRAM.

```
Full attention matrix: [N × N]    too large for SRAM
Tile:                  [Br × Bc]  fits in SRAM

Where:
  Br = block/tile size for rows (query dimension)
  Bc = block/tile size for columns (key dimension)
  Chosen so that tiles fit in available SRAM
```

### Tiling Sizes

SRAM capacity per SM: ~96 KB on A100.

For each tile iteration, SRAM must hold:
```
Q tile:   Br × d  elements
K tile:   Bc × d  elements
V tile:   Bc × d  elements
O tile:   Br × d  elements (accumulator)
misc:     statistics (running max, sum)

Total ≈ (2 × Br + 2 × Bc) × d × 4 bytes

For d=64, Br=Bc=64:
  = (128 + 128) × 64 × 4 = 65,536 bytes = 64 KB   ✓ fits in 96 KB
```

### Tiling Execution Pattern

```
For each row tile of Q (i = 0, Br, 2*Br, ...):
    Load Q[i : i+Br] into SRAM
    
    Initialize O_tile = 0
    Initialize running statistics (m = -∞, l = 0)
    
    For each column tile of K, V (j = 0, Bc, 2*Bc, ...):
        Load K[j : j+Bc] into SRAM
        Load V[j : j+Bc] into SRAM
        
        Compute score tile: S_ij = Q_tile × K_tile^T      [Br × Bc]
        Apply mask if needed
        Update running statistics using online softmax
        Update O_tile using this tile's contribution
        
        (K, V tiles evicted from SRAM here — no HBM write)
    
    Write O_tile to HBM    ← only ONE write per Q block, at the end
```

The N×N score matrix `S` is computed tile by tile, used immediately, and discarded. It never touches HBM.

### Why This Works

```
Without tiling:
  N×N matrix → must write to HBM (doesn't fit in SRAM)
  
With tiling:
  Br×Bc tile → stays in SRAM (fits comfortably)
  Used immediately to update running accumulator
  Then overwritten by the next tile
  Never written to HBM
```

The trade-off: instead of processing softmax on the complete row at once, we must compute softmax **incrementally** — one tile at a time. This requires the **online softmax** algorithm.

---

## 8. The Problem Tiling Creates — Softmax

Softmax over a full row is straightforward:

```python
scores = Q[i] @ K.T          # [N] vector, full row
probs  = softmax(scores)      # need the full vector to normalize
output = probs @ V            # weighted sum
```

But with tiling, you don't have the full row. You have one tile at a time:

```
Tile 1: scores[0:64]    ← only see these 64 values
Tile 2: scores[64:128]  ← then these
...

When processing tile 1, you don't know:
  - The maximum of the full row (needed for numerical stability)
  - The normalization sum (needed to normalize probabilities)
```

Standard softmax requires two passes over the full row: find max → compute exp → normalize. With tiling, you can't do two passes — you only load each tile once.

**You need an algorithm that computes exact softmax while seeing only one tile at a time, in a single pass.**

This is the **Online Softmax** algorithm.

---

# Part 3 — Online Softmax

---

## 9. Standard Softmax and Why It Needs the Full Row

### Standard Softmax Formula

```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

For a vector x = [x_0, x_1, ..., x_{N-1}]:

```
Step 1: Compute sum = Σ_j exp(x_j)
Step 2: For each i: output_i = exp(x_i) / sum
```

**Problem:** Step 1 requires seeing all values of x to compute the sum. You can't output any value until you've seen every value.

With tiling, you see values in chunks. After processing tile 1 (values 0–63), you don't know the total sum yet — so you can't normalize.

### Why Not Just Collect All Tiles First?

If you collected all tiles first, you'd have to store the full [N × N] score matrix somewhere. That somewhere is HBM — exactly what we're trying to avoid.

---

## 10. Numerically Stable Softmax — The First Fix

Before covering online softmax, there's a critical subtlety: numerical stability.

### The Overflow Problem

```
x = [1000, 1001, 999]

exp(1000) = 5.07 × 10⁴³²   ← overflows FP32 (max ~3.4 × 10³⁸)
exp(1001) = overflow
exp(999)  = overflow

Standard softmax crashes or returns NaN.
```

### The Stable Softmax Fix

Subtract the maximum before computing exp:

```
m = max(x)

softmax(x)_i = exp(x_i - m) / Σ_j exp(x_j - m)
```

Proof this is mathematically equivalent:

```
exp(x_i - m) / Σ_j exp(x_j - m)
= exp(x_i) × exp(-m) / (Σ_j exp(x_j) × exp(-m))
= exp(x_i) / Σ_j exp(x_j)
= original softmax
```

The `exp(-m)` factors cancel. The result is identical. But now:

```
x_i - m ≤ 0  for all i (since m is the max)
exp(x_i - m) ∈ (0, 1]   ← always safe, no overflow
```

### Two-Pass Stable Softmax

```
Pass 1: Find m = max(x_0, ..., x_{N-1})
Pass 2: For each i:
          compute exp(x_i - m)
          accumulate sum l = Σ exp(x_i - m)
Pass 3: Normalize: output_i = exp(x_i - m) / l
```

Two or three passes over the full vector. Requires storing the full vector or making two memory passes. Still requires seeing the entire row before producing output.

---

## 11. Online Softmax — One Pass, Any Number of Tiles

The online softmax algorithm (Milakov & Gimelshein, 2018; refined by Rabe & Staats, 2021) computes **numerically stable softmax incrementally**, maintaining running statistics that update as new values arrive.

### The Key State

Keep only two scalars:

```
m  = running maximum seen so far
l  = running normalization denominator (sum of exp values)
```

### The Update Rule

When you see a new chunk (tile) of values:

```
m_new = max(m_old, max(new_values))
l_new = exp(m_old - m_new) × l_old + Σ_{new} exp(x_i - m_new)
```

**Meaning of each term:**

```
exp(m_old - m_new) × l_old:
    The old sum l_old was computed with max m_old.
    Now the max increased to m_new.
    We "rescale" the old sum to the new reference maximum.
    (because exp(x_i - m_old) = exp(x_i - m_new) × exp(m_new - m_old))
    So old sum in new reference = l_old × exp(m_old - m_new)

Σ_{new} exp(x_i - m_new):
    Sum of exp values in the new chunk, using updated max.
    Safe since all x_i - m_new ≤ 0.
```

After processing all tiles:

```
m = true maximum of entire row
l = true normalization sum  = Σ_all exp(x_i - m)

softmax(x)_i = exp(x_i - m) / l
```

This is **exact** — not an approximation.

---

## 12. The Running Statistics Trick — Step by Step

Let's trace through a concrete small example.

### Setup

Row of scores: `x = [3, 1, 4, 1, 5, 9, 2, 6]` (8 values)
Tile size: 4 (process in 2 tiles)

True softmax result (we'll verify we get the same):
```
m_true = 9
exp(x - 9) = [exp(-6), exp(-8), exp(-5), exp(-8), exp(-4), exp(0), exp(-7), exp(-3)]
           = [0.0025, 0.0003, 0.0067, 0.0003, 0.0183, 1.0000, 0.0009, 0.0498]
sum = 1.0788
softmax = [0.0023, 0.0003, 0.0062, 0.0003, 0.0170, 0.9270, 0.0009, 0.0461]
```

### Tile 1: x_tile1 = [3, 1, 4, 1]

```
Initialize: m = -∞,  l = 0

m_new = max(-∞, max(3, 1, 4, 1)) = max(-∞, 4) = 4
l_new = exp(-∞ - 4) × 0  +  Σ exp(x_i - 4)
      = 0  +  [exp(-1) + exp(-3) + exp(0) + exp(-3)]
      = 0  +  [0.3679 + 0.0498 + 1.0000 + 0.0498]
      = 1.4675

State after tile 1:  m = 4,  l = 1.4675
```

Also accumulate output:
```
O_tile1 = Σ_j exp(x_j - m_new) × V[j]   (for j in tile 1)
```

### Tile 2: x_tile2 = [5, 9, 2, 6]

```
m_old = 4,  l_old = 1.4675

m_new = max(4, max(5, 9, 2, 6)) = max(4, 9) = 9
l_new = exp(4 - 9) × 1.4675  +  Σ exp(x_i - 9)
      = exp(-5) × 1.4675  +  [exp(-4) + exp(0) + exp(-7) + exp(-3)]
      = 0.00674 × 1.4675  +  [0.0183 + 1.0000 + 0.0009 + 0.0498]
      = 0.009893  +  1.069
      = 1.0789

State after tile 2:  m = 9,  l = 1.0789  ✓ (matches true sum 1.0788)
```

Also **rescale the accumulated output from tile 1**:
```
O = exp(m_old - m_new) × O_tile1 + Σ_j exp(x_j - m_new) × V[j]   (for j in tile 2)
  = exp(-5) × O_tile1  +  (new tile's contribution)
```

The old output was accumulated using max=4. Now we rescale it to max=9.

### Final Normalization

After all tiles:
```
output = O / l
```

This is the exact softmax-weighted sum of V.

---

## 13. Why Online Softmax Is Mathematically Exact

The key claim: online softmax produces the **exact same result** as standard two-pass softmax.

### Proof Sketch

Standard result:

```
softmax_i = exp(x_i - m_true) / l_true
output = Σ_i softmax_i × V_i
       = (1/l_true) × Σ_i exp(x_i - m_true) × V_i
```

Online algorithm produces:

```
O_unnorm = Σ_i exp(x_i - m_true) × V_i    ← same numerator
l_final  = Σ_i exp(x_i - m_true)           ← same denominator

output = O_unnorm / l_final                 ← same result
```

The rescaling step `exp(m_old - m_new) × O_old` ensures that as the running max increases, all previous contributions are correctly adjusted to the new reference point.

Each exp value `exp(x_i - m_true)` contributes exactly once, correctly scaled. The result is bit-for-bit identical to standard softmax (modulo floating point rounding order, which is negligible).

**Online softmax is not an approximation. It is exact.**

This is what makes FlashAttention valid — it's not trading accuracy for speed. It computes the mathematically correct result.

---

# Part 4 — FlashAttention-1

---

## 14. Backstory — The 2022 Stanford Paper

### The Context

The year is 2021–2022. The Transformer has become the dominant architecture. Models are scaling to billions of parameters. Researchers are training on sequences of thousands of tokens.

**The problem everyone knows:** attention is O(N²) in memory. Training on long sequences requires enormous GPU memory, limiting sequence length.

**The common response:** approximate attention — use sparse attention, linear attention, low-rank approximations. Many papers try to speed up attention by changing the math (and accepting some approximation error).

**The key insight Tri Dao and team had:** Everyone was trying to make attention faster by reducing FLOPs. But attention isn't slow because of FLOPs. It's slow because of HBM bandwidth. The N×N matrix is the problem — not because of the FLOPs to compute it, but because of the bandwidth cost to write/read it.

**The question they asked:** Can we compute **exact** attention without ever materializing the N×N matrix in HBM?

The answer became **FlashAttention** (Dao, Fu, Ermon, Rudra, Ré — Stanford, 2022).

### Why It Took Until 2022

Several things had to be true:
1. Sequences long enough that N² memory mattered (becoming true ~2020–2021)
2. Hardware fast enough that memory bandwidth was the real bottleneck (A100 made this clear)
3. The online softmax algorithm existing in the literature (Milakov & Gimelshein, 2018)
4. Someone thinking about the problem from a systems angle instead of an algorithms angle

Tri Dao (PhD student at Stanford) came from a background that combined both deep learning and systems/hardware. This cross-disciplinary view was what made the insight possible.

---

## 15. The FlashAttention-1 Algorithm

### High-Level Design

```
Goal:
  Compute Attention(Q, K, V) = softmax(Q × Kᵀ / √d) × V
  Without writing the N×N intermediate matrix to HBM

Strategy:
  1. Load Q, K, V tiles into SRAM
  2. Compute partial attention scores in SRAM
  3. Use online softmax to accumulate the result incrementally
  4. Only write the final output O to HBM

Result:
  O (the attention output) is the only HBM write
  S and P never touch HBM
```

### SRAM Requirements

For tile sizes Br (query rows) × Bc (key/value columns):

```
Must fit in SRAM simultaneously:
  Q tile:   Br × d floats
  K tile:   Bc × d floats
  V tile:   Bc × d floats
  O tile:   Br × d floats (accumulator)
  l, m:     Br floats (statistics vectors)
  S tile:   Br × Bc floats (temporary score tile)

Total ≈ (4 × Br + 2 × Bc) × d + Br × Bc floats

Constraint: fits within SRAM size M
```

Solving for optimal Br, Bc given SRAM size M.

---

## 16. FlashAttention-1 Forward Pass — Full Pseudocode

```
Input:  Q [N×d], K [N×d], V [N×d] in HBM
        Block sizes Br, Bc (chosen to fit in SRAM)
Output: O [N×d] in HBM

Divide Q into Tr = ceil(N/Br) row blocks: Q_1, ..., Q_Tr
Divide K, V into Tc = ceil(N/Bc) column blocks: K_1, ..., K_Tc, V_1, ..., V_Tc

Allocate O [N×d] in HBM  (output, initially zeros)
Allocate l [N]   in HBM  (normalization sums, initially zeros)
Allocate m [N]   in HBM  (row maxima, initially -∞)

─────────────────────────────────────────────────────────────────
OUTER LOOP: iterate over column tiles (K, V)
─────────────────────────────────────────────────────────────────
for j = 1 to Tc:
    Load K_j [Bc×d] from HBM → SRAM
    Load V_j [Bc×d] from HBM → SRAM
    
    ─────────────────────────────────────────────────────────────
    INNER LOOP: iterate over row tiles (Q, O)
    ─────────────────────────────────────────────────────────────
    for i = 1 to Tr:
        Load Q_i [Br×d]  from HBM → SRAM
        Load O_i [Br×d]  from HBM → SRAM   (current accumulated output)
        Load l_i [Br]    from HBM → SRAM   (current norm sums)
        Load m_i [Br]    from HBM → SRAM   (current row maxima)
        
        ── Compute score tile ──
        S_ij = Q_i × K_j^T / √d         [Br × Bc]   (in SRAM)
        Apply causal mask to S_ij if needed
        
        ── Online softmax update ──
        m_ij  = rowmax(S_ij)             [Br]        (new tile's max)
        m_new = max(m_i, m_ij)           [Br]        (updated running max)
        
        P_ij  = exp(S_ij - m_new)        [Br × Bc]   (tile softmax numerator)
        l_ij  = rowsum(P_ij)             [Br]        (tile sum)
        l_new = exp(m_i - m_new) × l_i  (rescale old sum)
              + l_ij                     (add new tile's contribution)
        
        ── Update output accumulator ──
        O_new = diag(exp(m_i - m_new)) × O_i    (rescale old output)
              + P_ij × V_j                       (add new tile's contribution)
        
        ── Write updated state back to HBM ──
        Write O_new → O_i in HBM
        Write l_new → l_i in HBM
        Write m_new → m_i in HBM

─────────────────────────────────────────────────────────────────
FINAL NORMALIZATION
─────────────────────────────────────────────────────────────────
for i = 1 to Tr:
    Load O_i, l_i from HBM
    O_i = O_i / l_i      (divide by normalization sum)
    Write O_i to HBM

Return O
```

### The Key Loop Order Decision

Note: FA-1 iterates **outer over j (K, V blocks), inner over i (Q blocks)**.

This means:
- For each K, V block: load it once, use it for ALL query blocks
- But: each Q block is loaded Tc times (once per K, V block)
- And: each O block is written/read Tc times (to update accumulator)

This loop order minimizes K, V loads (loaded once each). But causes O to be read/written many times.

This choice has implications for parallelism — addressed in FA-2.

---

## 17. The Backward Pass Problem — Recomputation

### Training Requires the Backward Pass

During training (but not inference), you need gradients:

```
∂Loss/∂Q = ∂Loss/∂O × (∂O/∂Q)
∂Loss/∂K = ∂Loss/∂O × (∂O/∂K)
∂Loss/∂V = ∂Loss/∂O × (∂O/∂V)
```

Computing these gradients requires the attention weights P = softmax(S). Standard backprop: store P during forward pass, load it during backward.

**But FlashAttention's whole point is to NOT store P.**

### FA-1's Solution: Recomputation

Instead of storing the N×N attention matrix P during the forward pass, **recompute it during the backward pass**.

```
Forward pass:
  Compute P tile by tile in SRAM
  DO NOT store P to HBM
  DO store: O, l (normalization sums), m (row maxima)
  → O(N) memory instead of O(N²)

Backward pass:
  Load O, l, m from HBM
  Recompute P tile by tile (same tiling algorithm)
  Use recomputed P to compute gradients for Q, K, V
```

### Recomputation Trick

Given the stored (m, l), you can reconstruct any tile of P:

```
P_ij = exp(S_ij - m_i) / l_i

where S_ij is recomputed from Q_i × K_j^T (tiles loaded from HBM)
```

You can recompute any tile of P on the fly, without ever storing the full N×N matrix.

### The Cost of Recomputation

```
Extra FLOPs from recomputation:
  Recompute S_ij = Q_i × K_j^T during backward: same cost as forward
  Total backward FLOPs ≈ 2× more than naive backward
  But: still much faster because of reduced HBM traffic

Naive backward:
  Stores P [N×N] → read during backward
  HBM traffic: O(N²)

FA-1 backward:
  Recomputes S tiles → no P in HBM
  HBM traffic: O(N × d)  (just Q, K, V, O)
```

More FLOPs, less memory traffic, net win.

### Connection to Gradient Checkpointing

This recomputation is the same idea as gradient checkpointing — trade compute for memory. FA-1 applies it specifically and very efficiently within the attention kernel, much more efficiently than naive checkpointing.

---

## 18. FlashAttention-1 Memory Analysis

### Memory for the Attention Step

```
Naive attention:
  Must store S [N×N] and P [N×N] in HBM (for backward)
  Memory: O(N²)   (dominates everything else)

FlashAttention-1:
  Stores: O [N×d], l [N], m [N]
  Memory: O(N × d)   (linear in sequence length!)

For N=2048, d=64:
  Naive:  N² = 4M floats = 16 MB  (per head per layer)
  FA-1:   N×d = 131K floats = 0.5 MB  (32× smaller!)

For N=16384, d=64:
  Naive:  N² = 268M floats = 1 GB  (per head per layer — enormous!)
  FA-1:   N×d = 1M floats = 4 MB  (250× smaller!)
```

This is what enables training on much longer sequences without running out of VRAM.

---

## 19. FlashAttention-1 IO Complexity Analysis

The paper formally analyzes HBM reads/writes as a function of N, d, and SRAM size M.

### Naive Attention HBM IOs

```
Write S = Q × Kᵀ:          Θ(N²)
Read S, write P = softmax:  Θ(N²)
Read P, write O = P × V:    Θ(N²)
Total:                      Θ(N²)
```

### FlashAttention-1 HBM IOs

```
Read Q, K, V:                    Θ(N × d)
Write O:                         Θ(N × d)
Read/write O, l, m in tile loop: Θ(N² × d / M)

Where M = SRAM size.

Total: Θ(N × d + N² × d / M)
```

### When Is FA-1 Faster?

```
FA-1 total IO: Θ(N² × d / M)
Naive IO:      Θ(N²)

FA-1 < Naive when:  N² × d / M < N²
                    d / M < 1
                    d < M

Since d (head dim) is typically 64–128
and M (SRAM) is ~100KB = 25,000 float32
→ d << M always holds

FA-1 reduces IO by factor ~M/d.

For M=100KB, d=64:
  M/d ≈ 25,000/64 ≈ 390×  potential IO reduction
```

In practice (not full theoretical max), FA-1 achieves ~2–4× speedup because:
- Other operations (projections, FFN) take significant time
- Kernel launch overhead
- Not always perfectly memory-bound

---

## 20. FlashAttention-1 Benchmarks and Results

From the original paper (A100 GPU, FP16):

### Attention Runtime vs Sequence Length

```
Sequence Length    Naive Attention    FlashAttention-1    Speedup
───────────────────────────────────────────────────────────────────
512                ~0.3 ms            ~0.2 ms             1.5×
1024               ~0.8 ms            ~0.4 ms             2.0×
2048               ~2.5 ms            ~0.7 ms             3.6×
4096               ~9.0 ms            ~1.5 ms             6.0×
8192               OOM                ~3.1 ms             ∞ (enables it)
```

Speedup grows with sequence length — as N grows, the N² memory traffic of naive attention becomes increasingly dominant.

### Memory Usage

```
Sequence Length    Naive Memory    FlashAttention-1 Memory
──────────────────────────────────────────────────────────
1024               1.2 GB          0.006 GB    (200× smaller)
2048               4.8 GB          0.012 GB    (400× smaller)
4096               19.2 GB         0.025 GB    (768× smaller)
```

### End-to-End Training Speedup (BERT)

```
BERT-large (sequence len 512):      FlashAttention 15% faster
GPT-2 (sequence len 1024):         FlashAttention 3× faster

Long Range Arena benchmark (seq 4096):  FlashAttention enables training
                                          that was impossible with naive
```

### GPU Utilization

```
Naive attention on A100:    ~25–35% GPU utilization (memory bound)
FlashAttention on A100:     ~50–60% GPU utilization (improved)
Still not near peak — FA-2 improves this further.
```

---

# Part 5 — FlashAttention-2

---

## 21. What FA-1 Left on the Table

FlashAttention-1 was a major breakthrough in memory efficiency but achieved only ~30–50% of A100's theoretical peak throughput (FLOPs/s).

The paper identified remaining inefficiencies:

### Problem 1: Too Many Non-Matmul FLOPs

Tensor Cores are specialized for matrix multiplication. They run at peak throughput only for GEMM (general matrix multiply) operations.

Other operations (exp, addition, max, division) run on regular CUDA cores at much lower throughput:

```
A100 FP16 matmul throughput:    312 TFLOPS
A100 FP16 non-matmul throughput: ~20 TFLOPS

If 10% of your FLOPs are non-matmul:
  Effective throughput ≈ 0.9 × 312 + 0.1 × 20 = 283 + 2 = 285 TFLOPs
  (only minor impact)

But if 30% are non-matmul:
  Effective throughput ≈ 0.7 × 312 + 0.3 × 20 = 218 + 6 = 224 TFLOPs
  (significant impact)
```

FA-1 had a relatively high ratio of non-matmul work in its inner loop (the online softmax updates, rescaling operations). FA-2 restructured the algorithm to minimize this.

### Problem 2: Limited Parallelism

FA-1 parallelized attention across:
- Batch dimension
- Head dimension

But NOT across the sequence length dimension.

For small batch sizes (e.g., inference with batch=1) or few heads, many SMs would sit idle because there wasn't enough work to distribute.

### Problem 3: Suboptimal Work Partitioning

Within each SM, FA-1 divided work across warps suboptimally, causing some warps to wait on others (warp serialization), reducing effective utilization.

---

## 22. FA-2 Improvement 1 — Fewer Non-Matmul FLOPs

### What FA-1 Did in the Inner Loop

FA-1's inner loop (per tile):

```
S = Q_i × K_j^T / √d             ← matmul (Tensor Core)
m_new = max(m_old, rowmax(S))     ← reduction, non-matmul
P = exp(S - m_new)                ← elementwise, non-matmul
l_new = exp(m_old - m_new) × l   ← rescale + add, non-matmul
      + rowsum(P)
O_new = exp(m_old - m_new) × O   ← rescale, non-matmul
      + P × V_j                   ← matmul (Tensor Core)
```

The rescaling operations `exp(m_old - m_new) × O` and `exp(m_old - m_new) × l` happen every tile, adding non-matmul work.

### FA-2's Restructuring

FA-2 defers the rescaling:

```
Key insight: Don't rescale O every tile.
Keep O in "un-normalized" form.
Only rescale once at the very end.

Inner loop now:
  S = Q_i × K_j^T / √d             ← matmul
  m_new = max(m_old, rowmax(S))     ← small, non-matmul
  P = exp(S - m_new)                ← elementwise, non-matmul
  l_new = exp(m_old - m_new) × l   ← scalar rescale, non-matmul
        + rowsum(P)
  O_accum += P × V_j                ← matmul (NO rescale of O here!)

After all j tiles:
  O_final = O_accum × diag(1/l)    ← one final rescale
```

By moving the O rescaling out of the inner loop:
- Fewer non-matmul ops per tile
- More time on Tensor Cores (matmuls)
- Better Tensor Core utilization

### Quantified Improvement

```
FA-1 non-matmul FLOPs fraction: ~20–30% of total
FA-2 non-matmul FLOPs fraction: ~10–15% of total

Result: better effective throughput on hardware
```

---

## 23. FA-2 Improvement 2 — Parallelism Across Sequence Dimension

### FA-1 Parallelism

FA-1 assigns one thread block per attention head per batch item:

```
Total thread blocks = batch_size × num_heads

For batch=1, 32 heads, A100 (108 SMs):
  32 thread blocks → only 32 SMs used (30% of GPU!)
  76 SMs sit idle!
```

This is catastrophic for inference (batch=1) or any setting with few heads/small batch.

### FA-2's Solution: Sequence Parallelism

FA-2 adds parallelism across the sequence dimension:

```
Total thread blocks = batch_size × num_heads × (N / Br)

For batch=1, 32 heads, N=2048, Br=64:
  1 × 32 × 32 = 1024 thread blocks → all 108 SMs easily occupied
```

### The Complication

You can't simply split the sequence dimension arbitrarily — different tiles of the sequence are interdependent through the softmax normalization.

FA-2's solution: Different thread blocks handle different Q tiles independently. Each block computes a partial output for its Q rows, with its own running (m, l) statistics. The partial outputs are combined at the end using the log-sum-exp trick:

```
Block A processed Q[0:Br]   → partial output O_A, statistics (m_A, l_A)
Block B processed Q[Br:2Br] → these are independent rows, no combination needed!

Wait — rows of Q are independent! Each query row i attends to all K, V rows.
The dependency is only in the K, V (column) dimension.

→ Q rows can be trivially parallelized.
→ Column (K, V) dimension requires accumulation (the online softmax loop).
```

FA-2 parallelizes across Q rows. Each SM handles a different subset of Q rows and runs the full K, V loop for those rows. No communication needed between SMs.

### Impact

```
Before FA-2 (batch=1, 32 heads, A100):
  GPU utilization: ~30% (only 32 of 108 SMs used)

After FA-2 (same settings):
  GPU utilization: ~70–80% (all SMs used via sequence parallelism)

Speedup for small batch / inference: up to 2–3×
```

---

## 24. FA-2 Improvement 3 — Better Work Partitioning Within an SM

### The Warp Serialization Problem in FA-1

Inside each SM, FA-1 divided the Br × Bc score tile across warps such that:

```
Warp 0: compute rows 0–7 of S tile
Warp 1: compute rows 8–15 of S tile
...

Then for the online softmax update:
  All warps need to share m and l statistics
  → All-reduce across warps (expensive synchronization)
  → Some warps stall while waiting for others
```

This intra-SM synchronization was a hidden overhead.

### FA-2's Solution: Warp-Independent Work

FA-2 restructures so that each warp handles a complete independent chunk of work:

```
Instead of splitting rows across warps (requiring sync on m, l):
  Each warp handles a full Q chunk × all K, V chunks
  Each warp maintains its own (m, l) statistics
  No cross-warp synchronization needed for the main loop
```

This eliminates warp serialization for the inner loop, achieving better SM utilization.

---

## 25. FA-2 Improvement 4 — Native GQA/MQA Support

FA-1 was designed for standard MHA. Supporting GQA/MQA required external tricks (like repeating K/V heads in memory before calling FA-1), which defeated the memory savings of GQA.

### FA-2's Approach

FA-2 natively handles cases where `num_kv_heads < num_q_heads`:

```
Standard MHA: 1 K/V head per Q head → 1:1 ratio
GQA (G=8):    1 K/V head per 4 Q heads → 4:1 ratio
MQA:          1 K/V head for all Q heads → H:1 ratio

FA-2 broadcasts K, V tiles to all Q heads in the group
without expanding in memory.

Within the kernel:
  Load K_j tile once [Bc × d]
  Use it for all 4 Q heads in the group
  No H× expansion of K, V in HBM
```

This means GQA + FA-2 gets the full benefit of both:
- GQA's H/G× memory reduction (smaller KV cache)
- FA-2's bandwidth efficiency (no N×N materialization)

---

## 26. FlashAttention-2 Algorithm — What Changed

Summary of algorithmic changes vs FA-1:

```
FA-1:
  Outer loop over j (K, V columns)
  Inner loop over i (Q rows)
  Rescale O every tile

FA-2:
  Outer loop over i (Q rows)    ← SWAPPED
  Inner loop over j (K, V columns)
  Rescale O only once at end    ← fewer non-matmul ops
  Parallelism across i (Q rows) ← new parallelism dimension
```

The loop order swap is important: FA-2's outer loop over Q rows means each thread block owns its Q rows completely, making parallelism across the sequence dimension natural.

### FA-2 Forward Pass (Simplified)

```
Parallelize across: batch × heads × Q_blocks

For each Q block i (in parallel across SMs):
    Load Q_i into SRAM
    Initialize: O_i = 0, l_i = 0, m_i = -∞
    
    For j = 1 to Tc:  (inner loop over K, V blocks)
        Load K_j, V_j into SRAM
        
        S = Q_i × K_j^T / √d     ← matmul
        m_ij = rowmax(S)
        m_new = max(m_i, m_ij)
        
        P = exp(S - m_new)
        l_new = exp(m_i - m_new) × l_i + rowsum(P)
        
        O_i += P × V_j            ← matmul (no rescale of O_i here!)
        
        l_i = l_new
        m_i = m_new
    
    O_i = O_i / l_i              ← single normalization at end
    Write O_i to HBM
```

---

## 27. FlashAttention-2 Benchmarks and Results

From the FA-2 paper (A100 80GB SXM5, FP16):

### Throughput vs Sequence Length (causal attention)

```
Sequence Length    FA-1          FA-2          Peak BW %
────────────────────────────────────────────────────────
1024               ~50 TFLOPs   ~100 TFLOPs   ~33%
2048               ~80 TFLOPs   ~170 TFLOPs   ~56%
4096               ~100 TFLOPs  ~200 TFLOPs   ~66%
8192               ~110 TFLOPs  ~230 TFLOPs   ~73%
16384              ~115 TFLOPs  ~240 TFLOPs   ~76%

Peak theoretical: ~312 TFLOPs (A100 FP16)
```

FA-2 roughly doubles FA-1's throughput.

### Speed vs xFormers (Meta's attention library)

```
FA-2 is 1.3–2.5× faster than xFormers
on A100 across all sequence lengths.
```

### Training Speedup (end-to-end)

```
GPT-3 style 6.7B model:
  Without FA-2:   ~150 tokens/sec/GPU
  With FA-2:      ~225 tokens/sec/GPU
  Speedup: ~1.5×

Improvement comes from attention layers only;
FFN and other layers unchanged.
```

### Memory Efficiency

Same as FA-1 (O(N) instead of O(N²)) — FA-2 doesn't change the memory complexity, only the compute efficiency.

---

# Part 6 — FlashAttention-3

---

## 28. The H100 Hopper Architecture — New Hardware Features

FlashAttention-3 targets the NVIDIA H100 (Hopper architecture, released 2022–2023). To understand FA-3, you must understand what new hardware features Hopper introduced.

### Feature 1: Tensor Memory Accelerator (TMA)

The TMA is a dedicated hardware unit for memory transfers — separate from the compute units.

```
Without TMA (Ampere / A100):
  Compute warps handle both data loading AND computation
  Loading data stalls computation
  
  Timeline:
  Warp: [Load tile] [Compute] [Load tile] [Compute]
         ──────────  ────────  ──────────  ────────
         memory      compute   memory      compute
         (stall)               (stall)

With TMA (Hopper / H100):
  TMA hardware handles ALL data loading independently
  Compute warps only do computation
  
  Timeline:
  TMA:   [Load tile 1]  [Load tile 2]  [Load tile 3]
  Warp:           [Compute tile 1]  [Compute tile 2]
                  ────────────────  ────────────────
                  fully overlapped!
```

TMA enables true overlap of memory transfer and computation.

### Feature 2: Warp Group Matrix Multiply Accumulate (WGMMA)

A new asynchronous matrix multiply instruction:

```
Old (Ampere): mma.sync — synchronous, warp must wait for result
New (Hopper): wgmma   — asynchronous, instruction issued and execution overlaps

4 warps form a "warp group" and cooperate on a single large matmul
Output goes to shared registers across the warp group
Higher throughput than issuing individual mma.sync instructions
```

### Feature 3: FP8 Tensor Cores

H100 adds native FP8 (8-bit floating point) matrix multiply support:

```
A100 FP16 throughput:  312 TFLOPs
H100 FP16 throughput:  989 TFLOPs (3× faster)
H100 FP8 throughput:   1978 TFLOPs (2× faster than FP16)
```

FP8 attention enables near-2× more throughput if numerical precision allows.

### Feature 4: Shared Memory Async Copy

H100 supports issuing async shared memory loads — the hardware fetches data into SRAM while the compute units run. Eliminates the "load, then compute" sequential pattern.

---

## 29. FA-3 Improvement 1 — Asynchronous TMA Copies

### The Problem FA-2 Had (Even on H100)

FA-2 was designed for A100 (Ampere) and didn't use H100's TMA. On H100, compute warps were still responsible for loading data, leading to:

```
Warp does:
  1. Issue load for K tile         ← warp stalls
  2. Wait for K tile to arrive     ← warp stalled
  3. Compute Q × K^T               ← warp active
  4. Issue load for V tile         ← warp stalls
  5. Wait for V tile               ← warp stalled
  6. Compute P × V                 ← warp active

Effective utilization: ~50% (stalled half the time)
```

### FA-3's TMA Solution

FA-3 uses TMA to decouple memory from compute completely:

```
TMA (hardware unit):
  Issues async load for K tile j
  → Tile lands in SRAM when ready
  → Signals compute warp via semaphore

Compute warp:
  While TMA loads K tile j:
    Compute with previously loaded tile j-1
  When semaphore fires (K tile j ready):
    Switch to computing with tile j
  Issue async load for tile j+1 immediately

Timeline:
  TMA:    [K0 load][K1 load][K2 load][K3 load]
  Warp:          [K0 calc][K1 calc][K2 calc][K3 calc]
                 ← fully overlapped ←
```

### Prefetching

FA-3 prefetches tiles 1–2 steps ahead:

```
Iteration j:
  1. Issue TMA load for K_{j+1}, V_{j+1}  (prefetch next)
  2. Wait for K_j, V_j  (already prefetched from last iteration)
  3. Compute S = Q × K_j^T
  4. Compute online softmax
  5. Compute O += P × V_j
  6. Loop
```

By the time you need K_j, it's already in SRAM from the previous iteration's prefetch.

---

## 30. FA-3 Improvement 2 — Warp Specialization

### The Idea

Not all warps do the same thing. FA-3 assigns different roles to different warps:

```
Producer warps:  Only issue TMA loads, manage data staging
Consumer warps:  Only compute (matmul, softmax updates)

They communicate via semaphores and circular buffers in SRAM.
```

```
SRAM layout with warp specialization:

  ┌──────────────────────────────────────────┐
  │ Double buffer for K tiles                │
  │   Buffer A: [Bc × d] ← TMA writing      │
  │   Buffer B: [Bc × d] ← Compute reading  │
  │   (swap roles each iteration)            │
  ├──────────────────────────────────────────┤
  │ Double buffer for V tiles                │
  │   Same pattern                           │
  ├──────────────────────────────────────────┤
  │ Q tile (loaded once per Q block)         │
  ├──────────────────────────────────────────┤
  │ Output accumulator O                     │
  │ Statistics m, l                          │
  └──────────────────────────────────────────┘
```

### Why This Helps

```
Without warp specialization:
  All warps do load + compute
  When loading: compute units idle
  When computing: memory bandwidth unused

With warp specialization:
  Producer warps keep memory pipeline full
  Consumer warps keep compute pipeline full
  Neither waits for the other (in steady state)
  
  Theoretical: 100% of both memory BW and compute
  Practical: ~80–90% of both
```

---

## 31. FA-3 Improvement 3 — GEMM and Softmax Overlap

### The Problem

Even with TMA handling memory, there's still serialization within the compute side:

```
FA-2 compute pattern (serial):
  Step 1: S = Q × K_j^T    [GEMM on Tensor Cores]    ← matmul
  Step 2: m, l update        [non-matmul on CUDA cores] ← softmax update
  Step 3: O += P × V_j      [GEMM on Tensor Cores]    ← matmul
  Step 4: m, l update        [non-matmul on CUDA cores]
  
  Tensor Cores idle during steps 2, 4.
  CUDA cores idle during steps 1, 3.
```

On H100, Tensor Cores and CUDA cores are physically separate — they can operate in parallel if scheduled correctly.

### FA-3's Overlap

FA-3 uses WGMMA's asynchronous nature to overlap the matmul with the softmax update:

```
FA-3 compute pattern (overlapped):

  Iteration j:
    Issue WGMMA for S = Q × K_j^T  (async, runs on Tensor Cores)
    While S computes:
      Update softmax stats for previous iteration's S_{j-1}
      (runs on CUDA cores — Tensor Cores busy with S)
    WGMMA result arrives
    Issue WGMMA for O += P × V_j   (async)
    While O updates:
      Compute rescaling for this tile's stats
      (CUDA cores busy — Tensor Cores computing)

  Result: Tensor Cores and CUDA cores active simultaneously
  → Higher effective throughput
```

### What Makes This Possible

WGMMA (Hopper's matrix multiply) is **asynchronous** — the instruction returns immediately while the Tensor Core computes in the background. The programmer explicitly synchronizes only when the result is needed. This gap is when other work runs.

---

## 32. FA-3 Improvement 4 — FP8 Tensor Core Support

### FP8 Formats

H100 supports two FP8 formats:

```
E4M3: 4 exponent bits, 3 mantissa bits → larger range, less precision
      Max value: 448
      Good for: weights, activations (need range more than precision)

E5M2: 5 exponent bits, 2 mantissa bits → less range, even less precision
      Max value: 57344
      Good for: gradients (need range for large gradient values)
```

### The Quantization Challenge in Attention

FP8 is trivial for dense matmuls (Q × K^T in linear layers). But attention has a unique challenge:

```
Problem: Attention scores (Q × K^T) can have very different ranges
per row. Different rows of Q × K^T might have:
  Row 0: values in [−5, 5]
  Row 1: values in [−50, 50]
  Row 2: values in [−0.1, 0.1]

A single global scale factor quantizing all rows uniformly
will cause overflow in row 1 or underflow in row 2.
```

### FA-3's FP8 Approach

FA-3 uses a block-level quantization approach:

```
For each tile S_ij of shape [Br × Bc]:
  Compute max absolute value in tile: scale = max(|S_ij|) / FP8_max
  Quantize tile: S_ij_fp8 = S_ij / scale
  Compute with FP8 Tensor Cores
  Dequantize result: O_tile_fp32 = result * scale
```

This is done tile-by-tile, exploiting the SRAM tiling structure that FlashAttention already uses.

### FP8 Throughput

```
H100 FP16 GEMM: 989 TFLOPs
H100 FP8 GEMM:  1978 TFLOPs

FA-3 with FP8 vs FA-2 with FP16:
  ~2× higher throughput (for compute-intensive scenarios)
```

FP8 in attention is still maturing — there are numerical precision concerns for training that don't apply to inference. FA-3's FP8 mode is primarily used for inference.

---

## 33. FA-3 Improvement 5 — Incoherent Processing

This is a subtle but important technique for FP8 numerical stability.

### The Problem: Outliers in Attention

Attention score matrices often have **outlier values** — a few scores much larger than the rest. In FP8, these outliers force a conservative scale factor that wastes precision for typical values:

```
Row scores: [0.1, 0.2, 0.3, 0.15, 50.0, 0.2, ...]

FP8 max = 448 (E4M3)
Scale = 50.0 / 448 ≈ 0.112

Quantized values: [0.9, 1.8, 2.7, 1.3, 447.5, 1.8, ...]
                   OK   OK   OK   OK   OK     OK

But precision lost: all small values crushed into small int codes
```

### The Incoherent Processing Trick

Apply a random orthogonal matrix transformation to reduce outlier magnitude while preserving the final attention result:

```
Instead of computing:  softmax(Q × K^T) × V

Compute:               softmax((Q×R) × (K×R)^T) × V
                     = softmax(Q × R×R^T × K^T) × V
                     = softmax(Q × K^T) × V   ← same result (R×R^T = I)
```

Where R is a random orthogonal matrix (Hadamard matrix works well — fast to apply).

The random rotation "spreads" outliers across dimensions, reducing the max absolute value in any particular tile, enabling more efficient FP8 quantization.

This technique comes from quantization literature (e.g., QuIP#, SpinQuant) applied innovatively to attention kernels.

---

## 34. FlashAttention-3 Benchmarks and Results

From the FA-3 paper (H100 80GB SXM, August 2024):

### Attention Forward Pass Throughput

```
Method              FP16 TFLOPs    % Peak
────────────────────────────────────────────
Naive Attention     ~100           ~10%
FA-2                ~380           ~38%
FA-3 (FP16)         ~740           ~75%
FA-3 (FP8)          ~1200          ~60% of FP8 peak

H100 peak FP16: 989 TFLOPs
H100 peak FP8:  1978 TFLOPs
```

### Speed Comparison (seq_len = 4096, head_dim = 128)

```
                 Forward    Forward+Backward
FA-2:             ~1.5×       ~1.3×
FA-3:             ~2.0×       ~1.8×
(vs FA-2)
```

FA-3 is roughly 2× faster than FA-2 on H100.

### End-to-End Training Impact

```
LLM training on H100 (forward + backward, including FFN, embeddings):
  FA-2:  Attention portion ~20% of time
  FA-3:  Attention portion ~12% of time

Overall training speedup: 10–15% reduction in total training time.
```

Note: FA-3 only speeds up attention kernels, which are one part of total training. The overall speedup depends on what fraction of training time is attention.

---

# Part 7 — Putting It All Together

---

## 35. FA-1 vs FA-2 vs FA-3 — Full Comparison

| Feature | Naive | FA-1 | FA-2 | FA-3 |
|---------|-------|------|------|------|
| N×N matrix in HBM | Yes (train+infer) | No | No | No |
| Memory complexity | O(N²) | O(N) | O(N) | O(N) |
| HBM IO complexity | O(N²) | O(N²d/M) | O(N²d/M) | O(N²d/M) |
| Parallelism axes | Batch, Head | Batch, Head | Batch, Head, **Seq** | Batch, Head, Seq |
| Non-matmul FLOPs | Low | Medium | **Low** | Low |
| GQA/MQA native | No | No | **Yes** | Yes |
| TMA async loads | No | No | No | **Yes** |
| Warp specialization | No | No | No | **Yes** |
| GEMM+softmax overlap | No | No | No | **Yes** |
| FP8 support | No | No | No | **Yes** |
| Incoherent processing | No | No | No | **Yes** |
| Target hardware | Any | A100 | A100 | **H100** |
| Speedup vs naive | 1× | 2–4× | 4–8× | 8–16× |
| H100 FP16 utilization | ~10% | ~35% | ~50–75% | ~75% |

---

## 36. How to Use FlashAttention in Practice

### PyTorch 2.0+ (Automatic)

PyTorch 2.0+ includes FlashAttention via `scaled_dot_product_attention`:

```python
import torch
import torch.nn.functional as F

# Standard usage — PyTorch picks best kernel automatically
output = F.scaled_dot_product_attention(
    query,    # [batch, heads, seq, head_dim]
    key,      # [batch, kv_heads, seq, head_dim]  (GQA: kv_heads < heads)
    value,    # [batch, kv_heads, seq, head_dim]
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True   # for decoder models
)

# Check which backend is being used
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,    # FlashAttention
    enable_math=False,    # Naive implementation
    enable_mem_efficient=False  # xFormers-style
):
    output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
```

### HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM
import torch

# FA-2 (via PyTorch SDPA)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"  # uses PyTorch's SDPA with FA-2
)

# FA-2 via flash_attn package (often faster)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
```

### Installing flash-attn Package (Tri Dao's implementation)

```bash
pip install flash-attn --no-build-isolation
# Requires CUDA 11.6+ and PyTorch 1.12+
# Compiles from source — takes ~5–10 minutes

# For FA-3 (H100 only, as of 2024):
pip install flash-attn-3  # separate package, still evolving
```

### Direct Usage (flash_attn package)

```python
from flash_attn import flash_attn_func, flash_attn_varlen_func

# Standard usage
output = flash_attn_func(
    q,           # [batch, seq, heads, head_dim]   note: different shape from PyTorch!
    k,           # [batch, seq, kv_heads, head_dim]
    v,           # [batch, seq, kv_heads, head_dim]
    dropout_p=0.0,
    causal=True,
    softmax_scale=None  # defaults to 1/sqrt(head_dim)
)

# Variable length sequences (for batches with different lengths)
output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q,   # cumulative sequence lengths
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal=True
)
```

### When Does FlashAttention Help Most?

```
Sequence length < 512:    Minimal benefit (N² not large enough)
Sequence length 512–2K:   Moderate speedup (2–3×)
Sequence length 2K–16K:   Large speedup (3–6×)
Sequence length 16K+:     Enables what was otherwise impossible

Batch size 1 (inference):  FA-2 especially important (sequence parallelism)
Large batch (training):    FA-1 already helpful, FA-2 adds more
```

---

## 37. What FlashAttention Does Not Solve

### 1. The O(N²) FLOPs Problem

FlashAttention dramatically reduces **memory** and **IO** — it makes attention fast enough to run.

But it doesn't reduce the **FLOPs** needed:

```
Attention FLOPs = O(N²)   — unchanged by FlashAttention
Memory          = O(N)    — FA's main contribution

For N=1M (1 million token context):
  FLOPs: still astronomical — FlashAttention can't fix this
  The compute alone at N=1M is infeasible regardless of memory
```

For truly long contexts, you need architectures that change the O(N²) compute: sparse attention, linear attention, sliding window, state space models (Mamba), etc.

### 2. KV Cache at Inference

FlashAttention speeds up the **prefill** (processing the prompt) dramatically.

But during **token-by-token generation**:
```
Each step generates 1 token.
Attention at each step: 1 query × N past keys
Computation: O(N) per step, not O(N²)
KV cache still grows as O(N) per step

FlashAttention still helps here (avoids materializing the full score vector),
but the dominant cost at long contexts is loading the KV cache from HBM —
which is unrelated to the N×N intermediate matrix problem FA solves.
```

GQA/MQA + PagedAttention address the KV cache problem.

### 3. Communication in Distributed Settings

For multi-GPU training with tensor parallelism, attention keys/values must be communicated between GPUs. FlashAttention operates within a single GPU — it doesn't address cross-GPU communication. Ring Attention and related work addresses this.

### 4. Non-Standard Attention Patterns

FlashAttention is designed for standard dense attention. Some variants are harder to fuse efficiently:
- Attention with complex custom masks (not just causal)
- Cross-attention with different sequence lengths (supported but more complex)
- Linear attention variants (different algorithm entirely)

---

## 38. Alternatives and Related Work

### xFormers (Meta)

Memory-efficient attention by Rabe & Staats (2021) independently discovered the tiling idea. Meta's xFormers library implements this. FA-2 is generally faster, but xFormers has broader architecture support and is often used as a fallback.

```python
from xformers.ops import memory_efficient_attention
output = memory_efficient_attention(query, key, value)
```

### Ring Attention

For sequences too long to fit on one GPU, Ring Attention (Liu et al., 2023) distributes attention across GPUs. Each GPU holds a portion of the KV sequence; they communicate in a "ring" pattern while computing attention.

```
GPU 0: holds K,V[0:N/4]     ← sends to GPU 1 while computing with current block
GPU 1: holds K,V[N/4:N/2]   ← ring communication
GPU 2: holds K,V[N/2:3N/4]
GPU 3: holds K,V[3N/4:N]

Combined with FlashAttention within each GPU.
Enables 1M+ context lengths across multiple GPUs.
```

### Linear Attention

A family of methods that reformulates attention to avoid O(N²):

```
Standard: softmax(Q × Kᵀ) × V       → O(N²) memory, O(N²d) compute
Linear:   φ(Q) × (φ(K)ᵀ × V)       → O(N) memory, O(Nd²) compute

Where φ is a feature map approximating the kernel.
Trade: no softmax, approximate, but fundamentally cheaper.
```

Examples: Performer (Google, 2021), RWKV (linear recurrence), RetNet.

### Mamba / State Space Models

An entirely different approach — replace attention with structured state space models (SSMs):

```
Attention: O(N²) in N (sequence length)
Mamba:     O(N) in N

No attention matrix at all.
Different inductive biases — excels at some tasks, weaker on others.
```

Increasingly competitive with Transformers on long-context tasks.

### Sparse Attention (BigBird, Longformer)

Restrict which tokens can attend to which:

```
Longformer:
  Local window: each token attends to ±W neighbors
  + Selected "global" tokens attend to everything

BigBird:
  Local + global + random attention patterns

Complexity: O(N × (W + global))  instead of O(N²)
Approximate: might miss some long-range dependencies
```

Can be combined with FlashAttention for the local window computation.

---

# Part 8 — Research Papers and Further Reading

---

## 39. Essential Papers with Summaries

### Foundational Softmax Work

**"Online normalizer calculation for softmax"**
Milakov & Gimelshein, NVIDIA, 2018
[arxiv.org/abs/1805.02867](https://arxiv.org/abs/1805.02867)

First to formally describe the online softmax algorithm — maintaining running (max, sum) statistics to compute softmax in a single pass. This is the mathematical foundation of FlashAttention's core trick. Short paper (~8 pages), highly readable.

---

**"Self-attention Does Not Need O(n²) Memory"**
Rabe & Staats, Google Brain, 2021
[arxiv.org/abs/2112.05682](https://arxiv.org/abs/2112.05682)

Independent discovery of memory-efficient attention tiling. Shows that attention can be computed with O(√N) memory by processing in blocks. Precursor to FlashAttention, less optimized but proves the concept. Important for understanding the intellectual lineage.

---

### FlashAttention Core Papers

**"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"**
Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
Stanford University, NeurIPS 2022
[arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)

The original paper. Introduces: IO-awareness as a design principle, the tiling algorithm for attention, online softmax for the forward pass, recomputation for the backward pass, and the formal IO complexity analysis. Benchmarks on A100 showing 2–4× speedup. Essential reading for anyone working on LLM systems.

**Key contribution:** Proved that the bottleneck in attention is HBM bandwidth, not FLOPs, and designed the first practical algorithm around this insight.

---

**"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"**
Tri Dao
Princeton University, ICLR 2024
[arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)

Builds on FA-1 with three key improvements: (1) reducing non-matmul FLOPs by deferring O rescaling, (2) adding parallelism across the sequence dimension, (3) better intra-SM work partitioning. Also adds native GQA/MQA support. ~2× speedup over FA-1. This is what PyTorch 2.0 integrates via `scaled_dot_product_attention`.

---

**"FlashAttention-3: Fast and Accurate Attention for GPUs from H100s to Blackwells"**
Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao
2024
[arxiv.org/abs/2407.08608](https://arxiv.org/abs/2407.08608)

Targets H100 specifically. Introduces: asynchronous TMA-based data loading, warp specialization, overlapping GEMM with softmax, FP8 Tensor Core support, and incoherent processing for FP8 numerical stability. Achieves ~75% of H100 FP16 peak throughput. Essential if you're running on H100.

---

### Related Systems Papers

**"Efficient Memory Management for Large Language Model Serving with PagedAttention"**
Woosuk Kwon et al., UC Berkeley, SOSP 2023
[arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

Introduced vLLM. Solves the KV cache memory fragmentation problem via OS-inspired paging. Orthogonal to FlashAttention (FA solves training efficiency, PagedAttention solves inference KV memory management). 10–24× throughput improvement. Both are now standard in production LLM serving.

---

**"Ring Attention with Blockwise Transformers for Near-Infinite Context"**
Liu et al., UC Berkeley, 2023
[arxiv.org/abs/2310.01889](https://arxiv.org/abs/2310.01899)

Combines FlashAttention-style blocking with ring-topology GPU communication to enable attention over sequences longer than a single GPU's memory. Enables 1M+ context windows across multiple GPUs. Basis for multi-GPU long-context training.

---

**"Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"**
Tillet et al., MIT, 2019 (updated 2021+)
[triton-lang.org](https://triton-lang.org)

OpenAI's Triton language — used to write custom GPU kernels in Python-like syntax. FlashAttention can be implemented in Triton (and often is for experimentation). Important for understanding how researchers actually implement and iterate on kernels like FlashAttention without writing raw CUDA C.

---

**"xFormers: A modular and hackable Transformer modelling library"**
Benjamin et al., Meta AI, 2022
[github.com/facebookresearch/xformers](https://github.com/facebookresearch/xformers)

Meta's library implementing memory-efficient attention independently. Often used alongside or instead of flash-attn. Has broader attention variant support. Good reference for how tiling is implemented in practice.

---

**"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"**
Ainslie et al., Google Research, 2023
[arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)

Directly relevant to FA-2's GQA support. FA-2 was specifically extended to natively support GQA after this paper showed GQA's quality/efficiency tradeoff. Reading GQA alongside FA-2 explains why FA-2 added native GQA support.

---

### Further Reading: Alternatives

**"Rethinking Attention with Performers"**
Choromanski et al., Google Brain, ICLR 2021
[arxiv.org/abs/2009.14794](https://arxiv.org/abs/2009.14794)

Linear attention via random feature approximation of the softmax kernel. O(N) attention. Understanding this helps clarify why FlashAttention chose to preserve exact softmax rather than approximate it.

---

**"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"**
Gu & Dao, 2023
[arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)

Tri Dao co-authored this too. Linear-time alternative to attention. Understanding Mamba alongside FlashAttention illustrates the two main approaches to the long-context problem: make O(N²) attention faster (FA) vs replace it with O(N) SSMs (Mamba).

---

## 40. Formula Sheet and Quick Reference

### Standard Attention

```
Attention(Q, K, V) = softmax(QKᵀ/√d) × V

HBM Memory: O(N²)
IO ops:     O(N²) reads/writes for intermediate matrices
```

### Online Softmax Update Rule

```
Initial state: m = -∞,  l = 0,  O = 0

For each new tile of scores x_new:
    m_new = max(m, max(x_new))
    l_new = exp(m - m_new) × l  +  rowsum(exp(x_new - m_new))
    O_new = exp(m - m_new) × O  +  exp(x_new - m_new) × V_new
    m, l, O = m_new, l_new, O_new

After all tiles:
    output = O / l
```

### FlashAttention IO Complexity

```
Naive attention:   Θ(N²) HBM reads/writes
FlashAttention:    Θ(Nd + N²d/M) HBM reads/writes
                   where M = SRAM size

Speedup factor:    ~M/d  (SRAM size / head dimension)
Typical M/d:       ~100KB / 128 ≈ 780×  (theoretical max)
Practical speedup: 2–4× (FA-1), 4–8× (FA-2), 8–16× (FA-3)
```

### Tile Size Selection

```
Must fit in SRAM:
  (4 × Br + 2 × Bc) × d × bytes ≤ SRAM_size

For d=128, FP16, SRAM=96KB:
  (4 × Br + 2 × Bc) × 128 × 2 ≤ 96 × 1024
  (4 × Br + 2 × Bc) ≤ 384
  With Br = Bc = 64: 4×64 + 2×64 = 384 ≤ 384 ✓
```

### Arithmetic Intensity Comparison

```
                    FLOPs           Bytes       AI (FLOPs/byte)
──────────────────────────────────────────────────────────────
Naive attention     4N²d            16N²        d/4 ≈ 32
FlashAttention      ~4N²d           ~2Nd        ~2Nd  (much higher)
GEMM (matmul)       2N³             ~N² bytes   ~N   (usually compute-bound)

H100 ridge point: ~295 FLOPs/byte
FlashAttention approaches but doesn't quite reach ridge point
FA-3 gets closest (~75% of peak)
```

---

## 41. Glossary

| Term | Definition |
|------|-----------|
| **IO-Awareness** | Designing algorithms to minimize reads/writes to slow memory (HBM), accepting more FLOPs if needed |
| **HBM** | High Bandwidth Memory — the main GPU memory (VRAM), fast but far from compute |
| **SRAM** | Static RAM — the small, very fast on-chip memory (shared memory in CUDA) |
| **Tiling** | Breaking large matrices into small tiles that fit in SRAM, processing tile by tile |
| **Online Softmax** | Algorithm to compute exact softmax incrementally, maintaining running (max, sum) statistics without seeing the full vector |
| **Arithmetic Intensity** | FLOPs per byte of HBM traffic — measures whether a kernel is compute-bound or memory-bound |
| **Roofline Model** | Framework plotting achievable performance vs arithmetic intensity to identify compute vs memory bound |
| **GEMM** | General Matrix Multiply — the operation Tensor Cores accelerate |
| **Tensor Core** | Specialized GPU hardware for fast matrix multiplication |
| **Non-matmul FLOPs** | Operations other than matmul (exp, max, sum) — run on regular CUDA cores at lower throughput |
| **Warp** | 32 GPU threads executing the same instruction simultaneously |
| **Warp Specialization** | Assigning different roles to different warps (e.g., load vs compute) to overlap memory and computation |
| **TMA** | Tensor Memory Accelerator — H100 hardware unit for asynchronous memory transfers |
| **WGMMA** | Warp Group Matrix Multiply Accumulate — H100's asynchronous matmul instruction |
| **Recomputation** | Recomputing intermediate values during backward pass instead of storing them — trades compute for memory |
| **Kernel Fusion** | Combining multiple GPU kernels into one to avoid intermediate writes to HBM |
| **Prefetching** | Loading the next data tile into SRAM while computing with the current tile |
| **Double Buffering** | Maintaining two SRAM buffers — one being computed, one being loaded — for continuous overlap |
| **Memory Bound** | Kernel where HBM bandwidth is the bottleneck, not FLOPs |
| **Compute Bound** | Kernel where FLOPs are the bottleneck, not memory bandwidth |
| **FP8** | 8-bit floating point — E4M3 or E5M2 formats, supported by H100 for ~2× throughput vs FP16 |
| **Incoherent Processing** | Random rotation of Q/K to spread outliers, improving FP8 quantization quality |
| **Ring Attention** | Distributes attention across multiple GPUs by passing KV blocks in a ring communication pattern |
| **Causal Mask** | Lower-triangular mask preventing tokens from attending to future positions |
| **Softmax Numerics** | Subtracting max before exp to prevent overflow while preserving mathematical equivalence |
| **Block Sparse Attention** | Attention restricted to sparse block patterns — can be combined with FA tiling |
| **Sequence Parallelism** | Parallelizing computation across the sequence length dimension (FA-2 key innovation) |
| **HBM Bandwidth** | H100: ~3.35 TB/s — the rate at which data can flow between HBM and SM |
| **SRAM Bandwidth** | ~10 TB/s — much faster than HBM, key to FA's speedup |
| **Backward Pass** | Computing gradients during training — requires attention weights P (FA recomputes these) |
| **xFormers** | Meta's memory-efficient attention library, independently discovered tiling idea |
| **Flash Decoding** | Variant of FlashAttention optimized for inference decode step (batch=1, growing KV) |
| **FA-1** | FlashAttention-1: original paper, A100-optimized, 2–4× over naive |
| **FA-2** | FlashAttention-2: sequence parallelism, fewer non-matmul FLOPs, native GQA, 2× over FA-1 |
| **FA-3** | FlashAttention-3: H100-specific, TMA, WGMMA, FP8, ~2× over FA-2 on H100 |

---

*Notes compiled from original research papers, CUDA programming guides, and GPU architecture documentation. Covers the full evolution from naive attention to FlashAttention-3 on H100.*