# CPU, GPU & LLM Systems — Complete Notes (Beginner to Engineer)

> From "what is a CPU core?" all the way to FlashAttention, distributed training, and inference optimization.  
> Written for ML beginners who want to deeply understand the *why* behind GPU hardware, CUDA, and LLM systems.

---

## Table of Contents

**Part 1 — CPU & GPU Fundamentals**
1. [What is a CPU?](#1-what-is-a-cpu)
2. [CPU Core — The Smart Engineer](#2-cpu-core--the-smart-engineer)
3. [What is a GPU?](#3-what-is-a-gpu)
4. [GPU Core — The Fast Calculator](#4-gpu-core--the-fast-calculator)
5. [CPU Core vs GPU Core — Side by Side](#5-cpu-core-vs-gpu-core--side-by-side)
6. [GPU Thread — The Smallest Worker](#6-gpu-thread--the-smallest-worker)
7. [Warp — Team of 32 Threads](#7-warp--team-of-32-threads)
8. [Warp Divergence — When Threads Disagree](#8-warp-divergence--when-threads-disagree)
9. [Streaming Multiprocessor (SM)](#9-streaming-multiprocessor-sm--the-department)
10. [CUDA Kernel — The Task Description](#10-cuda-kernel--the-task-description)
11. [Thread Block and Grid — The Full Hierarchy](#11-thread-block-and-grid--the-full-hierarchy)

**Part 2 — GPU Memory**
12. [GPU Memory Hierarchy — From Registers to VRAM](#12-gpu-memory-hierarchy--from-registers-to-vram)
13. [SRAM vs HBM — Speed vs Size](#13-sram-vs-hbm--speed-vs-size)
14. [Memory Bandwidth — The Highway](#14-memory-bandwidth--the-highway)
15. [Compute Bound vs Memory Bound](#15-compute-bound-vs-memory-bound)
16. [Tensor Cores — Specialized Matrix Hardware](#16-tensor-cores--specialized-matrix-hardware)

**Part 3 — CUDA Programming**
17. [CUDA Programming Basics — Thread Indexing](#17-cuda-programming-basics--thread-indexing)
18. [Matrix Multiplication Optimization — Tiling](#18-matrix-multiplication-optimization--tiling)
19. [GPU Occupancy — Keeping the GPU Busy](#19-gpu-occupancy--keeping-the-gpu-busy)
20. [CUDA Streams — Overlapping Work](#20-cuda-streams--overlapping-work)
21. [GPU Profiling — Finding Bottlenecks](#21-gpu-profiling--finding-bottlenecks)

**Part 4 — Transformers & LLMs**
22. [Transformer Architecture — Attention, FFN, Residuals](#22-transformer-architecture--attention-ffn-residuals)
23. [How Transformers Use the GPU](#23-how-transformers-use-the-gpu)
24. [Why LLMs Are Memory Monsters](#24-why-llms-are-memory-monsters)
25. [PyTorch Internals — How .backward() Maps to CUDA](#25-pytorch-internals--how-backward-maps-to-cuda)
26. [Mixed Precision Training — FP16/BF16](#26-mixed-precision-training--fp16bf16)
27. [FlashAttention — The Key Optimization](#27-flashattention--the-key-optimization)

**Part 5 — Scaling & Inference**
28. [Distributed Training — Data, Tensor, Pipeline Parallelism](#28-distributed-training--data-tensor-pipeline-parallelism)
29. [Inference Optimization — KV Cache, Quantization, Speculative Decoding](#29-inference-optimization--kv-cache-quantization-speculative-decoding)
30. [PCIe & NVLink — The GPU Communication Highway](#30-pcie--nvlink--the-gpu-communication-highway)
31. [Gradient Checkpointing — Trading Compute for Memory](#31-gradient-checkpointing--trading-compute-for-memory)

**Part 6 — Reference**
32. [Full Mental Model Summary](#32-full-mental-model-summary)
33. [Quick Reference Glossary](#33-quick-reference-glossary)

---

# Part 1 — CPU & GPU Fundamentals

---

## 1. What is a CPU?

A **CPU (Central Processing Unit)** is the main brain of your computer. It handles everything — running your OS, executing Python scripts, managing files, making decisions.

### Everyday Analogy

Think of a CPU as a **brilliant manager** who can:
- Read complex instructions
- Make decisions (if/else logic)
- Handle many different tasks
- Predict what comes next to work faster

### CPU Specs You've Seen

When you buy a laptop with an **Intel i5** or **AMD Ryzen 7**, the number is a performance tier. Cores = independent workers inside.

| CPU Model | Cores | Use Case |
|-----------|-------|----------|
| Intel i5 | 6–12 | Everyday computing |
| Intel i7 | 8–16 | Heavy workloads |
| Intel i9 | 16–24 | Extreme performance |
| AMD Ryzen 9 | 16–32 | Workstations |

### What Makes a CPU "Smart"?

A modern CPU core contains:

- **Control Unit** — Decodes instructions and decides what to do
- **Branch Predictor** — Guesses the next instruction in advance
- **Speculative Execution** — Runs future instructions before they're confirmed needed
- **Caches (L1, L2, L3)** — Ultra-fast memory close to the core
- **Scheduler** — Manages instruction ordering
- **ALUs** — Arithmetic Logic Units that do the actual math
- **Registers** — Tiny immediate memory right inside the core
- **Floating Point Units** — Handles decimal numbers

This complexity allows a CPU to handle:

```python
if user_logged_in:
    show_dashboard()
elif user_is_guest:
    show_login()
else:
    redirect_to_signup()
```

Complex, branchy, unpredictable logic — a CPU's specialty.

---

## 2. CPU Core — The Smart Engineer

```
CPU Core = Smart Engineer

Can:
  ✓ Make complex decisions
  ✓ Handle diverse tasks
  ✓ Work independently
  ✓ Predict future work
  ✓ Handle exceptions gracefully

Cost:
  ✗ Expensive to build
  ✗ Large silicon area
  ✗ High power consumption
  ✗ Slow for repeated simple tasks
```

### Why You Can't Have 10,000 CPU Cores

Each CPU core is enormous in silicon area because of all its prediction, scheduling, and caching logic. That's why CPUs cap out at tens of cores — not thousands.

---

## 3. What is a GPU?

A **GPU (Graphics Processing Unit)** was originally built to render pixels on screens. Rendering 1920×1080 means computing ~2 million pixel colors simultaneously, every frame.

That need created a completely different design: **many simple workers doing the same thing at once.**

Researchers discovered AI workloads (matrix math) look exactly like pixel rendering. The GPU became the engine of modern AI.

### GPU vs CPU — The Scale Difference

| Property | CPU | GPU |
|----------|-----|-----|
| Cores | 8–64 | Thousands |
| Per-core complexity | Very high | Very low |
| Best for | Serial, branchy logic | Parallel, repetitive math |
| Examples | Intel i9, AMD Ryzen | NVIDIA H100, RTX 4090 |

---

## 4. GPU Core — The Fast Calculator

**The most important correction:**

```
GPU Core ≠ CPU Core
```

A GPU core is a **simple arithmetic unit**:

```
+   (addition)
-   (subtraction)
*   (multiplication)
/   (division)
```

No branch prediction. No speculative execution. No complex caching. Just fast math.

```
CPU Core = Smart Engineer
GPU Core = Fast Calculator
```

### Why Simple Is Perfect for AI

AI workloads look like:

```python
result = a * b    # billions of times
result = c * d
result = e * f
```

Matrix multiplication — multiply and add, over and over. A simple fast calculator is perfect for this.

---

## 5. CPU Core vs GPU Core — Side by Side

```
CPU Core                          GPU Core
─────────────────────────────     ───────────────────────────
Control Unit          ✓           Control Unit          ✗
Branch Predictor      ✓           Branch Predictor      ✗
Speculative Execution ✓           Speculative Execution ✗
L1/L2 Cache           ✓           Minimal Cache         ~
Scheduler             ✓           Scheduler             ✗
ALU                   ✓           ALU                   ✓
Registers             Many        Registers             Few
FPU                   ✓           FPU                   ✓
Task Variety          High        Task Variety          Low
Power Per Core        High        Power Per Core        Low
Silicon Area Per Core Large       Silicon Area Per Core Small
```

Because GPU cores are so small, you can fit **thousands** on one chip.

---

## 6. GPU Thread — The Smallest Worker

Want to multiply every element by 2:

```python
x = [1, 2, 3, 4]
# Want: [2, 4, 6, 8]
```

**CPU way:**
```
Core: 1×2, 2×2, 3×2, 4×2   (sequential)
```

**GPU way:**
```
Thread 0 → 1 × 2 = 2
Thread 1 → 2 × 2 = 4
Thread 2 → 3 × 2 = 6
Thread 3 → 4 × 2 = 8
(all at once)
```

Each worker = **Thread**.

### What a GPU Thread Is NOT

| Thing | What it is |
|-------|------------|
| OS Thread | Heavy, managed by OS, expensive |
| Python Thread | GIL-limited, for I/O |
| CPU Thread | Runs on CPU core, complex work |
| **GPU Thread** | **Extremely lightweight, simple math only** |

GPU threads are so lightweight you can create millions instantly.

---

## 7. Warp — Team of 32 Threads

Managing 1 million threads individually is impossible. The GPU organizes them.

**NVIDIA GPUs execute threads in groups of 32 — called a Warp.**

```
Thread = Individual Worker
Warp   = Team of 32 Workers
```

### Organization

```
Warp 0:   Thread 0  ... Thread 31
Warp 1:   Thread 32 ... Thread 63
Warp 2:   Thread 64 ... Thread 95
```

### Why 32?

The hardware scheduler issues instructions to **one warp at a time**, not individual threads. This is far more efficient — like a manager giving orders to a team, not 32 individuals.

### SIMT — Single Instruction, Multiple Threads

All 32 threads in a warp execute the **same instruction** simultaneously, each on its own data:

```
Instruction: multiply by 2
Thread 0  → data[0] × 2
Thread 1  → data[1] × 2
...
Thread 31 → data[31] × 2
```

Same instruction. Different data. Simultaneously.

---

## 8. Warp Divergence — When Threads Disagree

```python
if x > 0:
    multiply()
else:
    divide()
```

Some threads want multiply, some want divide. The GPU cannot do both at once:

```
Step 1: Multiply for threads where x > 0   (others sit idle)
Step 2: Divide  for threads where x <= 0   (others sit idle)
```

This is **Warp Divergence** — a serious GPU performance killer.

### The Rule

Write GPU code with **minimal branching**. Keep all 32 threads in a warp doing the same thing. CUDA programmers obsess over this.

---

## 9. Streaming Multiprocessor (SM) — The Department

A GPU is divided into major compute units called **Streaming Multiprocessors (SMs)**.

```
GPU
 ├── SM 1
 ├── SM 2
 ...
 └── SM 132   (H100 has 132 SMs)
```

### Factory Analogy

```
GPU      = Factory
SM       = Department
Warp     = Team of 32 workers
Thread   = Individual worker
```

### Inside One SM

```
SM
├── 64–128 CUDA Cores
├── Tensor Cores
├── Registers          (ultra-fast, private per thread)
├── Shared Memory/L1   (fast, shared by block)
├── Warp Schedulers    (pick which warp runs next)
└── Warp Controllers
```

Multiple warps live inside one SM. While one warp waits for memory, another warp executes — keeping the SM busy (**latency hiding**).

### Real Scale

```
H100:
  132 SMs × 64 warps × 32 threads = 270,336 simultaneous threads
```

---

## 10. CUDA Kernel — The Task Description

A **kernel** is a GPU function that runs on thousands of threads simultaneously.

```c
__global__ void multiply_by_two(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}
```

### Launching

```c
multiply_by_two<<<1000, 256>>>(data, 256000);
//               ^^^^  ^^^
//               blocks  threads per block
```

GPU then:
```
1. Creates 256,000 threads
2. Groups into warps
3. Groups warps into blocks
4. Assigns blocks to SMs
5. Executes
6. Returns results
```

---

## 11. Thread Block and Grid — The Full Hierarchy

### Thread Block

A group of threads (up to 1024) that:
- Share **Shared Memory (SRAM)**
- Can **synchronize** with each other (`__syncthreads()`)
- All run on the **same SM**

```
Thread Block (256 threads)
 ├── Warp 0   (Thread 0–31)
 ├── Warp 1   (Thread 32–63)
 ...
 └── Warp 7   (Thread 224–255)
```

### Grid

Many blocks = a Grid = one complete kernel launch.

```
Grid
 ├── Block 0
 ├── Block 1
 ...
 └── Block M
```

### Complete Hierarchy (Top to Bottom)

```
Grid                 ← entire problem
 └── Thread Blocks   ← piece of the problem, shares SRAM
      └── Warps      ← group of 32 threads, same instruction
           └── Threads ← individual computation unit
```

---

# Part 2 — GPU Memory

---

## 12. GPU Memory Hierarchy — From Registers to VRAM

```
Fastest / Smallest
      │
      ▼
  Registers          ← Thread's personal memory (~1 cycle)
      │
      ▼
  Shared Memory      ← Block's whiteboard, SRAM (~5 cycles)
      │
      ▼
  L1 Cache           ← Automatic fast cache (~30 cycles)
      │
      ▼
  L2 Cache           ← Larger shared cache (~100 cycles)
      │
      ▼
  Global Memory      ← VRAM / HBM (~300–800 cycles)
      │
      ▼
Slowest / Largest
```

### Analogy

```
Registers       = Thread's pocket         (instant access)
Shared Memory   = Team whiteboard         (nearby, fast)
L1/L2 Cache     = Department storage      (short walk)
VRAM (HBM)      = Warehouse              (far away, slow)
```

---

### Registers

Every thread gets a small number of registers — the fastest memory on the GPU.

```c
float a = data[i];    // Register 1
float b = weights[i]; // Register 2
float c = a * b;      // Register 3
```

**Speed:** ~1 cycle. **Size:** ~32–255 per thread. **Scope:** Private per thread.

**Problem:** Limited. You can't give each of 100,000 threads huge storage.

---

### Shared Memory (SRAM)

All threads in the same block share this memory.

```c
__shared__ float tile[256];         // Declare shared memory
tile[threadIdx.x] = data[globalIdx]; // Load once
__syncthreads();                     // Wait for all threads to load
// Now all 256 threads can read tile[] at SRAM speed
```

**Speed:** ~5–10 cycles. **Size:** 48–96 KB per SM. **Scope:** All threads in same block.

#### Why It Matters (256× reduction example)

```
Without shared memory:
  256 threads each fetch same data from VRAM → 256 slow fetches

With shared memory:
  1 fetch from VRAM into SRAM → 256 fast reads from SRAM
  Result: 256× reduction in VRAM traffic for that data
```

---

### VRAM / Global Memory (HBM)

The main GPU memory. When people say "my GPU has 24GB" — they mean VRAM.

| GPU | VRAM |
|-----|------|
| RTX 4060 | 8 GB |
| RTX 4090 | 24 GB |
| A100 | 40–80 GB |
| H100 | 80 GB |

Stores: weights, activations, gradients, optimizer states, input data.

**Speed:** ~300–800 cycles latency. **Size:** 8–80+ GB. **Scope:** All threads everywhere.

#### The Warehouse Problem

```
Thread needs a value from VRAM.
  → Sends request
  → Waits 300–800 cycles
  → Value arrives
  → Computes
  → Sends result back

During that wait: thread does nothing.
GPU hides this by running OTHER warps instead.
This is called: Latency Hiding.
```

---

## 13. SRAM vs HBM — Speed vs Size

| Property | SRAM (Shared Mem) | HBM (VRAM) |
|----------|-------------------|------------|
| Speed | ~5 TB/s effective | ~3 TB/s |
| Latency | ~1–10 cycles | ~300–800 cycles |
| Size per GPU | ~20–50 MB | 24–80 GB |
| Location | On-chip (inside SM) | Off-chip (beside GPU die) |
| Cost per bit | Expensive | Cheaper |
| Analogy | Whiteboard at your desk | Warehouse down the road |

### Why Not Just Use More SRAM?

SRAM consumes far more silicon area per bit than HBM. An 80GB SRAM chip would be physically enormous and cost millions of dollars. The tradeoff is fundamental: **fast memory is always small, large memory is always slow.**

### What is HBM?

HBM = **High Bandwidth Memory**. Stacked directly beside the GPU die using advanced packaging (thousands of short connections).

```
Old GPUs (GDDR):
  GPU Chip ─────────────────── Memory chips
  (PCB traces, slow, thin wire)

Modern GPUs (HBM):
  GPU Die
  │││││││   ← Thousands of short parallel wires
  HBM Stack
  (physically adjacent, ultra-wide bus)
```

Much shorter distance + much wider bus = much more bandwidth.

---

## 14. Memory Bandwidth — The Highway

**Bandwidth** = how much data moves per second between memory and compute.

```
H100 HBM bandwidth: ~3.35 TB/s
= 3,350,000,000,000 bytes/second
```

### The Highway Analogy

```
Bandwidth = Number of highway lanes

1 lane  → few cars/second → GPU starves for data
100 lanes → many cars/second → GPU stays fed

More bandwidth = GPU compute stays busy = faster training
```

### Why H100 Is Fast

Not just more FLOPs — but dramatically more memory bandwidth than older GPUs. The compute units stay fed with data.

| GPU | Memory BW | FP16 TFLOPs |
|-----|-----------|-------------|
| RTX 3090 | 936 GB/s | 142 |
| A100 80GB | 2,000 GB/s | 312 |
| H100 SXM | 3,350 GB/s | 989 |

---

## 15. Compute Bound vs Memory Bound

### Compute Bound

```
Bottleneck = the math (FLOPs)
Memory can keep up, compute units are maxed out.

Examples:
  - Large dense matrix multiplications
  - Convolutions on big images
```

Adding more bandwidth wouldn't help. You need faster/more compute.

### Memory Bound

```
Bottleneck = data movement
Compute units wait for VRAM to deliver data.

Examples:
  - Attention in transformers
  - Layer normalization
  - Elementwise ops (ReLU, dropout, softmax)
  - Small batch inference
```

### The Shocking Truth

Most people assume LLM training = computation problem.

**It is actually a data movement problem.**

```
Matrix multiply:
  Cost of math:          1 unit
  Cost of loading data:  100 units

→ 99% of time is spent moving data, not computing
```

This is why researchers obsess over memory access patterns, not just FLOPs.

---

## 16. Tensor Cores — Specialized Matrix Hardware

Regular CUDA cores: `a * b + c` → one scalar operation.

**Tensor Cores**: `A × B + C` → entire **matrix** in ONE hardware instruction.

### Example

```
4×4 matrix multiply:
  Regular CUDA: 64 multiply-add operations
  Tensor Core:  1 instruction
```

### Why This Matters for AI

Transformers are almost entirely matrix multiplications:
- Embedding lookups
- Q, K, V projections
- Attention (Q × Kᵀ)
- Feed-forward layers

Tensor Cores accelerate all of these directly in hardware.

### Supported Precisions

| Format | Bits | Speed | Use Case |
|--------|------|-------|----------|
| FP32 | 32 | Baseline | High precision |
| TF32 | 19 | ~8× faster | Drop-in FP32 replacement |
| FP16 | 16 | ~16× faster | Standard training |
| BF16 | 16 | ~16× faster | Better numerical range |
| INT8 | 8 | ~32× faster | Inference |
| FP8 | 8 | ~32× faster | Cutting-edge training |

Lower precision = smaller data = faster to move = more fits in SRAM = faster overall.

---

# Part 3 — CUDA Programming

---

## 17. CUDA Programming Basics — Thread Indexing

CUDA is NVIDIA's programming model for GPU computation. When you write a CUDA kernel, every thread needs to know: *"Which piece of data am I responsible for?"*

### The Three Built-in Variables

```c
threadIdx.x   // Thread's position within its block (0 to blockDim.x - 1)
blockIdx.x    // Block's position within the grid   (0 to gridDim.x - 1)
blockDim.x    // Total threads per block
```

### Computing Global Index

```c
__global__ void kernel(float* data, int n) {
    // Each thread computes its unique global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard: don't go out of bounds
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}
```

### Visualizing the Index Math

```
blockDim.x = 4 (4 threads per block)

Block 0:  Thread 0,1,2,3  → Global idx 0,1,2,3
Block 1:  Thread 0,1,2,3  → Global idx 4,5,6,7
Block 2:  Thread 0,1,2,3  → Global idx 8,9,10,11
```

Formula: `global_idx = blockIdx.x * blockDim.x + threadIdx.x`

### 2D Indexing (for Matrices)

For matrix operations, use 2D blocks:

```c
__global__ void matrix_kernel(float* A, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // Each thread handles one element: A[row][col]
        int idx = row * cols + col;
        A[idx] = A[idx] * 2.0f;
    }
}

// Launch with 2D grid and 2D blocks
dim3 blockSize(16, 16);          // 256 threads per block in 16×16
dim3 gridSize((cols+15)/16, (rows+15)/16);
matrix_kernel<<<gridSize, blockSize>>>(A, rows, cols);
```

### Memory Allocation

```c
float* d_data;          // 'd_' prefix = device (GPU) memory

// Allocate on GPU
cudaMalloc(&d_data, n * sizeof(float));

// Copy from CPU to GPU
cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

// Launch kernel
my_kernel<<<numBlocks, threadsPerBlock>>>(d_data, n);

// Copy result back to CPU
cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

// Free GPU memory
cudaFree(d_data);
```

### Choosing Block Size

```
Common choices: 128, 256, 512 threads per block.

Rule of thumb:
  - Multiple of 32 (warp size)
  - At least 128 for good occupancy
  - 256 is a safe default

Number of blocks:
  numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock
  (ceiling division ensures all elements are covered)
```

---

## 18. Matrix Multiplication Optimization — Tiling

Naive matrix multiplication is **memory bound** — it reads from VRAM constantly and wastes bandwidth. The solution: **tiling with shared memory**.

### Naive Matrix Multiply (Bad)

```c
__global__ void naive_matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
        // Problem: every A[row*N+k] and B[k*N+col] fetched from VRAM
        // Same values fetched repeatedly by different threads
    }
    C[row * N + col] = sum;
}
```

Each element of A and B is loaded from VRAM **N times** by different threads. For N=1024, that's 1024 redundant fetches per value.

### Tiled Matrix Multiply (Good)

The idea: load a **tile** of A and B into shared memory once, reuse it for all threads in the block.

```c
#define TILE 16

__global__ void tiled_matmul(float* A, float* B, float* C, int N) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    // Step through tiles of A and B
    for (int t = 0; t < N / TILE; t++) {

        // Each thread loads ONE element into shared memory
        tileA[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];

        __syncthreads(); // Wait for all threads to finish loading

        // Compute partial dot product using tile (fast SRAM access)
        for (int k = 0; k < TILE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads(); // Wait before loading next tile
    }

    C[row * N + col] = sum;
}
```

### Why Tiling Is Faster

```
Without tiling:
  Each value loaded from VRAM N times
  For N=1024: 1024 VRAM fetches per value

With tiling (TILE=16):
  Load tile once into SRAM: 1 VRAM fetch
  Reuse 16 times from SRAM
  Result: 16× reduction in VRAM traffic
```

### Additional Optimizations

**Loop Unrolling:** Tell the compiler to unroll inner loops, reducing loop overhead.

```c
#pragma unroll
for (int k = 0; k < TILE; k++) {
    sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
}
```

**Register Caching:** Keep frequently used values in registers instead of re-reading shared memory.

**Double Buffering:** While threads compute on tile N, prefetch tile N+1 into a second shared memory buffer — overlapping compute and memory load.

---

## 19. GPU Occupancy — Keeping the GPU Busy

**Occupancy** = the ratio of active warps to the maximum possible warps on an SM.

```
Occupancy = Active Warps / Max Warps per SM

100% occupancy = every warp slot is filled
50% occupancy  = half the slots are empty
```

### Why Occupancy Matters

The GPU hides memory latency by switching between warps. If an SM has few warps, there's nothing to switch to when one stalls — the SM sits idle.

```
Low occupancy:
  Warp 0 waiting for memory...
  No other warps to run
  SM sits idle → wasted cycles

High occupancy:
  Warp 0 waiting for memory...
  → Switch to Warp 1 (it's ready)
  → Switch to Warp 2
  → Warp 0 ready again
  SM always busy → full utilization
```

### What Limits Occupancy?

Three resources cap how many warps an SM can hold:

```
1. Registers per thread
   More registers per thread → fewer threads fit → lower occupancy

2. Shared memory per block
   More shared mem per block → fewer blocks fit per SM → lower occupancy

3. Threads per block (block size)
   Too small blocks → SM can't schedule enough warps
```

### The Occupancy Tradeoff

```
More registers per thread:
  + Faster kernel (no register spilling to local memory)
  - Lower occupancy (fewer warps fit per SM)

Less shared memory:
  + Higher occupancy
  - More VRAM fetches (lost the tiling benefit)
```

This is why CUDA optimization is nuanced — you tune occupancy vs register/memory usage for each specific kernel.

### Checking Occupancy

NVIDIA provides the **Occupancy Calculator** and `cudaOccupancyMaxActiveBlocksPerMultiprocessor()` API to measure this programmatically.

---

## 20. CUDA Streams — Overlapping Work

By default, GPU operations are sequential. **CUDA Streams** allow multiple operations to run concurrently or overlap compute with data transfer.

### Default Behavior (Sequential)

```
Host → GPU: copy data      [===]
GPU kernel:                      [===]
GPU → Host: copy results              [===]

Total time = copy + compute + copy
```

### With Streams (Overlapped)

```
Stream 1: copy chunk 1   [===]
Stream 1: compute 1            [===]
Stream 2: copy chunk 2   [=====]
Stream 2: compute 2                  [===]
Stream 1: copy result 1        [===]
Stream 2: copy result 2              [===]

Total time < sum of all parts
```

### Code Pattern

```c
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Async copy on stream1
cudaMemcpyAsync(d_data1, h_data1, size, cudaMemcpyHostToDevice, stream1);

// Async copy on stream2 (overlaps with stream1's copy)
cudaMemcpyAsync(d_data2, h_data2, size, cudaMemcpyHostToDevice, stream2);

// Kernels on different streams (run concurrently)
my_kernel<<<grid, block, 0, stream1>>>(d_data1);
my_kernel<<<grid, block, 0, stream2>>>(d_data2);

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

### Why This Matters for LLMs

During large model inference, while the GPU computes on batch N, you can transfer batch N+1 data — hiding PCIe transfer latency entirely.

---

## 21. GPU Profiling — Finding Bottlenecks

You can't optimize what you can't measure. GPU profiling tells you exactly where time is being spent.

### NVIDIA Tools

**Nsight Systems** — Timeline profiler, shows CPU/GPU activity, kernel launches, memory transfers.

**Nsight Compute** — Deep per-kernel profiler, shows:
- Memory throughput (how close to peak bandwidth?)
- Compute utilization (how close to peak FLOPs?)
- Occupancy
- Warp stall reasons
- L2 cache hit rates

**nvprof** (older) / **ncu** (modern CLI) — Command-line profiling.

### PyTorch Profiler

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    with record_function("model_inference"):
        output = model(input)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### What to Look For

```
Roofline Model:
  Plot your kernel's achieved FLOPs vs achieved bandwidth.

  If near compute roof:   → compute bound (need more FLOPs capacity)
  If near memory roof:    → memory bound  (need more bandwidth / better reuse)
  If far from both:       → something else (kernel launch overhead, synchronization)
```

### Common Bottlenecks Found by Profiling

| Symptom | Likely Cause |
|---------|-------------|
| Low memory BW utilization | Poor memory access patterns (non-coalesced) |
| Many warp stalls | Waiting for VRAM (memory bound) |
| Low occupancy | Too many registers or too much shared memory per block |
| Long kernel launch gaps | Too many small kernels (kernel launch overhead) |
| High L2 miss rate | Working set too large for cache |

### Memory Coalescing

Threads in a warp accessing **contiguous** memory addresses is called coalesced access — the GPU can serve all 32 threads in one memory transaction.

```
Coalesced (GOOD):
  Thread 0 reads addr 0
  Thread 1 reads addr 1
  Thread 2 reads addr 2
  ...
  → 1 memory transaction for entire warp

Non-coalesced (BAD):
  Thread 0 reads addr 0
  Thread 1 reads addr 100
  Thread 2 reads addr 200
  ...
  → 32 separate memory transactions = 32× slower
```

---

# Part 4 — Transformers & LLMs

---

## 22. Transformer Architecture — Attention, FFN, Residuals

Understanding Transformer architecture is essential to understanding why GPUs are used the way they are.

### High-Level Structure

A Transformer model (like GPT) consists of:

```
Input Tokens
     ↓
Token Embedding + Position Encoding
     ↓
┌─────────────────────────────────┐
│  Transformer Block × N layers  │
│                                 │
│  ┌──────────────────────────┐  │
│  │   Multi-Head Attention   │  │
│  └──────────────────────────┘  │
│              ↓                  │
│  ┌──────────────────────────┐  │
│  │    Add & LayerNorm       │  │  ← Residual connection
│  └──────────────────────────┘  │
│              ↓                  │
│  ┌──────────────────────────┐  │
│  │  Feed-Forward Network    │  │
│  └──────────────────────────┘  │
│              ↓                  │
│  ┌──────────────────────────┐  │
│  │    Add & LayerNorm       │  │  ← Residual connection
│  └──────────────────────────┘  │
└─────────────────────────────────┘
     ↓
Output Logits → Softmax → Token Probabilities
```

---

### Multi-Head Attention (MHA)

The core of every Transformer. Each attention head independently attends to different parts of the input.

**Step 1: Project input into Q, K, V**

```
Q = Input × W_Q    [seq_len × d_model] × [d_model × d_head] = [seq_len × d_head]
K = Input × W_K
V = Input × W_V
```

These are matrix multiplications — runs on Tensor Cores.

**Step 2: Compute attention scores**

```
scores = Q × Kᵀ / √d_head     [seq_len × seq_len]
```

For seq_len=4096, d_head=128: this is a [4096 × 4096] matrix. 16 million values per head.

**Step 3: Softmax**

```
attention_weights = softmax(scores)    [seq_len × seq_len]
```

Elementwise operation — memory bound.

**Step 4: Weighted sum of values**

```
output = attention_weights × V         [seq_len × d_head]
```

Another matrix multiply — Tensor Cores again.

**Step 5: Concatenate heads, project**

```
concat all heads → [seq_len × d_model]
output = concat × W_O
```

---

### Feed-Forward Network (FFN)

After attention, every position passes through an independent FFN:

```
FFN(x) = max(0, x × W_1 + b_1) × W_2 + b_2
```

Or with SwiGLU (used in LLaMA):
```
FFN(x) = (SiLU(x × W_gate) ⊙ (x × W_up)) × W_down
```

The FFN typically has hidden dimension = **4× the model dimension** (or 8/3× for SwiGLU).

```
For d_model = 4096:
  FFN hidden = 16384 (4× expansion)
  Two large matrix multiplies per layer
```

The FFN is often responsible for **more FLOPs than attention** for long-context tasks.

---

### Residual Connections

Every sub-layer has a residual (skip) connection:

```
output = LayerNorm(x + SubLayer(x))
```

**Why this matters:**
- Allows gradients to flow directly back to early layers (solves vanishing gradient problem)
- Makes very deep networks (100+ layers) trainable
- The `+ x` addition is cheap but architecturally critical

**GPU impact:** The residual addition is an elementwise operation — memory bound, touches entire activation tensor.

---

### Layer Normalization

```
LayerNorm(x) = γ * (x - mean(x)) / sqrt(var(x) + ε) + β
```

Normalizes across the feature dimension. Also memory bound — reads entire activation, computes stats, writes result.

---

### Causal Masking

During training, each token can only attend to previous tokens (not future ones). This is implemented via a mask:

```
Set scores[i][j] = -infinity   for all j > i
Then softmax makes those positions → 0
```

GPU-wise: the mask is applied before softmax, often fused into the softmax kernel.

---

## 23. How Transformers Use the GPU

Tracing one forward pass of a GPT-style model:

```
Input tokens → Embedding lookup
      ↓
For each of N layers:

  [Attention]
  Q = X × W_Q    ← matmul on Tensor Cores
  K = X × W_K    ← matmul on Tensor Cores
  V = X × W_V    ← matmul on Tensor Cores
  S = Q × Kᵀ     ← matmul on Tensor Cores (large!)
  S = S / √d     ← elementwise, memory bound
  S = mask(S)    ← elementwise, memory bound
  A = softmax(S) ← memory bound
  O = A × V      ← matmul on Tensor Cores
  O = O × W_out  ← matmul on Tensor Cores

  [Add & Norm]
  X = LayerNorm(X + O)   ← memory bound

  [FFN]
  H = X × W_1    ← matmul on Tensor Cores
  H = SiLU(H)    ← elementwise, memory bound
  O = H × W_2    ← matmul on Tensor Cores

  [Add & Norm]
  X = LayerNorm(X + O)   ← memory bound

Output projection → logits
Softmax → token probabilities
```

**Pattern:** Matrix multiplications (compute heavy, Tensor Core) alternating with elementwise ops (memory bound, CUDA cores).

---

## 24. Why LLMs Are Memory Monsters

### Weights Alone

```
LLaMA 7B:
  7B parameters × 2 bytes (FP16) = 14 GB

LLaMA 70B:
  70B × 2 bytes = 140 GB    ← doesn't fit on a single GPU
```

### Full Training Memory

```
Component             Size (7B model)
──────────────────────────────────────
Weights (FP16)        14 GB
Gradients (FP16)      14 GB
Master weights (FP32) 28 GB
Adam m (FP32)         28 GB
Adam v (FP32)         28 GB
Activations           10–50 GB (varies by batch/seq len)

Total:                ~120–170 GB
```

This is why training 7B models requires multiple 80GB H100s.

### The Attention Memory Problem

```
Sequence Length = 8192

Attention matrix per head: 8192 × 8192 = 67M values
                           × 4 bytes (FP32) = 256 MB

For 32 heads: 256 MB × 32 = 8 GB

Just for ONE forward pass of ONE layer's attention scores.
Multiply by N layers for full training.
```

For context lengths of 128K or 1M tokens — completely impossible without optimization (FlashAttention).

---

## 25. PyTorch Internals — How .backward() Maps to CUDA

When you call `loss.backward()` in PyTorch, what actually happens on the GPU?

### The Computation Graph

PyTorch builds a **dynamic computation graph** during the forward pass:

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2          # Node: multiply
z = y.sum()        # Node: sum
z.backward()       # Traverse graph backwards
```

Each operation records:
- What inputs it received
- What function computed it
- How to compute its gradient

### The Autograd Engine

When `backward()` is called:

```
1. Start from loss node
2. Call each node's backward function
3. Accumulate gradients into .grad buffers
4. Propagate backwards through the graph
```

Each "backward function" is a hand-written CUDA kernel (or calls into cuBLAS/cuDNN).

### Example: Linear Layer Backward

Forward: `output = input @ weight`

Backward needs two gradients:
```
grad_input  = grad_output @ weight.T    ← another matrix multiply
grad_weight = input.T @ grad_output     ← another matrix multiply
```

Both run on Tensor Cores. The backward pass is typically **2× the FLOPs** of the forward pass.

### Key Operations and Their Kernels

| Operation | Forward CUDA | Backward CUDA |
|-----------|-------------|---------------|
| Linear (matmul) | cuBLAS GEMM | Two cuBLAS GEMMs |
| Attention | Custom kernel | Custom kernel |
| LayerNorm | Fused kernel | Fused kernel |
| Softmax | Fused kernel | Fused kernel |
| ReLU/SiLU | Elementwise kernel | Elementwise kernel |
| Embedding | Gather kernel | Scatter kernel |

### Custom CUDA Kernels vs torch.compile

PyTorch has two approaches to custom optimization:

**Custom CUDA Kernels** (FlashAttention, xFormers):
- Write C++/CUDA directly
- Complete control over memory access, shared memory, etc.
- High effort, maximum performance

**torch.compile** (introduced in PyTorch 2.0):
- Python-level decorator: `@torch.compile`
- PyTorch traces your model and generates optimized Triton/CUDA kernels
- Much easier, often 1.5–3× speedup for free

```python
model = MyTransformer()
model = torch.compile(model)   # That's it — auto-optimization
```

### Gradient Accumulation

When memory is tight, you can accumulate gradients across multiple mini-batches before stepping the optimizer:

```python
optimizer.zero_grad()
for i, (x, y) in enumerate(dataloader):
    loss = model(x, y) / accumulation_steps
    loss.backward()          # Accumulates .grad
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()     # Update weights
        optimizer.zero_grad()
```

**GPU effect:** Smaller batch per forward pass → less VRAM for activations → allows effectively larger batch sizes.

---

## 26. Mixed Precision Training — FP16/BF16

### Why Reduce Precision?

```
FP32 parameter: 4 bytes
FP16 parameter: 2 bytes

Half the memory → can fit 2× larger models or batches
Also: Tensor Cores run much faster on FP16/BF16
```

### FP32 vs FP16 vs BF16

| Format | Sign | Exponent | Mantissa | Range | Precision |
|--------|------|----------|----------|-------|-----------|
| FP32 | 1 | 8 | 23 | ±3.4e38 | High |
| FP16 | 1 | 5 | 10 | ±65504 | Medium |
| BF16 | 1 | 8 | 7 | ±3.4e38 | Lower |

**FP16 problem:** Small range (max 65504). Large gradients can overflow to infinity. Small gradients can underflow to zero.

**BF16 advantage:** Same exponent range as FP32 (handles large/small numbers) but fewer mantissa bits. Generally preferred for training LLMs.

### Automatic Mixed Precision (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()   # For FP16 overflow prevention

for x, y in dataloader:
    with autocast():            # Run forward in FP16/BF16
        output = model(x)
        loss = criterion(output, y)

    scaler.scale(loss).backward()   # Scale loss to prevent underflow
    scaler.step(optimizer)          # Unscale before optimizer step
    scaler.update()
```

### Loss Scaling

FP16 gradients can underflow to zero (subnormal numbers become 0). Loss scaling prevents this:

```
1. Multiply loss by large scale factor (e.g., 2^16 = 65536)
2. Backward pass with scaled loss → scaled gradients (now in valid range)
3. Before optimizer.step(): divide gradients by scale factor
4. If overflow detected: skip step, reduce scale
5. If no overflow for N steps: increase scale
```

### Master Weights (FP32)

Mixed precision training typically keeps FP32 "master weights":

```
Master Weights (FP32)  ← optimizer updates these
        ↓
Copy to FP16          ← forward/backward uses FP16
        ↓
Gradients (FP16)      ← computed in FP16
        ↓
Accumulate in FP32    ← gradients cast to FP32 before weight update
```

This ensures the weight update has sufficient numerical precision.

---

## 27. FlashAttention — The Key Optimization

### The Problem with Naive Attention

```
1. Load Q from HBM
2. Load K from HBM
3. Compute Q × Kᵀ → Stores [seq × seq] matrix in HBM  ← huge!
4. Load matrix from HBM
5. Softmax in HBM
6. Store result to HBM
7. Load again
8. Load V from HBM
9. Compute Attention × V
10. Store to HBM
```

For seq=4096: creates a 4096×4096 = 16M value matrix, stored and reloaded multiple times.

### The FlashAttention Insight

> Never store the full attention matrix. Process it in tiles that fit in SRAM.

```
For each tile of Q (fits in SRAM):
    Load tile_Q → SRAM

    For each tile of K, V:
        Load tile_K, tile_V → SRAM
        Compute partial scores = tile_Q × tile_Kᵀ
        Update running softmax (online normalization trick)
        Accumulate partial output
        Discard intermediate scores (never go to HBM)

    Write output tile → HBM
```

The key trick: **online softmax normalization** lets you compute softmax incrementally without seeing the whole row first.

### Online Softmax (The Math Trick)

Normal softmax requires two passes: find max, then normalize. FlashAttention uses a numerically stable one-pass algorithm that updates a running maximum and normalization factor as new tiles arrive.

```
Traditional:
  Step 1: Read all scores, find max m
  Step 2: Compute exp(score - m) / sum(exp)

FlashAttention:
  For each new tile:
    Update running max m_new = max(m_old, tile_max)
    Rescale previous output with correction factor
    Accumulate new tile's contribution
  → Mathematically identical result, never needs full matrix
```

### Memory Complexity

```
Naive Attention:
  Memory = O(N²)   where N = sequence length
  For N=4096: 16M values × 4 bytes = 64 MB per head

FlashAttention:
  Memory = O(N)
  Only stores output + running statistics, no attention matrix
```

### Speed Comparison

```
Feature                 Naive Attention   FlashAttention
────────────────────────────────────────────────────────
Memory usage            O(N²)             O(N)
HBM reads/writes        O(N²)             O(N)
Speed                   Baseline          2–4× faster
Max seq len (A100 80GB) ~4K–8K            100K+
Gradient computation    Stored in VRAM    Recomputed from tiles
```

### FlashAttention-2 Improvements

Published 2023. Key changes:
- Parallelizes across sequence length dimension (better GPU utilization)
- Reduces non-matmul FLOPs (warps spend more time on Tensor Cores)
- Better work partitioning across SMs
- ~2× speedup over FlashAttention-1

### FlashAttention-3 Improvements

Published 2024. Specifically targets H100:
- Uses H100's asynchronous memory copy instructions
- Overlaps GEMM and softmax computation (pipeline them)
- Exploits H100's FP8 Tensor Cores for even faster computation
- ~2× speedup over FlashAttention-2 on H100

### Using FlashAttention in Practice

```python
# PyTorch 2.0+ (uses FlashAttention automatically when possible)
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# Or via HuggingFace Transformers
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
```

---

# Part 5 — Scaling & Inference

---

## 28. Distributed Training — Data, Tensor, Pipeline Parallelism

When a model doesn't fit on one GPU — or you want to train faster — you use multiple GPUs.

### Data Parallelism (DP)

**Idea:** Same model on every GPU. Different data on each GPU.

```
GPU 0: Model copy + Batch 0 → gradient_0
GPU 1: Model copy + Batch 1 → gradient_1
GPU 2: Model copy + Batch 2 → gradient_2
GPU 3: Model copy + Batch 3 → gradient_3
              ↓
    All-Reduce: average gradients across all GPUs
              ↓
    Each GPU updates its model copy (now identical again)
```

**Pros:** Simple, scales batch size linearly.  
**Cons:** Model must fit on ONE GPU. Communication cost for gradient sync.

```python
# PyTorch DDP
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

### Tensor Parallelism (TP)

**Idea:** Split individual weight matrices across GPUs.

```
weight matrix W [d_model × d_model]

With 4-way tensor parallelism:
  GPU 0: W[:, 0:d/4]
  GPU 1: W[:, d/4:d/2]
  GPU 2: W[:, d/2:3d/4]
  GPU 3: W[:, 3d/4:]
```

Each GPU computes its column shard of the output. Results are summed via All-Reduce.

**Example: Attention heads**

```
32 heads split across 4 GPUs:
  GPU 0: heads 0–7
  GPU 1: heads 8–15
  GPU 2: heads 16–23
  GPU 3: heads 24–31
```

**Pros:** Each GPU only needs 1/N of the weight memory.  
**Cons:** Requires All-Reduce after each layer — expensive, needs fast GPU interconnect (NVLink).

### Pipeline Parallelism (PP)

**Idea:** Split model layers across GPUs.

```
GPU 0: Layers 1–8    (bottom of model)
GPU 1: Layers 9–16
GPU 2: Layers 17–24
GPU 3: Layers 25–32  (top of model)
```

Data flows through GPUs sequentially like a pipeline.

**Problem: Pipeline Bubbles**

```
Naive:
  GPU 0 runs layer 1–8
  GPU 1 waits...
  GPU 2 waits...
  GPU 3 waits...
  → GPU 1,2,3 idle most of the time (bubble)
```

**Solution: Micro-batching**

Split the batch into micro-batches and pipeline them:

```
Time:  1   2   3   4   5   6   7
GPU0: [mb1][mb2][mb3][mb4]
GPU1:      [mb1][mb2][mb3][mb4]
GPU2:           [mb1][mb2][mb3][mb4]
GPU3:                [mb1][mb2][mb3][mb4]
```

Reduces bubble to `(stages - 1) / (stages - 1 + micro_batches)`.

### 3D Parallelism

Production LLM training (GPT-4, LLaMA 3) uses all three:

```
Data Parallelism    → multiple replicas, different data
Tensor Parallelism  → split matrices within a layer
Pipeline Parallelism→ split layers across GPUs

Example (Megatron-LM style for 1000B model):
  DP × TP × PP = 8 × 8 × 16 = 1024 GPUs
```

### ZeRO Optimization (DeepSpeed)

ZeRO (Zero Redundancy Optimizer) partitions optimizer states, gradients, and parameters across data-parallel ranks:

```
ZeRO Stage 1: Partition optimizer states only
ZeRO Stage 2: + Partition gradients
ZeRO Stage 3: + Partition model parameters
```

ZeRO Stage 3 allows training models much larger than single-GPU VRAM — the model is sharded across all GPUs.

---

## 29. Inference Optimization — KV Cache, Quantization, Speculative Decoding

Training and inference have very different bottlenecks. Inference is about latency and throughput at minimal cost.

---

### KV Cache

During autoregressive generation, every new token attends to all previous tokens. Without caching:

```
Token 1: Compute Q,K,V for token 1
Token 2: Recompute Q,K,V for token 1, compute for token 2
Token 3: Recompute Q,K,V for tokens 1,2, compute for token 3
...

Cost grows quadratically! Completely wasteful.
```

**KV Cache:** Store K and V from previous tokens, only compute K and V for the new token.

```
Token 1: Compute K1, V1 → cache them
Token 2: Load K1,V1 from cache + Compute K2,V2 → cache K2,V2
Token 3: Load K1,V1,K2,V2 from cache + Compute K3,V3 → cache K3,V3
...

Each step: O(1) compute instead of O(N) recompute
```

**Memory cost:**

```
KV cache for LLaMA 7B:
  Per layer: 2 (K+V) × num_heads × head_dim × seq_len × 2 bytes
  For 4096 sequence, 32 heads, 128 dim, 32 layers:
    = 2 × 32 × 128 × 4096 × 2 bytes × 32 layers
    = ~2 GB for one sequence
```

For long contexts (128K tokens) or large batches, KV cache dominates GPU memory.

**KV Cache optimizations:**
- **Multi-Query Attention (MQA):** All heads share one K and V head → reduces KV cache by `num_heads×`
- **Grouped Query Attention (GQA):** Groups of heads share K/V → used in LLaMA 2/3
- **PagedAttention (vLLM):** KV cache stored in non-contiguous "pages" like OS virtual memory → enables dynamic memory management

---

### Quantization

Quantization reduces parameter precision to save memory and increase throughput.

```
FP32: 4 bytes/param   ← training default
FP16: 2 bytes/param   ← standard inference
INT8: 1 byte/param    ← quantized inference
INT4: 0.5 byte/param  ← heavily quantized
```

**Weight-only quantization:**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=bnb_config
)
```

**GPTQ (Post-Training Quantization):**
- Compresses weights layer by layer
- Uses calibration data to minimize quantization error
- 4-bit INT4 with minimal accuracy loss

**AWQ (Activation-aware Weight Quantization):**
- Recognizes that not all weights are equally important
- Protects salient weights (those that affect activation magnitude most)
- Better accuracy than GPTQ at same bitwidth

**Quantization impact:**

```
Model           FP16 Size   INT4 Size   Speed (tokens/sec)
────────────────────────────────────────────────────────
LLaMA 7B        14 GB       4 GB        ~4× faster
LLaMA 70B       140 GB      35 GB       Can run on 2× GPU instead of 8×
```

---

### Speculative Decoding

**Problem:** Autoregressive decoding generates one token at a time. The large model is underutilized (batch size = 1 is memory-bound, not compute-bound).

**Idea:** Use a small fast "draft" model to guess multiple tokens ahead, then verify with the large model in parallel.

```
Step 1: Draft model generates k tokens quickly
        (cheap, small model like 7B when target is 70B)

        draft: [token1, token2, token3, token4, token5]

Step 2: Target model verifies all k draft tokens IN PARALLEL
        (one forward pass, similar cost to generating 1 token)

        target checks: [✓, ✓, ✓, ✗, —]

Step 3: Accept verified tokens (1–3), reject from first mismatch
        Accepted: token1, token2, token3 (plus one new from target)

Step 4: Restart from rejection point
```

**Why it's faster:**

```
Without speculative decoding:
  5 tokens = 5 forward passes of large model

With speculative decoding:
  5 tokens ≈ 1 forward pass of large model + 5 passes of small model
  Small model is ~10× cheaper → net speedup 2–3×
```

**Condition:** Works best when draft model has high acceptance rate (similar distribution to target). Token acceptance rate ~70–80% is typical.

---

### Batching Strategies

**Static batching:** Wait for N requests, process together.
```
Problem: Fast requests wait for slow ones (different sequence lengths)
```

**Dynamic batching (continuous batching):** Insert new requests mid-generation.
```
While request A is generating token 50, 
insert request B and process together.
Result: Much higher GPU utilization at serving time.
```

Used by vLLM, TensorRT-LLM, and other production inference engines.

---

## 30. PCIe & NVLink — The GPU Communication Highway

When using multiple GPUs, data must travel between them. The interconnect is often the bottleneck.

### PCIe (Peripheral Component Interconnect Express)

Standard GPU-to-CPU and GPU-to-GPU connection:

```
CPU
 ├── PCIe x16 lane → GPU 0
 ├── PCIe x16 lane → GPU 1
 ...
```

**PCIe Bandwidth:**

| Version | Bandwidth (x16) |
|---------|----------------|
| PCIe 3.0 | 16 GB/s |
| PCIe 4.0 | 32 GB/s |
| PCIe 5.0 | 64 GB/s |

**Problem:** For distributed training, gradients must be exchanged between GPUs. At 16 GB/s, this is far slower than GPU HBM bandwidth (~3 TB/s). PCIe becomes the bottleneck in multi-GPU setups connected only via CPU.

### NVLink

NVIDIA's direct GPU-to-GPU interconnect, bypassing the CPU:

```
GPU 0 ══════NVLink══════ GPU 1
```

**NVLink bandwidth:**

| Version | Per-link BW | H100 (18 links) |
|---------|------------|-----------------|
| NVLink 3 | 25 GB/s × 2 | — |
| NVLink 4 | 25 GB/s × 2 | 900 GB/s total |

H100's 900 GB/s NVLink is ~28× faster than PCIe 4.0. This enables efficient tensor parallelism (which requires All-Reduce between GPUs every layer).

### NVSwitch

For large clusters (8+ GPUs), NVSwitch creates an all-to-all full bandwidth topology:

```
DGX H100 (8 GPUs):
  All 8 GPUs connected via NVSwitch
  Any GPU can talk to any other at full 900 GB/s
  Total system bandwidth: 3.6 TB/s
```

### InfiniBand

For connecting multiple nodes (racks of servers) in a GPU cluster:

```
Node 0 (8× H100) ─── InfiniBand ─── Node 1 (8× H100)
```

**Bandwidth:** 400 Gb/s (HDR) or 800 Gb/s (NDR) per port.

Still much slower than NVLink, which is why **inter-node communication is the main bottleneck** in large-scale distributed training.

### Practical Impact

```
2-GPU training strategy decision:
  Tensor Parallelism → lots of All-Reduce per layer → needs NVLink
  Pipeline Parallelism → few point-to-point sends → works over PCIe
  Data Parallelism → gradient sync at end → works over PCIe
```

---

## 31. Gradient Checkpointing — Trading Compute for Memory

### The Activation Memory Problem

During training, PyTorch stores all intermediate activations from the forward pass — needed for backpropagation.

```
For a 32-layer Transformer, batch=16, seq=2048, d=4096:

Activations per layer ≈ batch × seq × d × 4 bytes
                      = 16 × 2048 × 4096 × 4 ≈ 512 MB

Total for 32 layers: 32 × 512 MB = 16 GB
Just for activations.
```

### The Gradient Checkpointing Solution

**Idea:** Don't store all activations. Re-compute them during backward when needed.

```
Without checkpointing:
  Forward:  Compute and STORE all activations (high memory)
  Backward: Load stored activations to compute gradients

With checkpointing:
  Forward:  Compute activations but only STORE at checkpoint boundaries
  Backward: Re-compute activations from checkpoints when needed
```

### Memory vs Compute Tradeoff

```
Without checkpointing:
  Memory: O(N)     where N = number of layers
  Compute: 1× forward

With checkpointing every layer:
  Memory: O(1)     (only store one layer's activation at a time)
  Compute: ~1.33× forward  (re-compute adds ~33% overhead)

With checkpointing every k layers:
  Memory: O(√N)    (optimal checkpoint frequency)
  Compute: ~1.33× forward
```

### PyTorch Implementation

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        # Normal forward
        return self.attention(x) + self.ffn(x)

class Transformer(nn.Module):
    def forward(self, x):
        for block in self.blocks:
            # Wrap with checkpoint — activations NOT stored
            x = checkpoint(block, x)
        return x
```

### When to Use It

```
Small model, large batch:
  → Memory tight → use gradient checkpointing

Large model barely fits:
  → Definitely use checkpointing

Inference:
  → No backward pass needed → no checkpointing needed
```

---

# Part 6 — Reference

---

## 32. Full Mental Model Summary

### The Complete GPU Hierarchy

```
GPU
├── HBM (VRAM)           ← 24–80 GB, ~3 TB/s, slow latency
│
├── SM 1
│   ├── Registers        ← Private per thread, ~1 cycle
│   ├── Shared Mem (SRAM)← 48–96 KB, shared by block, ~5 cycles
│   ├── Warp 0           ← 32 threads, same instruction
│   │   ├── Thread 0     ← ALU + few registers
│   │   ├── Thread 1
│   │   └── ... Thread 31
│   ├── Warp 1
│   └── Tensor Cores     ← Matrix multiply in one instruction
│
├── SM 2 ... SM 132
│
└── NVLink / PCIe        ← GPU-to-GPU / CPU-to-GPU communication
```

### The Speed Spectrum

```
Memory Type     Location    Latency      Size        Bandwidth
──────────────────────────────────────────────────────────────
Registers       On-thread   ~1 cycle     ~16 KB/SM   ~20 TB/s
Shared Memory   On-SM       ~5 cycles    ~96 KB/SM   ~10 TB/s
L1 Cache        On-SM       ~30 cycles   ~128 KB/SM  ~10 TB/s
L2 Cache        On-GPU      ~100 cycles  ~50 MB      ~5 TB/s
HBM (VRAM)      Off-chip    ~300 cycles  24–80 GB    ~3 TB/s
PCIe (CPU RAM)  Off-GPU     ~1000 cycles TBs         ~64 GB/s
```

### The Central Insight

```
Modern AI performance bottleneck:

NOT:   How fast can the GPU compute?
BUT:   How fast can data move from HBM to compute units?

Secondary bottleneck in multi-GPU:
       How fast can GPUs communicate with each other?
```

### How Every Optimization Connects

| Optimization | Problem Solved | Mechanism |
|---|---|---|
| Shared Memory (SRAM) tiling | VRAM traffic | Reuse data in fast on-chip memory |
| FlashAttention | Huge attention matrix in VRAM | Tile computation in SRAM, never materialize full matrix |
| Tensor Cores | Slow matrix multiply | Specialized hardware, 1 instruction per 4×4 block |
| BF16 / FP16 | Memory capacity & BW | 2× smaller data, 2× faster to move |
| INT8 / INT4 Quantization | Memory capacity & BW | 4–8× smaller data for weights |
| KV Cache | Recomputing K,V each step | Cache and reuse K,V from previous tokens |
| Gradient Checkpointing | Activation memory | Recompute activations instead of storing |
| Data Parallelism | Training throughput | More GPUs, more data per step |
| Tensor Parallelism | Model doesn't fit on 1 GPU | Split weight matrices across GPUs |
| Pipeline Parallelism | Model doesn't fit on 1 GPU | Split layers across GPUs |
| ZeRO Optimization | Optimizer state memory | Shard optimizer states across GPUs |
| NVLink | Slow GPU-GPU communication | Direct high-speed GPU interconnect |
| Speculative Decoding | Inference latency | Parallel verification of draft tokens |
| Continuous Batching | Low GPU utilization at serving | Dynamic insertion of requests |
| torch.compile | Kernel overhead | Fuse operations, generate optimized kernels |

---

## 33. Quick Reference Glossary

| Term | Definition |
|------|-----------|
| **ALU** | Arithmetic Logic Unit — the hardware that does math |
| **Autoregressive** | Generating one token at a time, each conditioned on previous |
| **Bandwidth** | Data throughput, measured in GB/s or TB/s |
| **BF16** | 16-bit float with same exponent range as FP32, less precision |
| **CUDA** | NVIDIA's GPU programming platform and API |
| **CUDA Core** | Simple ALU in a GPU, does scalar float math |
| **Compute Bound** | GPU bottleneck is math operations, not memory |
| **CPU Core** | Full complex processor: control, prediction, scheduling, math |
| **Data Parallelism** | Same model on many GPUs, each with different data |
| **FlashAttention** | IO-aware attention algorithm, tiles into SRAM, never stores N² matrix |
| **FLOPs** | Floating Point Operations — measure of compute work |
| **FP16** | 16-bit float — half precision, used for training/inference |
| **FP8** | 8-bit float — emerging format, very fast on H100 |
| **GPU Core** | Simple arithmetic unit (ALU) for fast parallel math |
| **GPU Thread** | Lightest unit of GPU execution, runs one scalar operation |
| **GQA** | Grouped Query Attention — heads share K/V, reduces KV cache |
| **Gradient Checkpointing** | Trade compute for memory — recompute activations in backward pass |
| **Grid** | All thread blocks in one kernel launch |
| **HBM** | High Bandwidth Memory — modern GPU VRAM, stacked beside chip |
| **INT8** | 8-bit integer — used for quantized inference |
| **KV Cache** | Cached Key/Value tensors from previous tokens, avoids recomputation |
| **Kernel** | GPU function that runs on many threads in parallel |
| **L1/L2 Cache** | Automatic hardware cache between SRAM and VRAM |
| **Latency Hiding** | GPU switches to another warp while one waits for memory |
| **Memory Bound** | GPU bottleneck is data movement, not math |
| **Mixed Precision** | Training with FP16/BF16 forward + FP32 master weights |
| **MQA** | Multi-Query Attention — all heads share one K/V head |
| **NVLink** | Direct NVIDIA GPU-to-GPU high-bandwidth interconnect |
| **NVSwitch** | NVIDIA chip enabling all-to-all full bandwidth between 8 GPUs |
| **Occupancy** | Ratio of active warps to maximum warps per SM |
| **PCIe** | Standard CPU-GPU and GPU-GPU bus (slower than NVLink) |
| **Pipeline Parallelism** | Model layers split across GPUs |
| **Quantization** | Reducing parameter precision (FP32 → INT8/INT4) to save memory |
| **Register** | Per-thread ultra-fast memory, physically beside the ALU |
| **Residual Connection** | Skip connection: `output = x + SubLayer(x)` |
| **SIMT** | Single Instruction Multiple Threads — GPU execution model |
| **SM** | Streaming Multiprocessor — major compute block in GPU |
| **SRAM** | Static RAM — fast, expensive, small, used for shared memory/cache |
| **Speculative Decoding** | Small draft model proposes tokens, large model verifies in parallel |
| **Tensor Core** | Specialized GPU hardware for fast matrix multiplication |
| **Tensor Parallelism** | Weight matrices split across GPUs |
| **Thread Block** | Group of threads that share SRAM and can synchronize |
| **Transformer** | Neural network architecture based on attention mechanism |
| **VRAM** | Video RAM — GPU memory (the "24GB" on your GPU spec) |
| **Warp** | Group of 32 threads executing the same instruction simultaneously |
| **Warp Divergence** | Performance loss when threads in a warp take different code paths |
| **ZeRO** | Zero Redundancy Optimizer — shards optimizer states/gradients across GPUs |

---

*Complete notes covering GPU architecture, CUDA programming, Transformer systems, and LLM optimization for engineers building or understanding modern AI infrastructure.*