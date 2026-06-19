"""
Problem 2: Profile compute-bound vs. memory-bandwidth-bound regimes
           Using Nsight Systems (nsys) and CUDA profiling APIs

What this file teaches:
  - How to use CUDA profiling ranges to annotate your code for Nsight
  - How to isolate individual operations for profiling (matmul, attention, FFN)
  - How to compute arithmetic intensity and use it to predict bound regime
  - How to read roofline numbers from code without opening a GUI
  - What "memory-bound" and "compute-bound" mean in practice for attention

Two parts:
  Part A — CUDA profiling ranges: mark regions so nsys timeline is readable
  Part B — Software roofline: compute AI from FLOPs and bytes, classify bound

Requirements:
  pip install torch transformers
  # For actual Nsight profiling:
  # nsys profile --trace=cuda,nvtx python 02_profile_bound_regime.py
  # Then open the .nsys-rep file in Nsight Systems GUI, or:
  # nsys stats report.nsys-rep   (CLI summary)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — NVTX Ranges (Nsight annotation API)
#
# NVTX = NVIDIA Tools Extension Library
# Lets you add named time ranges to your code that appear as colored bars
# on the Nsight Systems timeline. Without NVTX, you just see "kernel" with
# no context about what it's doing.
#
# Usage in Nsight Systems:
#   nsys profile --trace=cuda,nvtx python this_script.py
#   → Open .nsys-rep in GUI
#   → NVTX row shows your named ranges aligned with CUDA kernel bars
#
# torch.cuda.nvtx.range_push / range_pop are PyTorch's wrappers.
# They're no-ops when not profiling — safe to leave in production code.
# ─────────────────────────────────────────────────────────────────────────────

class NVTXRange:
    """
    Context manager that wraps torch.cuda.nvtx for clean usage.

    Usage:
        with NVTXRange("attention_forward"):
            output = attention(q, k, v)

    In Nsight Systems timeline, this creates a labeled bar covering
    all CUDA kernels launched inside the `with` block.

    Color is optionally set via nvtx.range_push color argument —
    helps visually separate different operation types.
    """

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        # Pushes a named range onto the NVTX stack.
        # Ranges can be nested — inner ranges appear as children in the GUI.
        torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, *args):
        # Pops the most recently pushed range.
        torch.cuda.nvtx.range_pop()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Roofline Analysis Helper
#
# The Roofline Model classifies operations as either:
#   Compute-bound  — FLOPs are the bottleneck
#   Memory-bound   — HBM bandwidth is the bottleneck
#
# Key metric: Arithmetic Intensity (AI) = FLOPs / bytes_of_HBM_traffic
#
# If AI > ridge_point  →  compute-bound (GPU is maxing out Tensor Cores)
# If AI < ridge_point  →  memory-bound  (GPU is waiting for HBM)
#
# Ridge point = peak_flops / peak_bandwidth
#   For H100: 989 TFLOPs FP16 / 3.35 TB/s = ~295 FLOPs/byte
#   For A100: 312 TFLOPs FP16 / 2.0 TB/s  = ~156 FLOPs/byte
#   For RTX 4090: 165 TFLOPs FP16 / 1.0 TB/s = ~165 FLOPs/byte
# ─────────────────────────────────────────────────────────────────────────────

def get_gpu_roofline_params():
    """
    Return (peak_flops_per_sec, peak_bandwidth_bytes_per_sec, ridge_point)
    for the current GPU, based on known hardware specs.

    These are theoretical peaks — real kernels achieve 50–80% of these.
    The ridge point is what matters for classification.
    """
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {gpu_name}")

    # Known GPU specs (FP16, TFLOPs, memory bandwidth GB/s)
    # Add more GPUs here as needed.
    specs = {
        "H100":   (989e12,  3350e9),   # H100 SXM
        "A100":   (312e12,  2000e9),   # A100 80GB SXM
        "A10G":   (125e12,   600e9),   # A10G (AWS g5)
        "3090":   (142e12,   936e9),   # RTX 3090
        "4090":   (165e12,  1008e9),   # RTX 4090
        "4080":   ( 97e12,   736e9),   # RTX 4080
    }

    # Find matching spec by substring match on GPU name
    peak_flops = None
    peak_bw    = None
    for key, (flops, bw) in specs.items():
        if key in gpu_name:
            peak_flops = flops
            peak_bw    = bw
            break

    if peak_flops is None:
        # Fallback: use a conservative estimate for unknown GPUs
        print("  GPU not in known list — using conservative fallback (RTX 3090 class)")
        peak_flops = 100e12
        peak_bw    = 700e9

    ridge_point = peak_flops / peak_bw   # FLOPs per byte
    print(f"  Peak FLOPs: {peak_flops/1e12:.0f} TFLOPs")
    print(f"  Peak BW:    {peak_bw/1e9:.0f} GB/s")
    print(f"  Ridge point: {ridge_point:.1f} FLOPs/byte")
    print()
    return peak_flops, peak_bw, ridge_point


class OperationProfiler:
    """
    Profile a single operation: measure its actual FLOPs, bytes transferred,
    elapsed time, and compute vs. memory utilization.

    Usage:
        profiler = OperationProfiler("attention_naive")
        with profiler.profile():
            output = naive_attention(q, k, v)
        profiler.report(peak_flops, peak_bw, ridge_point)
    """

    def __init__(self, name: str):
        self.name = name
        self.elapsed_ms   = None
        self.flops        = None  # must be set before .report() call
        self.bytes_hbm    = None  # must be set before .report() call

    def profile(self, warmup_runs: int = 3, timed_runs: int = 20):
        """
        Returns a context manager. The enclosed code is run warmup_runs
        times (discarded) then timed_runs times (averaged).

        Important: this runs the enclosed code (warmup + timed) times total.
        Don't put side-effecting code (file writes etc.) inside.
        """
        # We use a generator-based context manager for the dual-phase timing.
        # This is more complex but gives cleaner usage at the call site.
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            # Warmup — discard timing
            for _ in range(warmup_runs):
                yield   # caller's body runs here
                torch.cuda.synchronize()

            # Timed runs
            times = []
            for _ in range(timed_runs):
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                yield       # caller's body runs again
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))

            self.elapsed_ms = sum(times) / len(times)

        # Note: Because a @contextmanager can only yield once per call,
        # for repeated runs in a benchmark we use the simpler timer below.
        # This method is illustrative — see time_operation() for actual usage.
        return _ctx()

    def report(self, peak_flops: float, peak_bw: float, ridge_point: float):
        """Print a roofline analysis report for this operation."""
        assert self.elapsed_ms is not None, "Must time the operation first"
        assert self.flops      is not None, "Must set .flops before calling report()"
        assert self.bytes_hbm  is not None, "Must set .bytes_hbm before calling report()"

        elapsed_s         = self.elapsed_ms / 1000.0
        achieved_flops    = self.flops    / elapsed_s     # FLOPs/sec achieved
        achieved_bw       = self.bytes_hbm / elapsed_s    # bytes/sec achieved
        arithmetic_intensity = self.flops / self.bytes_hbm  # FLOPs/byte

        # Utilization: how much of peak are we using?
        flop_utilization  = achieved_flops / peak_flops  * 100
        bw_utilization    = achieved_bw    / peak_bw     * 100

        # Classification
        is_compute_bound  = arithmetic_intensity >= ridge_point
        bound_label       = "COMPUTE-BOUND" if is_compute_bound else "MEMORY-BOUND"

        # Roofline limit: what's the max achievable throughput given our AI?
        # If memory-bound:  roofline_flops = AI × peak_bw
        # If compute-bound: roofline_flops = peak_flops
        roofline_flops    = min(arithmetic_intensity * peak_bw, peak_flops)
        roofline_pct      = achieved_flops / roofline_flops * 100

        print(f"\n{'─'*60}")
        print(f"  Operation: {self.name}")
        print(f"{'─'*60}")
        print(f"  Time:                  {self.elapsed_ms:.3f} ms")
        print(f"  FLOPs:                 {self.flops/1e9:.2f} GFLOPs")
        print(f"  HBM bytes:             {self.bytes_hbm/1e6:.2f} MB")
        print(f"  Arithmetic Intensity:  {arithmetic_intensity:.1f} FLOPs/byte")
        print(f"  Ridge point:           {ridge_point:.1f} FLOPs/byte")
        print(f"  → Regime:              {bound_label}")
        print(f"  Achieved FLOPs/s:      {achieved_flops/1e12:.2f} TFLOPs")
        print(f"  Peak FLOPs/s:          {peak_flops/1e12:.2f} TFLOPs")
        print(f"  FLOPs utilization:     {flop_utilization:.1f}%")
        print(f"  BW utilization:        {bw_utilization:.1f}%")
        print(f"  Roofline efficiency:   {roofline_pct:.1f}% of {bound_label.lower()} peak")


def time_operation(fn, warmup=3, runs=20):
    """
    Time a callable fn() using CUDA events.
    Returns average elapsed time in milliseconds.

    fn must launch CUDA kernels (or be a torch operation).
    """
    # Warmup
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    # Timed
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FLOPs and bytes calculations
#
# To classify an operation, you need to count:
#   FLOPs:      arithmetic operations performed
#   HBM bytes:  bytes read from + written to VRAM
#
# These are theoretical counts — the actual kernel may do slightly more
# (due to fused ops, register spilling, etc.) but these are accurate enough
# for roofline classification.
# ─────────────────────────────────────────────────────────────────────────────

def flops_matmul(M: int, N: int, K: int) -> int:
    """
    FLOPs for a matrix multiply C = A @ B where:
      A: [M × K],  B: [K × N],  C: [M × N]

    Each output element requires K multiplications and K-1 additions ≈ 2K FLOPs.
    Total: 2 × M × N × K
    """
    return 2 * M * N * K


def bytes_matmul(M: int, N: int, K: int, bytes_per_elem: int = 2) -> int:
    """
    HBM bytes for C = A @ B (FP16 → bytes_per_elem=2).

    Reads:  A [M×K] + B [K×N]
    Writes: C [M×N]
    Total:  (M×K + K×N + M×N) × bytes_per_elem

    Note: For large matmuls, the A+B reads are often cached; C write dominates.
    For small matmuls (fitting in L2/SRAM), actual HBM bytes are less.
    We use the "no caching" worst case here for conservative analysis.
    """
    return (M * K + K * N + M * N) * bytes_per_elem


def flops_naive_attention(N: int, d: int, H: int) -> int:
    """
    FLOPs for naive attention on H heads, sequence length N, head dim d.

    Per head:
      Q × Kᵀ:   2 × N × N × d FLOPs   (matrix multiply)
      softmax:   ~5 × N × N   FLOPs    (exp, sum, divide — small vs matmul)
      P × V:    2 × N × N × d FLOPs   (matrix multiply)

    Total per head: ≈ 4 × N² × d
    All H heads: 4 × H × N² × d = 4 × N² × (H × d) = 4 × N² × d_model
    """
    return 4 * H * N * N * d


def bytes_naive_attention(N: int, d: int, H: int, bytes_per_elem: int = 2) -> int:
    """
    HBM bytes for naive attention.

    The dominant cost is writing and reading the N×N attention matrix S (and P).
    Per head:
      Write S = Q × Kᵀ:       N × N × bytes
      Read S, write P = softmax(S): N × N × bytes  (read) + N × N × bytes (write)
      Read P, write O = P × V:    N × N × bytes  (read)

    Total per head: 4 × N² × bytes  (4 passes over the N×N matrix)
    All H heads:    4 × H × N² × bytes

    Plus the smaller Q, K, V, O reads/writes (O(N×d×H), minor for large N):
      Read Q, K, V:  3 × N × d × H × bytes
      Write O:         N × d × H × bytes
    """
    intermediate_bytes = 4 * H * N * N * bytes_per_elem   # N×N matrix cost
    qkvo_bytes = (3 + 1) * N * d * H * bytes_per_elem     # Q, K, V, O
    return intermediate_bytes + qkvo_bytes


def bytes_flash_attention(N: int, d: int, H: int, M_sram: int = 98304,
                          bytes_per_elem: int = 2) -> int:
    """
    HBM bytes for FlashAttention (no N×N matrix in HBM).

    FA reads Q, K, V from HBM in tiles and writes O.
    Each tile of K, V is loaded once per Q tile (for non-causal attention).

    Total reads:  Q (once) + K (Tr times) + V (Tr times)
                  where Tr = ceil(N / Br) is number of Q tiles

    For simplicity, we use the O(N × d) approximation:
      Read Q:   N × d × H × bytes
      Read K:   N × d × H × bytes × (Tr passes, but amortized ≈ 1× for large N)
      Read V:   N × d × H × bytes
      Write O:  N × d × H × bytes

    More precisely from the paper: Θ(N × d × H × Tr) for the K, V reads.
    But Tr = N/Br and Br ≈ M/(4d), so:
      K, V reads ≈ N × d × H × (N/Br) = N² × H × d / Br

    We compute both and note the full formula.
    """
    Br = M_sram // (4 * d * bytes_per_elem)   # tile size that fits in SRAM
    Br = max(Br, 1)
    Tr = math.ceil(N / Br)                    # number of Q tiles

    # Q read (once), O write (once)
    qo_bytes = 2 * N * d * H * bytes_per_elem
    # K read Tr times, V read Tr times
    kv_bytes = 2 * N * d * H * bytes_per_elem * Tr

    return qo_bytes + kv_bytes


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Kernel implementations to benchmark
# ─────────────────────────────────────────────────────────────────────────────

def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                    causal: bool = True) -> torch.Tensor:
    """
    Standard attention: materializes the full N×N score matrix in VRAM.

    Args:
        Q, K, V: [batch, heads, seq, head_dim]
        causal:  apply causal mask (decoder-style)
    Returns:
        output: [batch, heads, seq, head_dim]
    """
    d_head = Q.shape[-1]
    scale  = 1.0 / math.sqrt(d_head)

    # Score matrix: [batch, heads, seq, seq]
    # This is the tensor that FlashAttention avoids writing to HBM.
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale   # [B, H, N, N]

    if causal:
        N = Q.shape[-2]
        # Upper triangle is future tokens — mask to -inf
        mask = torch.triu(
            torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float('-inf'))

    # softmax row-wise: [B, H, N, N]
    # After this, `probs` still lives in VRAM — the whole N×N matrix.
    probs = torch.softmax(scores, dim=-1)

    # Weighted sum: [B, H, N, d_head]
    output = torch.matmul(probs, V)
    return output


def flash_attention_wrapper(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                             causal: bool = True) -> torch.Tensor:
    """
    FlashAttention via PyTorch 2.0's scaled_dot_product_attention (SDPA).

    SDPA automatically dispatches to FlashAttention when:
      - Input dtype is FP16 or BF16
      - Device is CUDA
      - No custom attention mask (or is_causal=True)
      - No dropout

    The N×N attention matrix is computed in SRAM tiles, never written to HBM.

    Args:
        Q, K, V: [batch, heads, seq, head_dim]  (same as naive_attention)
        causal:  use causal mask
    Returns:
        output: [batch, heads, seq, head_dim]
    """
    # SDPA expects: [batch, heads, seq, head_dim]
    return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)


def large_matmul(M: int = 4096, N: int = 4096, K: int = 4096,
                 device: str = "cuda", dtype=torch.float16) -> torch.Tensor:
    """
    A large matrix multiply to use as a compute-bound reference.

    Large GEMMs have high arithmetic intensity and are typically compute-bound.
    For M=N=K=4096, FP16:
      FLOPs:  2 × 4096³ ≈ 137 GFLOPs
      Bytes:  (4096² × 3) × 2 bytes ≈ 100 MB
      AI = 137e9 / 100e6 ≈ 1370 FLOPs/byte   >> ridge point → compute-bound
    """
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    return torch.matmul(A, B)


def elementwise_scale(x: torch.Tensor) -> torch.Tensor:
    """
    Pure elementwise operation to use as a memory-bound reference.

    Each element is read once and written once — minimal compute.
    FLOPs: N (one multiply)
    Bytes: 2N × 2 bytes (one read + one write of FP16)
    AI = N / (4N) = 0.25 FLOPs/byte   << ridge point → deeply memory-bound
    """
    return x * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Nsight-annotated profiling run
#
# This function wraps all operations with NVTX ranges so that when you run:
#   nsys profile --trace=cuda,nvtx python 02_profile_bound_regime.py
# you get a timeline where each operation is clearly labeled.
#
# In the Nsight Systems GUI:
#   - NVTX row shows colored bars for each named range
#   - CUDA row shows individual kernels (cuBLAS, flash_fwd, etc.)
#   - You can hover over kernels to see duration, SM utilization, etc.
# ─────────────────────────────────────────────────────────────────────────────

def run_with_nvtx_annotations(batch: int, heads: int, seq: int, d_head: int):
    """
    Run all operations with NVTX annotations for Nsight Systems profiling.

    When profiling with nsys, this produces a timeline with labeled regions.
    Without nsys, the NVTXRange calls are no-ops — safe to run normally.
    """
    device = "cuda"
    dtype  = torch.float16

    Q = torch.randn(batch, heads, seq, d_head, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq, d_head, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq, d_head, device=device, dtype=dtype)

    # ── Reference: deeply memory-bound (elementwise) ──────────────────────
    large_tensor = torch.randn(100_000_000, device=device, dtype=dtype)

    with NVTXRange("elementwise_membound"):
        # Elementwise op: AI ≈ 0.25 FLOPs/byte → deeply memory-bound
        # Nsight will show this as a very short kernel with high memory util
        _ = elementwise_scale(large_tensor)
        torch.cuda.synchronize()

    # ── Reference: compute-bound (large matmul) ────────────────────────────
    with NVTXRange("large_matmul_computebound"):
        # AI ≈ 1000+ FLOPs/byte → compute-bound, Tensor Cores near saturation
        _ = large_matmul()
        torch.cuda.synchronize()

    # ── Naive attention (memory-bound due to N×N HBM traffic) ─────────────
    with NVTXRange("naive_attention_seq{seq}".format(seq=seq)):
        # The Q×Kᵀ matmul is compute-bound, but softmax kernel and the
        # write/read of N×N matrix are memory-bound.
        # Overall: naive attention is bottlenecked by the N×N materialization.
        with NVTXRange("naive_attn_qkt"):
            # Can add sub-ranges for individual operations to see breakdown
            _ = naive_attention(Q, K, V)
        torch.cuda.synchronize()

    # ── FlashAttention (no N×N in HBM → more compute-bound) ──────────────
    with NVTXRange("flash_attention_seq{seq}".format(seq=seq)):
        # SDPA with FA backend: all computation stays in SRAM tiles
        # You will see FEWER kernel launches and SHORTER total time
        # Even though the computation is the same!
        _ = flash_attention_wrapper(Q, K, V)
        torch.cuda.synchronize()

    # ── FFN layers (for comparison — typically compute-bound) ─────────────
    d_model  = heads * d_head
    d_hidden = 4 * d_model   # standard 4× expansion
    W1 = torch.randn(d_model, d_hidden, device=device, dtype=dtype)
    W2 = torch.randn(d_hidden, d_model, device=device, dtype=dtype)
    x  = torch.randn(batch * seq, d_model, device=device, dtype=dtype)

    with NVTXRange("ffn_layer"):
        # FFN: two large matmuls + elementwise activation
        # Matmuls are compute-bound; GELU/SiLU is memory-bound
        with NVTXRange("ffn_linear1"):
            h = torch.matmul(x, W1)
        with NVTXRange("ffn_gelu"):
            h = F.gelu(h)
        with NVTXRange("ffn_linear2"):
            _ = torch.matmul(h, W2)
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Software roofline analysis
#
# This is the "without Nsight GUI" version of roofline analysis.
# Compute AI from our theoretical formulas, time the operations,
# then classify and print a full roofline report.
# ─────────────────────────────────────────────────────────────────────────────

def run_software_roofline():
    """
    Compute roofline metrics for attention variants and matmul.
    Prints a full classification table: memory-bound vs compute-bound.
    """
    peak_flops, peak_bw, ridge_point = get_gpu_roofline_params()

    device = "cuda"
    dtype  = torch.float16

    # Configurations to test
    configs = [
        # (batch, heads, seq_len, head_dim, label)
        (1, 32, 512,  128, "attention N=512"),
        (1, 32, 1024, 128, "attention N=1024"),
        (1, 32, 2048, 128, "attention N=2048"),
        (1, 32, 4096, 128, "attention N=4096"),
    ]

    print("\n" + "=" * 70)
    print("Software Roofline Analysis — Attention Variants")
    print("=" * 70)

    for batch, heads, seq, d_head, label in configs:

        Q = torch.randn(batch, heads, seq, d_head, device=device, dtype=dtype)
        K = torch.randn(batch, heads, seq, d_head, device=device, dtype=dtype)
        V = torch.randn(batch, heads, seq, d_head, device=device, dtype=dtype)

        # ── Naive attention ──────────────────────────────────────────────
        profiler_naive = OperationProfiler(f"naive_attention  ({label})")
        profiler_naive.flops     = flops_naive_attention(seq, d_head, heads)
        profiler_naive.bytes_hbm = bytes_naive_attention(seq, d_head, heads)
        profiler_naive.elapsed_ms = time_operation(
            lambda: naive_attention(Q, K, V), warmup=5, runs=20
        )
        profiler_naive.report(peak_flops, peak_bw, ridge_point)

        # ── FlashAttention ───────────────────────────────────────────────
        profiler_fa = OperationProfiler(f"flash_attention  ({label})")
        profiler_fa.flops     = flops_naive_attention(seq, d_head, heads)  # same FLOPs
        profiler_fa.bytes_hbm = bytes_flash_attention(seq, d_head, heads)  # fewer bytes
        profiler_fa.elapsed_ms = time_operation(
            lambda: flash_attention_wrapper(Q, K, V), warmup=5, runs=20
        )
        profiler_fa.report(peak_flops, peak_bw, ridge_point)

    # ── Reference: elementwise (memory-bound extreme) ────────────────────
    print("\n" + "─" * 70)
    print("  Reference operations for comparison")

    N_elem = 100_000_000   # 100M elements × 2 bytes = 200 MB
    x_large = torch.randn(N_elem, device=device, dtype=dtype)

    profiler_elem = OperationProfiler("elementwise ×0.5 (memory-bound reference)")
    profiler_elem.flops     = N_elem                    # 1 FLOP per element
    profiler_elem.bytes_hbm = N_elem * 2 * 2            # read + write, FP16
    profiler_elem.elapsed_ms = time_operation(
        lambda: elementwise_scale(x_large), warmup=5, runs=20
    )
    profiler_elem.report(peak_flops, peak_bw, ridge_point)

    # ── Reference: large matmul (compute-bound extreme) ─────────────────
    M = N = K = 4096
    profiler_mm = OperationProfiler("matmul 4096×4096×4096 (compute-bound reference)")
    profiler_mm.flops     = flops_matmul(M, N, K)
    profiler_mm.bytes_hbm = bytes_matmul(M, N, K)
    profiler_mm.elapsed_ms = time_operation(
        lambda: large_matmul(M, N, K), warmup=5, runs=20
    )
    profiler_mm.report(peak_flops, peak_bw, ridge_point)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Nsight usage instructions (printed at runtime)
# ─────────────────────────────────────────────────────────────────────────────

NSIGHT_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  HOW TO USE NSIGHT SYSTEMS WITH THIS SCRIPT                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Step 1: Install Nsight Systems                                            ║
║    https://developer.nvidia.com/nsight-systems                             ║
║    (pre-installed in NVIDIA NGC containers)                                ║
║                                                                            ║
║  Step 2: Profile this script                                               ║
║    nsys profile \\                                                          ║
║      --trace=cuda,nvtx,cudnn,cublas \\                                     ║
║      --gpu-metrics-device=all \\                                            ║
║      --output=attention_profile \\                                          ║
║      python 02_profile_bound_regime.py                                     ║
║                                                                            ║
║    Flags explained:                                                        ║
║      --trace=cuda      CUDA API calls (cudaMalloc, cudaLaunch, ...)        ║
║      --trace=nvtx      Your NVTXRange annotations (named bars)             ║
║      --trace=cudnn     cuDNN library calls (FA-2 uses cuDNN on some paths) ║
║      --trace=cublas    cuBLAS calls (every matmul, all attention)          ║
║      --gpu-metrics-device=all  Hardware counters: SM util, memory util     ║
║      --output=name     Output file prefix (.nsys-rep created)             ║
║                                                                            ║
║  Step 3: Open the report                                                   ║
║    Option A: nsight-sys attention_profile.nsys-rep   (GUI)                ║
║    Option B: nsys stats attention_profile.nsys-rep   (CLI summary)        ║
║                                                                            ║
║  Step 4: What to look for in the GUI timeline                             ║
║    Row "NVTX":    Your named ranges (colored bars)                        ║
║    Row "CUDA":    Individual kernel bars (hover for duration)             ║
║    Row "SM Act":  GPU SM utilization (%) — should be high for fast code   ║
║    Row "Mem BW":  Memory bandwidth utilization (%) — high = memory-bound  ║
║                                                                            ║
║  Key observations to make:                                                 ║
║    naive_attention:   You see THREE separate kernel groups                 ║
║                       (matmul → softmax → matmul)                         ║
║                       Between groups: cudaMemcpy or cudaDeviceSynchronize ║
║                       "Mem BW" row spikes during softmax kernel            ║
║                                                                            ║
║    flash_attention:   You see ONE kernel (flash_fwd or flash_attn_fwd)    ║
║                       No intermediate cudaMemcpy                          ║
║                       Kernel is shorter despite same FLOPs                ║
║                       "SM Act" row higher (GPU busier)                    ║
║                                                                            ║
║  Step 5: CLI stats (no GUI needed)                                        ║
║    nsys stats --report gputrace attention_profile.nsys-rep                ║
║    → Shows per-kernel time breakdown, sorted by duration                  ║
║    Look for: cutlass, flash_fwd, softmax, volta_sgemm etc.               ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — torch.profiler (alternative to Nsight, no install needed)
#
# torch.profiler is PyTorch's built-in profiler. Less detailed than Nsight
# but works without any additional installation. Good for quick checks.
# ─────────────────────────────────────────────────────────────────────────────

def run_torch_profiler(batch: int = 1, heads: int = 32,
                        seq: int = 2048, d_head: int = 128):
    """
    Profile using torch.profiler — PyTorch's built-in profiler.
    Outputs a trace viewable in Chrome's trace viewer (chrome://tracing)
    or TensorBoard.

    No Nsight install required. Less hardware detail but more accessible.
    """
    from torch.profiler import profile, ProfilerActivity, record_function

    device = "cuda"
    dtype  = torch.float16
    Q = torch.randn(batch, heads, seq, d_head, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq, d_head, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq, d_head, device=device, dtype=dtype)

    print(f"\nRunning torch.profiler (seq={seq}) ...")

    with profile(
        activities=[
            ProfilerActivity.CPU,   # CPU operations (kernel launches)
            ProfilerActivity.CUDA,  # CUDA kernels (actual GPU work)
        ],
        record_shapes=True,         # log tensor shapes (helpful for debugging)
        with_flops=True,            # estimate FLOPs where possible
        on_trace_ready=None,        # we'll print manually
    ) as prof:

        with record_function("naive_attention"):
            _ = naive_attention(Q, K, V)
            torch.cuda.synchronize()

        with record_function("flash_attention"):
            _ = flash_attention_wrapper(Q, K, V)
            torch.cuda.synchronize()

    # Print summary sorted by CUDA time
    print("\nTop 10 operations by CUDA time:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))

    # Export Chrome trace for visual inspection
    # Open chrome://tracing in Chrome, then load this file.
    trace_path = "/tmp/attention_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace saved to {trace_path}")
    print("Open chrome://tracing in Chrome and load the file to visualize.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    print(NSIGHT_INSTRUCTIONS)

    # Annotate operations with NVTX (only meaningful when running under nsys)
    print("Running NVTX-annotated forward passes ...")
    run_with_nvtx_annotations(batch=1, heads=32, seq=2048, d_head=128)
    print("NVTX annotations complete.")

    # Software roofline — works without Nsight
    run_software_roofline()

    # Optional: torch.profiler (no nsys needed)
    run_torch_profiler(batch=1, heads=32, seq=2048, d_head=128)


if __name__ == "__main__":
    main()