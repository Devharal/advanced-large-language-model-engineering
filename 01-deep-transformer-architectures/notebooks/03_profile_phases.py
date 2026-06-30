"""
Problem 3: Profile prefill (compute-bound) vs. decoding (memory-bandwidth-bound)

What this code teaches:
  - WHY prefill is compute-bound: large Q×Kᵀ matmul saturates Tensor Cores
  - WHY decode is memory-bound: tiny matmul, huge VRAM reads of weights + cache
  - How to measure achieved FLOPs/s and GB/s to CONFIRM which regime you're in
  - How to compute arithmetic intensity from first principles
  - How throughput (tokens/sec) and latency (ms/token) differ between phases
  - How batch size changes the regime (larger batch → decode becomes more compute-bound)
  - CUDA event timing, NVTX annotation, and torch.profiler for both phases

Requirements:
    pip install torch transformers
"""

import gc
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Theory: Why Each Phase Has a Different Bottleneck
#
# PREFILL:
#   Input: [batch, prompt_len, d_model]
#   Q×Kᵀ:  [batch, heads, prompt_len, prompt_len]  ← large matmul
#
#   FLOPs  = 4 × prompt_len² × d_model × batch            (quadratic in seq len)
#   Bytes  = 3 × batch × prompt_len × d_model × 2         (read Q,K,V once)
#          + batch × heads × prompt_len² × 2               (write attention matrix)
#   AI = FLOPs / Bytes  → large for long sequences
#
#   On H100: AI >> ridge point (~295 FLOPs/byte) → compute-bound
#   GPU compute utilization: ~70–90%
#   Memory bandwidth utilization: ~20–40%
#
# DECODE:
#   Input: [batch, 1, d_model]  ← ONE token at a time
#   Q×Kᵀ:  [batch, heads, 1, seq_len]  ← one row, all columns
#
#   FLOPs  ≈ 4 × 1 × seq_len × d_model × batch   (linear in seq len)
#   Bytes  = load weights (~14 GB for 7B)
#          + load KV cache (~seq_len × n_kv_heads × d_head × 2 bytes per layer)
#   AI = FLOPs / Bytes  → very small for batch=1
#
#   On H100: AI << ridge point → memory-bound
#   GPU compute utilization: ~5–15%
#   Memory bandwidth utilization: ~70–90%
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    d_model:    int = 2048
    n_heads:    int = 16
    n_kv_heads: int = 8          # GQA
    n_layers:   int = 24
    vocab_size: int = 32000
    ffn_mult:   int = 4          # FFN hidden = d_model × ffn_mult
    dtype:      torch.dtype = torch.float16

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def prefill_flops(self, prompt_len: int, batch: int = 1) -> int:
        """
        Total FLOPs for processing a prompt of `prompt_len` tokens.

        Attention FLOPs per layer:
          Q,K,V projections: 3 × 2 × batch × prompt_len × d_model × d_model
          Q×Kᵀ:              2 × batch × n_heads × prompt_len² × head_dim
          P×V:               2 × batch × n_heads × prompt_len² × head_dim
          Output proj:       2 × batch × prompt_len × d_model × d_model

        FFN FLOPs per layer:
          Up:    2 × batch × prompt_len × d_model × (ffn_mult × d_model)
          Down:  2 × batch × prompt_len × (ffn_mult × d_model) × d_model
        """
        d   = self.d_model
        H   = self.n_heads
        dk  = self.head_dim
        L   = self.n_layers
        N   = prompt_len
        B   = batch
        ffn = self.ffn_mult * d

        attn_proj = 4 * B * N * d * d    # Q,K,V,O projections (4 × 2BNd² / 2 = 4BNd²)
        attn_qk   = 2 * B * H * N * N * dk    # Q×Kᵀ
        attn_pv   = 2 * B * H * N * N * dk    # P×V
        ffn_flops = 2 * B * N * d * ffn + 2 * B * N * ffn * d   # up + down

        return L * (attn_proj + attn_qk + attn_pv + ffn_flops)

    def decode_flops(self, kv_len: int, batch: int = 1) -> int:
        """
        FLOPs for one decode step (generating one new token).

        The key: seq_len in the matmul is 1 (one new token), not kv_len.
        But attention must scan ALL kv_len cached keys.

        Q,K,V projections: applied to 1 new token
        Q×Kᵀ:             [1, kv_len] — one query over all keys
        P×V:               [1, kv_len] — weighted sum over all values
        """
        d  = self.d_model
        H  = self.n_heads
        dk = self.head_dim
        L  = self.n_layers
        S  = kv_len   # sequence length in cache
        B  = batch
        ffn = self.ffn_mult * d

        attn_proj = 4 * B * 1 * d * d       # project one token
        attn_qk   = 2 * B * H * 1 * S * dk  # one query × S keys
        attn_pv   = 2 * B * H * 1 * S * dk  # weighted sum over S values
        ffn_flops = 2 * B * 1 * d * ffn + 2 * B * 1 * ffn * d

        return L * (attn_proj + attn_qk + attn_pv + ffn_flops)

    def weight_bytes(self) -> int:
        """
        Total bytes of model parameters (loaded from VRAM every decode step).

        Every decode step must read ALL model weights — this is the dominant
        memory cost for decode at small batch sizes.
        """
        d   = self.d_model
        dk  = self.head_dim
        ffn = self.ffn_mult * d
        vocab = self.vocab_size

        # Per layer
        attn_weights = (
            self.n_heads    * dk * d  +    # W_Q
            self.n_kv_heads * dk * d  +    # W_K
            self.n_kv_heads * dk * d  +    # W_V
            d * d                          # W_O
        )
        ffn_weights = d * ffn + ffn * d   # W1, W2

        layer_bytes = (attn_weights + ffn_weights) * 2  # FP16 = 2 bytes

        embed_bytes  = vocab * d * 2
        lm_head_bytes = vocab * d * 2

        return self.n_layers * layer_bytes + embed_bytes + lm_head_bytes

    def kv_cache_bytes(self, kv_len: int, batch: int = 1) -> int:
        """Bytes to load KV cache for one decode step."""
        return (
            2                   # K and V
            * self.n_layers
            * self.n_kv_heads
            * kv_len
            * self.head_dim
            * 2                 # FP16
            * batch
        )

    def decode_bytes(self, kv_len: int, batch: int = 1) -> int:
        """Total VRAM bytes read per decode step: weights + KV cache."""
        return self.weight_bytes() + self.kv_cache_bytes(kv_len, batch)

    def arithmetic_intensity(self, prompt_len: int = None,
                              kv_len: int = None, batch: int = 1) -> float:
        """
        Arithmetic intensity = FLOPs / HBM_bytes.

        High AI (>> ridge_point) → compute-bound (prefill at long sequences)
        Low AI (<< ridge_point)  → memory-bound  (decode, especially batch=1)
        """
        if prompt_len is not None:
            # Prefill: bytes dominated by attention matrix + QKV reads
            flops = self.prefill_flops(prompt_len, batch)
            # Bytes: read Q,K,V weights (once), write+read attention matrix
            weight_b  = self.n_layers * 4 * self.d_model * self.d_model * 2  # QKV+O
            attn_b    = self.n_layers * self.n_heads * prompt_len ** 2 * 2    # N×N matrix
            total_bytes = batch * (weight_b + attn_b)
            return flops / max(total_bytes, 1)
        elif kv_len is not None:
            flops = self.decode_flops(kv_len, batch)
            total_bytes = self.decode_bytes(kv_len, batch)
            return flops / max(total_bytes, 1)
        raise ValueError("Provide either prompt_len or kv_len")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Minimal model for timing
# ─────────────────────────────────────────────────────────────────────────────

class AttentionLayer(nn.Module):
    """Single attention layer, cache-aware, supports prefill and decode."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Linear(cfg.d_model, cfg.n_heads    * cfg.head_dim, bias=False)
        self.W_K = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.W_V = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.W_O = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model,    bias=False)
        self.norm = nn.RMSNorm(cfg.d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor,
                past_kv: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple]:
        B, T, _ = x.shape
        H   = self.cfg.n_heads
        Hkv = self.cfg.n_kv_heads
        dk  = self.cfg.head_dim

        Q = self.W_Q(x).view(B, T, H,   dk).transpose(1, 2)
        K = self.W_K(x).view(B, T, Hkv, dk).transpose(1, 2)
        V = self.W_V(x).view(B, T, Hkv, dk).transpose(1, 2)

        # Append to KV cache if provided
        if past_kv is not None:
            past_K, past_V = past_kv
            K = torch.cat([past_K, K], dim=2)
            V = torch.cat([past_V, V], dim=2)

        new_kv = (K, V)

        # GQA: expand KV heads to match query heads
        if Hkv < H:
            g = H // Hkv
            K = K.repeat_interleave(g, dim=1)
            V = V.repeat_interleave(g, dim=1)

        # Attention
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=(past_kv is None))
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_O(out), new_kv


class MiniModel(nn.Module):
    """Multi-layer model for prefill vs decode profiling."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([AttentionLayer(cfg) for _ in range(cfg.n_layers)])
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, ids: torch.Tensor,
                past_kvs: list | None = None
                ) -> tuple[torch.Tensor, list]:
        x = self.embed(ids)
        new_kvs = []
        for i, layer in enumerate(self.layers):
            past = past_kvs[i] if past_kvs else None
            x, new_kv = layer(x, past_kv=past)
            new_kvs.append(new_kv)
        return self.head(x), new_kvs


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Timing utilities
# ─────────────────────────────────────────────────────────────────────────────

def cuda_time_ms(fn, warmup: int = 3, runs: int = 20) -> float:
    """
    Time a CUDA operation using events (accurate GPU-side timing).
    Returns average elapsed time in milliseconds.
    """
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return sum(times) / len(times)


def get_gpu_peak(device_name: str) -> tuple[float, float]:
    """
    Return (peak_tflops_fp16, peak_bandwidth_tbps) for known GPUs.
    Used to compute achieved utilization %.
    """
    specs = {
        "H100 SXM":  (989e12,  3.35e12),
        "H100 PCIe": (800e12,  2.0e12),
        "A100 SXM":  (312e12,  2.0e12),
        "A100 PCIe": (250e12,  2.0e12),
        "A10G":      (125e12,  0.6e12),
        "RTX 4090":  (165e12,  1.0e12),
        "RTX 3090":  (142e12,  0.936e12),
        "RTX 3080":  ( 90e12,  0.760e12),
        "V100":      ( 56e12,  0.9e12),
    }
    for key, (flops, bw) in specs.items():
        if key.lower() in device_name.lower():
            return flops, bw
    # Conservative fallback for unknown GPUs
    return 100e12, 0.7e12


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Prefill profiling
# ─────────────────────────────────────────────────────────────────────────────

def profile_prefill(model: MiniModel, cfg: ModelConfig,
                    prompt_lengths: list[int], batch: int = 1,
                    device: str = "cuda") -> list[dict]:
    """
    Profile the prefill phase across multiple prompt lengths.

    For each prompt_len:
      - Create dummy input tokens
      - Run forward pass (no cache — this IS the prefill)
      - Measure time, compute achieved FLOPs/s and GB/s
      - Classify as compute or memory bound

    Prefill is compute-bound because:
      - Large batches of tokens → big Q×Kᵀ matmul → Tensor Core saturation
      - Arithmetic intensity scales with prompt_len (more FLOPs per byte)
      - AI grows as O(N) → crosses ridge point at moderate N
    """
    results = []
    model.eval()

    for prompt_len in prompt_lengths:
        # Dummy token IDs
        ids = torch.randint(0, cfg.vocab_size, (batch, prompt_len), device=device)

        with torch.no_grad():
            elapsed_ms = cuda_time_ms(
                lambda: model(ids, past_kvs=None),
                warmup=5, runs=15
            )

        elapsed_s   = elapsed_ms / 1000.0
        flops       = cfg.prefill_flops(prompt_len, batch)
        achieved    = flops / elapsed_s
        ai          = cfg.arithmetic_intensity(prompt_len=prompt_len, batch=batch)

        results.append({
            "prompt_len":  prompt_len,
            "batch":       batch,
            "phase":       "prefill",
            "elapsed_ms":  elapsed_ms,
            "elapsed_s":   elapsed_s,
            "flops":       flops,
            "achieved_tflops": achieved / 1e12,
            "arith_intensity": ai,
            "tokens_per_sec":  (batch * prompt_len) / elapsed_s,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Decode profiling
# ─────────────────────────────────────────────────────────────────────────────

def profile_decode(model: MiniModel, cfg: ModelConfig,
                   kv_lengths: list[int], batch_sizes: list[int],
                   device: str = "cuda") -> list[dict]:
    """
    Profile the decode phase at various KV cache sizes and batch sizes.

    For each (kv_len, batch):
      - Build a fake KV cache of kv_len tokens
      - Run one decode step (forward pass on 1 new token)
      - Measure time, compute achieved bandwidth
      - Classify as memory-bound (it almost always is for batch=1)

    Decode is memory-bound because:
      - Only 1 new token → Q×Kᵀ is a tiny [1×kv_len] matmul
      - FLOPs are negligible vs weight + KV cache loads from VRAM
      - Arithmetic intensity ≈ d_model / (weight_size + cache_size) << ridge_point

    Batch size matters:
      - batch=1:  1 token of compute, same weight reads → deeply memory-bound
      - batch=32: 32 tokens of compute, same weight reads → less memory-bound
      - batch >> enough: transitions to compute-bound
    """
    results = []
    model.eval()

    for kv_len in kv_lengths:
        for batch in batch_sizes:
            # Build a fake KV cache (preloaded, like post-prefill state)
            # Shape: (past_K: [B, Hkv, kv_len, dk], past_V: same)
            past_kvs = [
                (
                    torch.randn(batch, cfg.n_kv_heads, kv_len, cfg.head_dim,
                                device=device, dtype=cfg.dtype),
                    torch.randn(batch, cfg.n_kv_heads, kv_len, cfg.head_dim,
                                device=device, dtype=cfg.dtype),
                )
                for _ in range(cfg.n_layers)
            ]

            # One new token to decode
            new_token = torch.randint(0, cfg.vocab_size, (batch, 1), device=device)

            with torch.no_grad():
                elapsed_ms = cuda_time_ms(
                    lambda: model(new_token, past_kvs=past_kvs),
                    warmup=5, runs=30
                )

            elapsed_s    = elapsed_ms / 1000.0
            flops        = cfg.decode_flops(kv_len, batch)
            bytes_moved  = cfg.decode_bytes(kv_len, batch)
            achieved_bw  = bytes_moved / elapsed_s    # bytes/sec
            achieved_fl  = flops / elapsed_s          # FLOPs/sec
            ai           = cfg.arithmetic_intensity(kv_len=kv_len, batch=batch)

            results.append({
                "kv_len":     kv_len,
                "batch":      batch,
                "phase":      "decode",
                "elapsed_ms": elapsed_ms,
                "flops":      flops,
                "bytes_hbm":  bytes_moved,
                "achieved_tbps":   achieved_bw / 1e12,
                "achieved_tflops": achieved_fl / 1e12,
                "arith_intensity": ai,
                "tokens_per_sec":  batch / elapsed_s,     # throughput
                "ms_per_token":    elapsed_ms / batch,    # latency
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Arithmetic intensity sweep
#
# This shows how AI changes across both phases and batch sizes, and
# where the transition from memory-bound to compute-bound happens.
# ─────────────────────────────────────────────────────────────────────────────

def compute_ai_sweep(cfg: ModelConfig) -> dict:
    """
    Compute theoretical arithmetic intensity for all configurations.
    No GPU needed — pure arithmetic from the formula.

    Returns:
        dict with AI values for prefill and decode at various settings.
    """
    prefill_ai = {}
    for prompt_len in [128, 256, 512, 1024, 2048, 4096, 8192]:
        for batch in [1, 8, 32]:
            ai = cfg.arithmetic_intensity(prompt_len=prompt_len, batch=batch)
            prefill_ai[(prompt_len, batch)] = ai

    decode_ai = {}
    for kv_len in [128, 512, 2048, 8192, 32768]:
        for batch in [1, 4, 16, 64, 256]:
            ai = cfg.arithmetic_intensity(kv_len=kv_len, batch=batch)
            decode_ai[(kv_len, batch)] = ai

    return {"prefill": prefill_ai, "decode": decode_ai}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — NVTX annotation for Nsight profiling
# ─────────────────────────────────────────────────────────────────────────────

def run_nvtx_annotated_comparison(model: MiniModel, cfg: ModelConfig,
                                   prompt_len: int = 512,
                                   n_decode_steps: int = 20,
                                   batch: int = 1,
                                   device: str = "cuda"):
    """
    Run prefill + decode with NVTX annotations so Nsight Systems
    shows labeled regions on the timeline.

    Run with:
        nsys profile --trace=cuda,nvtx python 03_profile_phases.py
    Then open the .nsys-rep file and look at the NVTX row.

    What you'll see:
        "prefill_N512":  One long kernel burst — Tensor Cores active, BW low
        "decode_step_N": Many short iterations — BW high, SM utilization low
        "decode_total":  Much longer wall time per-token than prefill
    """
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (batch, prompt_len), device=device)

    with torch.no_grad():

        # ── PREFILL ────────────────────────────────────────────────────────────
        # This is the compute-bound phase.
        # In Nsight: you'll see Tensor Core kernels (cublasHgemm/cutlass) dominating.
        # SM utilization should be high (~60–80%).
        torch.cuda.nvtx.range_push(f"prefill_N{prompt_len}_B{batch}")
        logits, past_kvs = model(ids, past_kvs=None)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()

        # ── DECODE ────────────────────────────────────────────────────────────
        # Each decode step is memory-bound.
        # In Nsight: you'll see shorter kernels with high memory bandwidth usage.
        # SM utilization should be low (~5–15%) between heavy VRAM reads.
        torch.cuda.nvtx.range_push(f"decode_total_{n_decode_steps}steps")
        current_token = logits[:, -1:, :].argmax(dim=-1)

        for step in range(n_decode_steps):
            # Sub-range per step lets you see per-token breakdown in Nsight
            torch.cuda.nvtx.range_push(f"decode_step_{step}")
            logits, past_kvs = model(current_token, past_kvs=past_kvs)
            current_token    = logits[:, -1:, :].argmax(dim=-1)
            torch.cuda.nvtx.range_pop()   # decode_step_{step}

        torch.cuda.nvtx.range_pop()   # decode_total
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — torch.profiler comparison
#
# Runs both phases under torch.profiler and prints the operator breakdown.
# Shows which CUDA kernels dominate in each phase.
# ─────────────────────────────────────────────────────────────────────────────

def run_torch_profiler_comparison(model: MiniModel, cfg: ModelConfig,
                                   prompt_len: int = 256,
                                   n_decode_steps: int = 10,
                                   batch: int = 1,
                                   device: str = "cuda"):
    """
    Profile both phases with torch.profiler and print operator breakdown.

    Key operators to look for:

    PREFILL:
        aten::mm / aten::bmm / aten::linear  → matrix multiplications
        aten::scaled_dot_product_attention    → fused attention (FA-2)
        These dominate because prompt_len is large.

    DECODE:
        aten::mm / aten::linear               → still present, but FAST
        aten::cat                             → KV cache append
        Total CUDA time much shorter per token, but wall time dominated by
        memory-bound reads (not visible as separate ops in profiler).
    """
    from torch.profiler import profile, ProfilerActivity, record_function

    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (batch, prompt_len), device=device)

    # ── Profile prefill ───────────────────────────────────────────────────────
    print("\n[Prefill phase — torch.profiler]")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True, with_flops=True
    ) as prof_prefill:
        with torch.no_grad():
            with record_function("prefill"):
                logits, past_kvs = model(ids, past_kvs=None)
                torch.cuda.synchronize()

    print(prof_prefill.key_averages().table(
        sort_by="cuda_time_total", row_limit=8
    ))

    # ── Profile decode ────────────────────────────────────────────────────────
    print("\n[Decode phase — torch.profiler]")
    current_token = logits[:, -1:, :].argmax(dim=-1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof_decode:
        with torch.no_grad():
            with record_function("decode"):
                for _ in range(n_decode_steps):
                    logits, past_kvs  = model(current_token, past_kvs=past_kvs)
                    current_token     = logits[:, -1:, :].argmax(dim=-1)
                torch.cuda.synchronize()

    print(prof_decode.key_averages().table(
        sort_by="cuda_time_total", row_limit=8
    ))

    # Export for Chrome trace viewer
    prof_prefill.export_chrome_trace("/tmp/trace_prefill.json")
    prof_decode.export_chrome_trace("/tmp/trace_decode.json")
    print("\nChrome traces saved to /tmp/trace_prefill.json and /tmp/trace_decode.json")
    print("Open chrome://tracing and load each file to compare visually.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Report printing
# ─────────────────────────────────────────────────────────────────────────────

def print_ai_sweep_table(ai_data: dict, ridge_point: float = 295.0):
    """Print arithmetic intensity for all configs."""
    print("\n" + "═" * 75)
    print(f"Arithmetic Intensity (FLOPs/byte)  |  Ridge point: {ridge_point:.0f} FLOPs/byte")
    print(f"  > Ridge → COMPUTE-BOUND  |  < Ridge → MEMORY-BOUND")
    print("═" * 75)

    print("\n  PREFILL PHASE:")
    print(f"  {'prompt_len':>12} {'batch':>8} {'AI FLOPs/B':>14} {'regime':>16}")
    print(f"  {'─'*12} {'─'*8} {'─'*14} {'─'*16}")
    for (pl, b), ai in sorted(ai_data["prefill"].items()):
        regime = "COMPUTE-BOUND" if ai > ridge_point else "memory-bound"
        flag   = " ←" if ai > ridge_point else ""
        print(f"  {pl:>12,} {b:>8} {ai:>14.1f} {regime:>16}{flag}")

    print("\n  DECODE PHASE:")
    print(f"  {'kv_len':>12} {'batch':>8} {'AI FLOPs/B':>14} {'regime':>16}")
    print(f"  {'─'*12} {'─'*8} {'─'*14} {'─'*16}")
    for (kv, b), ai in sorted(ai_data["decode"].items()):
        regime = "compute-bound" if ai > ridge_point else "MEMORY-BOUND"
        flag   = " ←" if ai < ridge_point else ""
        print(f"  {kv:>12,} {b:>8} {ai:>14.2f} {regime:>16}{flag}")


def print_prefill_results(results: list[dict], peak_tflops: float):
    """Print prefill timing results."""
    print("\n" + "═" * 75)
    print("Prefill Phase — Timing Results")
    print("═" * 75)
    print(f"  {'prompt_len':>12} {'batch':>6} {'ms':>8} {'TFLOPs/s':>12} "
          f"{'peak %':>8} {'tok/s':>10}")
    print(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*12} {'─'*8} {'─'*10}")
    for r in results:
        pct = r["achieved_tflops"] / (peak_tflops / 1e12) * 100
        print(f"  {r['prompt_len']:>12,} {r['batch']:>6} {r['elapsed_ms']:>8.2f} "
              f"{r['achieved_tflops']:>12.2f} {pct:>7.1f}% {r['tokens_per_sec']:>10.0f}")
    print(f"""
  Key observation:
    AI grows with prompt_len → longer prompts are more compute-bound
    GPU utilization (peak %) rises with prompt_len
    tokens/sec is HIGH for prefill (many tokens processed in parallel)
    """)


def print_decode_results(results: list[dict], peak_tbps: float):
    """Print decode timing results."""
    print("\n" + "═" * 75)
    print("Decode Phase — Timing Results")
    print("═" * 75)
    print(f"  {'kv_len':>10} {'batch':>6} {'ms':>8} {'ms/tok':>8} "
          f"{'TB/s':>8} {'BW %':>8} {'tok/s':>10}")
    print(f"  {'─'*10} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
    for r in results:
        bw_pct = r["achieved_tbps"] / (peak_tbps / 1e12) * 100
        print(f"  {r['kv_len']:>10,} {r['batch']:>6} {r['elapsed_ms']:>8.2f} "
              f"{r['ms_per_token']:>8.3f} {r['achieved_tbps']:>8.3f} "
              f"{bw_pct:>7.1f}% {r['tokens_per_sec']:>10.1f}")
    print(f"""
  Key observations:
    Memory bandwidth utilization (BW %) is HIGH — confirms memory-bound
    ms/token is HIGH at batch=1 — GPU is underutilized
    Larger batch → more tokens/step → better bandwidth amortization
    kv_len increasing → more cache to read → latency grows slightly
    """)


def print_phase_comparison_summary(prefill_res: list[dict],
                                    decode_res: list[dict]):
    """Print a side-by-side comparison of the two phases."""
    print("\n" + "═" * 75)
    print("Phase Comparison Summary")
    print("═" * 75)

    # Pick representative numbers
    pf  = next((r for r in prefill_res if r["prompt_len"] == 512 and r["batch"] == 1), prefill_res[0])
    dc1 = next((r for r in decode_res  if r["kv_len"] == 512 and r["batch"] == 1), decode_res[0])
    dc8 = next((r for r in decode_res  if r["kv_len"] == 512 and r["batch"] == 8), decode_res[-1])

    print(f"""
  ┌──────────────────────────┬───────────────────────┬───────────────────────┐
  │                          │  PREFILL              │  DECODE               │
  │                          │  (prompt_len=512)     │  (kv_len=512)         │
  ├──────────────────────────┼───────────────────────┼───────────────────────┤
  │ Batch size               │       1               │    1          8       │
  │ Tokens processed/step    │     512               │    1          8       │
  │ Wall time (ms)           │  {pf['elapsed_ms']:6.1f}               │ {dc1['elapsed_ms']:5.2f}       {dc8['elapsed_ms']:6.2f}       │
  │ Time per token (ms)      │   {pf['elapsed_ms']/512:5.3f}               │ {dc1['ms_per_token']:5.3f}       {dc8['ms_per_token']:6.3f}       │
  │ AI (FLOPs/byte)          │  {pf['arith_intensity']:6.1f}               │ {dc1['arith_intensity']:5.2f}        {dc8['arith_intensity']:5.2f}       │
  │ Achieved TFLOPs/s        │  {pf['achieved_tflops']:6.2f}               │ {dc1['achieved_tflops']:5.3f}       {dc8['achieved_tflops']:6.3f}       │
  │ Bottleneck               │  Compute (Tensor Core)│  Memory BW    Memory BW│
  └──────────────────────────┴───────────────────────┴───────────────────────┘

  Why the difference?
    Prefill:  512 tokens × 512 keys = large Q×Kᵀ matmul → Tensor Core saturation
    Decode:   1  token  × 512 keys = trivial matmul → spend 99% loading weights

  Why batch=8 helps decode:
    Same weights loaded (one read), but 8 tokens computed → 8× throughput
    AI increases from {dc1['arith_intensity']:.2f} to {dc8['arith_intensity']:.2f} FLOPs/byte (still memory-bound, but less so)
    """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        peak_tflops, peak_tbps = get_gpu_peak(gpu_name)
        ridge = peak_tflops / peak_tbps
        print(f"GPU:          {gpu_name}")
        print(f"Peak FLOPs:   {peak_tflops/1e12:.0f} TFLOPs (FP16)")
        print(f"Peak BW:      {peak_tbps/1e12:.2f} TB/s")
        print(f"Ridge point:  {ridge:.1f} FLOPs/byte")
    else:
        peak_tflops, peak_tbps = 10e12, 0.1e12
        ridge = peak_tflops / peak_tbps
        print("Warning: running on CPU, timing numbers won't reflect GPU behavior")

    # Small model for fast experimentation
    # Scale up n_layers/d_model/n_heads for more realistic numbers
    cfg = ModelConfig(
        d_model=512, n_heads=8, n_kv_heads=4,
        n_layers=6, vocab_size=4096, dtype=torch.float16

    print(f"\nModel: {cfg.n_layers}L × d{cfg.d_model} × {cfg.n_heads}H (kv:{cfg.n_kv_heads})")
    print(f"Weights: {cfg.weight_bytes()/1e6:.1f} MB")

    model = MiniModel(cfg).to(device=device, dtype=cfg.dtype)
    model.eval()

    # ── Part A: Arithmetic Intensity Analysis (no GPU needed) ─────────────────
    print("\n[Part A] Arithmetic Intensity Analysis")
    ai_data = compute_ai_sweep(cfg)
    print_ai_sweep_table(ai_data, ridge_point=ridge)

    if device == "cuda":
        # ── Part B: Prefill timing ─────────────────────────────────────────────
        print("\n[Part B] Profiling Prefill Phase")
        prefill_results = profile_prefill(
            model, cfg,
            prompt_lengths=[64, 128, 256, 512, 1024],
            batch=1, device=device
        )
        print_prefill_results(prefill_results, peak_tflops)

        # ── Part C: Decode timing ──────────────────────────────────────────────
        print("\n[Part C] Profiling Decode Phase")
        decode_results = profile_decode(
            model, cfg,
            kv_lengths=[128, 512, 1024, 2048],
            batch_sizes=[1, 4, 8],
            device=device
        )
        print_decode_results(decode_results, peak_tbps)

        # ── Part D: Side-by-side comparison ────────────────────────────────────
        print_phase_comparison_summary(prefill_results, decode_results)

        # ── Part E: NVTX annotations (run under nsys to see effect) ────────────
        print("\n[Part E] Running NVTX-annotated forward passes ...")
        print("(Run under: nsys profile --trace=cuda,nvtx python 03_profile_phases.py)")
        run_nvtx_annotated_comparison(
            model, cfg, prompt_len=256, n_decode_steps=10, batch=1, device=device
        )
        print("NVTX passes complete.")

        # ── Part F: torch.profiler ─────────────────────────────────────────────
        print("\n[Part F] torch.profiler operator breakdown")
        run_torch_profiler_comparison(
            model, cfg, prompt_len=128, n_decode_steps=5, batch=1, device=device
        )

    print("\n" + "═" * 75)
    print("Nsight Systems profiling commands:")
    print("═" * 75)
    print("""
  Full profile:
    nsys profile \\
      --trace=cuda,nvtx,cublas \\
      --gpu-metrics-device=all \\
      --output=phase_profile \\
      python 03_profile_phases.py

  Open report:
    nsight-sys phase_profile.nsys-rep

  CLI stats:
    nsys stats phase_profile.nsys-rep

  What to look for in the timeline:
    "prefill_N256":  dense Tensor Core activity, high SM utilization
    "decode_step_0": sparse compute, long gaps = waiting for VRAM reads
    "decode_total":  per-step time grows slightly as KV cache grows
    """)


if __name__ == "__main__":
    main()