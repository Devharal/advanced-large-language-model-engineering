"""
============================================================
  Attention Deep Dive — Pure PyTorch Implementation
  Covers:
    1. Scaled dot-product attention (no libraries)
    2. Causal masking + future-token leak verification
    3. FLOPs vs sequence-length benchmark (quadratic growth)
    4. MHA / GQA / MQA variants — KV cache memory & throughput
    5. MHA → GQA checkpoint conversion (Llama-3 style head averaging)
    6. GQA vs MHA inference latency at long contexts
============================================================
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

# ── reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")


# ════════════════════════════════════════════════════════════════════════════
# PART 1 — Scaled Dot-Product Attention (pure PyTorch, no F.scaled_dot_product)
# ════════════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    q: torch.Tensor,           # (B, H, T, D)
    k: torch.Tensor,           # (B, H, S, D)
    v: torch.Tensor,           # (B, H, S, Dv)
    mask: Optional[torch.Tensor] = None,   # broadcastable bool/float mask
    dropout_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch scaled dot-product attention.
    Returns (output, attention_weights).
    """
    d_k = q.size(-1)
    scale = math.sqrt(d_k)

    # (B, H, T, S)  — raw attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale

    if mask is not None:
        # mask=True  → positions to MASK OUT (set to -inf)
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)

    # Numerical guard: rows that were fully masked become NaN after softmax;
    # replace with 0 so they don't poison the output.
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    if dropout_p > 0.0 and torch.is_grad_enabled():
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    output = torch.matmul(attn_weights, v)   # (B, H, T, Dv)
    return output, attn_weights


# ════════════════════════════════════════════════════════════════════════════
# PART 2 — Causal Mask + Future-Token Leak Verification
# ════════════════════════════════════════════════════════════════════════════

def make_causal_mask(seq_len: int, device: str = "cpu") -> torch.Tensor:
    """
    Upper-triangular mask (excluding diagonal).
    mask[i, j] = True  means token i cannot attend to token j.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    return mask  # (T, T)


def verify_no_future_leakage(seq_len: int = 8) -> None:
    """
    Checks that with a causal mask, attention weights on future
    positions are exactly zero for every query position.
    """
    print("=" * 60)
    print("PART 2 — Causal Mask & Future-Token Leak Verification")
    print("=" * 60)

    B, H, T, D = 1, 1, seq_len, 16
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    causal = make_causal_mask(T)            # (T, T)
    causal_4d = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    _, attn_w = scaled_dot_product_attention(q, k, v, mask=causal_4d)

    # attn_w shape: (B, H, T, T)
    # For each query i, weights at positions j>i must be 0
    upper = torch.triu(attn_w[0, 0], diagonal=1)
    max_future_weight = upper.abs().max().item()

    print(f"Sequence length : {T}")
    print(f"Max attention weight on future tokens : {max_future_weight:.2e}")
    assert max_future_weight == 0.0, "LEAK DETECTED — future tokens have non-zero weight!"
    print("✓ No future-token leakage confirmed\n")

    print("Attention weight matrix (query × key):")
    print(torch.round(attn_w[0, 0] * 100) / 100)
    print()


# ════════════════════════════════════════════════════════════════════════════
# PART 3 — FLOPs vs Sequence Length (quadratic growth)
# ════════════════════════════════════════════════════════════════════════════

def estimate_attention_flops(B: int, H: int, T: int, D: int) -> int:
    """
    Theoretical FLOPs for scaled dot-product attention.
      QK^T      : 2 * B * H * T * T * D
      softmax   : ~5 * B * H * T * T  (approx, ignored vs matmul)
      AV        : 2 * B * H * T * T * D
    Total ≈ 4 * B * H * T^2 * D
    """
    return 4 * B * H * T * T * D


def benchmark_flops_vs_seqlen() -> None:
    print("=" * 60)
    print("PART 3 — FLOPs vs Sequence Length (quadratic)")
    print("=" * 60)

    B, H, D = 1, 8, 64
    seq_lens = [64, 128, 256, 512, 1024, 2048]

    print(f"{'SeqLen':>8} {'Theory GFLOPs':>16} {'Ratio to prev':>16} {'Wall ms':>10}")
    print("-" * 55)

    prev_flops = None
    for T in seq_lens:
        q = torch.randn(B, H, T, D, device=DEVICE)
        k = torch.randn(B, H, T, D, device=DEVICE)
        v = torch.randn(B, H, T, D, device=DEVICE)
        mask = make_causal_mask(T, device=DEVICE).unsqueeze(0).unsqueeze(0)

        # warm-up
        _ = scaled_dot_product_attention(q, k, v, mask)

        # time
        reps = max(1, 200 // (T // 64))
        start = time.perf_counter()
        for _ in range(reps):
            scaled_dot_product_attention(q, k, v, mask)
        elapsed_ms = (time.perf_counter() - start) / reps * 1e3

        flops = estimate_attention_flops(B, H, T, D)
        gflops = flops / 1e9
        ratio = f"{flops / prev_flops:.2f}x" if prev_flops else "   —"
        prev_flops = flops
        print(f"{T:>8} {gflops:>16.4f} {ratio:>16} {elapsed_ms:>10.3f}")

    print("\n  ↑ Doubling seq_len ≈ 4× GFLOPs (quadratic O(T²))\n")


# ════════════════════════════════════════════════════════════════════════════
# PART 4 — MHA / GQA / MQA — KV Cache Memory & Throughput
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class AttentionConfig:
    d_model: int = 512
    n_q_heads: int = 8       # query heads
    n_kv_heads: int = 8      # key/value heads  (MHA=8, GQA=2, MQA=1)
    d_head: int = 64
    max_seq_len: int = 1024


class MultiHeadAttention(nn.Module):
    """MHA / GQA / MQA unified implementation."""

    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        assert cfg.n_q_heads % cfg.n_kv_heads == 0, \
            "n_q_heads must be divisible by n_kv_heads"

        self.n_q   = cfg.n_q_heads
        self.n_kv  = cfg.n_kv_heads
        self.n_rep = cfg.n_q_heads // cfg.n_kv_heads   # repetitions for GQA
        self.d_h   = cfg.d_head
        self.d_model = cfg.d_model

        self.Wq = nn.Linear(cfg.d_model, cfg.n_q_heads  * cfg.d_head, bias=False)
        self.Wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.Wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.Wo = nn.Linear(cfg.n_q_heads * cfg.d_head, cfg.d_model,  bias=False)

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, D)
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_causal_mask: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape

        # ── project ──────────────────────────────────────────────────────
        q = self.Wq(x).view(B, T, self.n_q,  self.d_h).transpose(1, 2)  # (B, Hq, T, D)
        k = self.Wk(x).view(B, T, self.n_kv, self.d_h).transpose(1, 2)  # (B, Hkv, T, D)
        v = self.Wv(x).view(B, T, self.n_kv, self.d_h).transpose(1, 2)

        # ── KV cache append ───────────────────────────────────────────────
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        new_cache = (k, v)
        S = k.size(2)  # full key/value sequence length

        # ── GQA head expansion (repeat KV heads) ──────────────────────────
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, self.n_kv, self.n_rep, S, self.d_h) \
                 .reshape(B, self.n_q, S, self.d_h)
            v = v.unsqueeze(2).expand(B, self.n_kv, self.n_rep, S, self.d_h) \
                 .reshape(B, self.n_q, S, self.d_h)

        # ── causal mask ───────────────────────────────────────────────────
        mask = None
        if use_causal_mask:
            # query tokens (T) attend to key tokens (S); past tokens only
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device)
            mask = torch.triu(mask, diagonal=S - T + 1)          # causal
            mask = mask.unsqueeze(0).unsqueeze(0)                  # (1,1,T,S)

        out, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.Wo(out), new_cache


def kv_cache_memory_bytes(
    n_kv_heads: int, d_head: int, seq_len: int,
    batch: int = 1, dtype_bytes: int = 2   # fp16
) -> int:
    """Memory for a single KV cache (both K and V)."""
    return 2 * batch * n_kv_heads * seq_len * d_head * dtype_bytes


def benchmark_variants() -> None:
    print("=" * 60)
    print("PART 4 — MHA / GQA / MQA: KV Cache Memory & Throughput")
    print("=" * 60)

    d_model, d_head = 512, 64
    n_q = 8
    seq_lens = [256, 512, 1024, 2048]
    variants = {"MHA": 8, "GQA(2kv)": 2, "MQA": 1}

    # ── KV cache memory table ─────────────────────────────────────────────
    print("\n── KV Cache Memory (MB, fp16, batch=1) ──")
    header = f"{'SeqLen':>8}" + "".join(f"{name:>14}" for name in variants)
    print(header)
    print("-" * (8 + 14 * len(variants)))

    for T in seq_lens:
        row = f"{T:>8}"
        for name, n_kv in variants.items():
            mb = kv_cache_memory_bytes(n_kv, d_head, T) / 1e6
            row += f"{mb:>14.3f}"
        print(row)

    # ── throughput (tokens/sec) ───────────────────────────────────────────
    print("\n── Throughput: tokens/sec (decode — single token, ctx=512) ──")
    ctx_len = 512
    B = 1
    reps = 100

    header2 = f"{'Variant':>12} {'Hq':>5} {'Hkv':>5} {'tok/s':>12} {'rel. MHA':>12}"
    print(header2)
    print("-" * 50)

    ref_tps = None
    for name, n_kv in variants.items():
        cfg = AttentionConfig(d_model=d_model, n_q_heads=n_q,
                              n_kv_heads=n_kv, d_head=d_head)
        model = MultiHeadAttention(cfg).to(DEVICE).eval()

        # build KV cache from context
        ctx = torch.randn(B, ctx_len, d_model, device=DEVICE)
        with torch.no_grad():
            _, cache = model(ctx, kv_cache=None)

        # single decode step
        tok = torch.randn(B, 1, d_model, device=DEVICE)
        with torch.no_grad():
            _ = model(tok, kv_cache=cache)   # warm-up

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(reps):
                model(tok, kv_cache=cache)
        elapsed = time.perf_counter() - start
        tps = reps / elapsed

        rel = f"{tps / ref_tps:.2f}x" if ref_tps else "1.00x"
        if ref_tps is None:
            ref_tps = tps
        print(f"{name:>12} {n_q:>5} {n_kv:>5} {tps:>12.1f} {rel:>12}")
    print()


# ════════════════════════════════════════════════════════════════════════════
# PART 5 — MHA → GQA Checkpoint Conversion (Llama-3 head-averaging)
# ════════════════════════════════════════════════════════════════════════════

def convert_mha_to_gqa(
    mha_state: dict,
    n_q_heads_src: int,
    n_kv_heads_dst: int,
    d_head: int,
) -> dict:
    """
    Convert MHA weight tensors to GQA by averaging groups of Q/K/V heads.
    Mimics what Meta did when releasing Llama-3 GQA variants.

    For Wk, Wv: group consecutive heads, average within each group.
    For Wq:     keep unchanged (one Q head per group → GQA definition).
    """
    assert n_q_heads_src % n_kv_heads_dst == 0
    group_size = n_q_heads_src // n_kv_heads_dst
    new_state = {}

    for key, tensor in mha_state.items():
        if key in ("Wk.weight", "Wv.weight"):
            # tensor: (n_q_heads * d_head, d_model) — reshape then average
            d_model = tensor.size(1)
            heads = tensor.view(n_q_heads_src, d_head, d_model)      # (H, d_h, D)
            groups = heads.view(n_kv_heads_dst, group_size, d_head, d_model)
            averaged = groups.mean(dim=1)                             # (Hkv, d_h, D)
            new_state[key] = averaged.view(n_kv_heads_dst * d_head, d_model)
        else:
            new_state[key] = tensor.clone()

    return new_state


def demo_checkpoint_conversion() -> None:
    print("=" * 60)
    print("PART 5 — MHA → GQA Checkpoint Conversion")
    print("=" * 60)

    n_q, n_kv_dst, d_head, d_model = 8, 2, 64, 512

    # ── train / create a fake MHA checkpoint ─────────────────────────────
    mha_cfg = AttentionConfig(d_model=d_model, n_q_heads=n_q,
                              n_kv_heads=n_q, d_head=d_head)
    mha = MultiHeadAttention(mha_cfg)
    mha_state = {
        "Wq.weight": mha.Wq.weight.data.clone(),
        "Wk.weight": mha.Wk.weight.data.clone(),
        "Wv.weight": mha.Wv.weight.data.clone(),
        "Wo.weight": mha.Wo.weight.data.clone(),
    }

    print(f"Source MHA  — Wk shape: {mha_state['Wk.weight'].shape}")

    # ── convert ───────────────────────────────────────────────────────────
    gqa_state = convert_mha_to_gqa(mha_state, n_q, n_kv_dst, d_head)

    print(f"Target GQA  — Wk shape: {gqa_state['Wk.weight'].shape}")
    print(f"Head reduction: {n_q} KV heads → {n_kv_dst} KV heads "
          f"(group_size={n_q // n_kv_dst})")

    # ── load into GQA model ───────────────────────────────────────────────
    gqa_cfg = AttentionConfig(d_model=d_model, n_q_heads=n_q,
                              n_kv_heads=n_kv_dst, d_head=d_head)
    gqa = MultiHeadAttention(gqa_cfg)
    gqa.Wq.weight.data.copy_(gqa_state["Wq.weight"])
    gqa.Wk.weight.data.copy_(gqa_state["Wk.weight"])
    gqa.Wv.weight.data.copy_(gqa_state["Wv.weight"])
    gqa.Wo.weight.data.copy_(gqa_state["Wo.weight"])
    print("✓ Weights loaded into GQA model successfully\n")

    # ── sanity: same input should produce similar (not identical) output ──
    x = torch.randn(1, 16, d_model)
    with torch.no_grad():
        out_mha, _ = mha(x)
        out_gqa, _ = gqa(x)
    print(f"MHA output norm : {out_mha.norm():.4f}")
    print(f"GQA output norm : {out_gqa.norm():.4f}")
    print("  (outputs differ — GQA KV weights were averaged, so norms "
          "are similar but values diverge)\n")


# ════════════════════════════════════════════════════════════════════════════
# PART 6 — GQA vs MHA Inference Latency at Long Contexts
# ════════════════════════════════════════════════════════════════════════════

def profile_latency_comparison() -> None:
    print("=" * 60)
    print("PART 6 — GQA vs MHA Inference Latency at Long Contexts")
    print("=" * 60)

    d_model, d_head, n_q = 512, 64, 8
    context_lengths = [128, 256, 512, 1024, 2048, 4096]
    B = 1
    reps = 50

    variants = {"MHA": 8, "GQA(2kv)": 2, "MQA": 1}
    results = {name: [] for name in variants}

    for T_ctx in context_lengths:
        for name, n_kv in variants.items():
            cfg = AttentionConfig(d_model=d_model, n_q_heads=n_q,
                                  n_kv_heads=n_kv, d_head=d_head)
            model = MultiHeadAttention(cfg).to(DEVICE).eval()

            ctx = torch.randn(B, T_ctx, d_model, device=DEVICE)
            tok = torch.randn(B, 1,     d_model, device=DEVICE)

            with torch.no_grad():
                _, cache = model(ctx)
                _ = model(tok, kv_cache=cache)   # warm-up

            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(reps):
                    model(tok, kv_cache=cache)
            elapsed_ms = (time.perf_counter() - start) / reps * 1e3
            results[name].append(elapsed_ms)

    # ── print table ───────────────────────────────────────────────────────
    header = f"{'Context':>9}" + "".join(f"{n:>14}" for n in variants)
    print(f"\n{'Context':>9}" + "".join(f"{n:>14}" for n in variants) + "  [decode latency ms]")
    print("-" * (9 + 14 * len(variants)))

    mha_times = results["MHA"]
    for i, T in enumerate(context_lengths):
        row = f"{T:>9}"
        for j, (name, _) in enumerate(variants.items()):
            ms = results[name][i]
            row += f"{ms:>14.3f}"
        print(row)

    print("\n── Speedup of GQA & MQA over MHA ──")
    print(f"{'Context':>9}" + "".join(
        f"{n + ' speedup':>18}" for n in ("GQA(2kv)", "MQA")
    ))
    print("-" * (9 + 18 * 2))
    for i, T in enumerate(context_lengths):
        mha_t = results["MHA"][i]
        gqa_t = results["GQA(2kv)"][i]
        mqa_t = results["MQA"][i]
        print(f"{T:>9} {mha_t/gqa_t:>18.2f}x {mha_t/mqa_t:>18.2f}x")
    print()

    # ── KV memory savings ────────────────────────────────────────────────
    print("── KV Cache Memory Savings at max context (4096) ──")
    T_peak = 4096
    for name, n_kv in variants.items():
        mb = kv_cache_memory_bytes(n_kv, d_head, T_peak) / 1e6
        ratio = kv_cache_memory_bytes(8, d_head, T_peak) / \
                kv_cache_memory_bytes(n_kv, d_head, T_peak)
        print(f"  {name:>12}: {mb:.3f} MB  ({ratio:.1f}x smaller than MHA)")
    print()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    verify_no_future_leakage(seq_len=8)
    benchmark_flops_vs_seqlen()
    benchmark_variants()
    demo_checkpoint_conversion()
    profile_latency_comparison()
    print("All parts completed successfully.")