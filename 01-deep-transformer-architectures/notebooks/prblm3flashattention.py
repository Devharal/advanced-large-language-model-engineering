"""
Problem 3: Verify mathematical equivalence of FlashAttention vs naive attention

What this file teaches:
  - Why FlashAttention should produce EXACTLY the same output as naive attention
  - Why in practice there are tiny numerical differences (floating point order)
  - How to correctly measure "are these the same?" — not ==, but tolerances
  - What absolute error, relative error, and max error mean and which to use
  - How to stress-test equivalence across edge cases (long seqs, extreme values)
  - How to verify the online softmax reordering doesn't break the math

The core claim being verified:
  FlashAttention(Q, K, V) ≈ softmax(Q × Kᵀ / √d) × V
  to within floating point rounding error (~1e-3 for FP16).

Requirements:
  pip install torch
  pip install flash-attn --no-build-isolation  (for FA-2 verification)
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Why FlashAttention outputs are not EXACTLY identical
#
# FlashAttention and naive attention compute the same mathematical function.
# But floating point arithmetic is not mathematically associative:
#   (a + b) + c  ≠  a + (b + c)   in floating point
#
# Naive attention order:
#   1. Compute full Q × Kᵀ in one shot → accumulates in a specific order
#   2. Softmax over complete row
#   3. P × V in one shot
#
# FlashAttention order:
#   1. Compute Q × Kᵀ tile by tile, updating running (max, sum) after each tile
#   2. Online softmax rescaling at each tile boundary
#   3. Accumulate P × V tile by tile
#
# Different summation order → different rounding errors → tiny differences.
# These differences are O(ε_machine × condition_number) where:
#   ε_machine ≈ 1e-7 (FP32) or 1e-3 (FP16)
#
# What we expect:
#   FP32: max absolute difference < 1e-5
#   FP16: max absolute difference < 1e-2  (FP16 has only ~3 decimal digits)
#   BF16: max absolute difference < 1e-2  (same precision range)
#
# These differences are NOT bugs. They are the expected consequence of
# finite precision arithmetic with different summation orders.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EquivalenceResult:
    """
    Structured result from one equivalence test.
    Contains all error statistics and a pass/fail verdict.
    """
    test_name:         str
    max_abs_error:     float    # largest absolute difference |FA[i] - naive[i]|
    mean_abs_error:    float    # average absolute difference
    max_rel_error:     float    # largest relative difference |FA-naive|/|naive|
    rel_error_pct_99:  float    # 99th percentile of relative errors
    atol:              float    # tolerance used for pass/fail
    rtol:              float    # relative tolerance used for pass/fail
    passed:            bool     # True if torch.allclose(fa_out, naive_out, atol, rtol)
    dtype:             str
    shape:             tuple
    note:              str = ""  # optional note (e.g., causal mask on/off)

    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return (
            f"{status} | {self.test_name}\n"
            f"       dtype={self.dtype} shape={self.shape}\n"
            f"       max_abs={self.max_abs_error:.2e}  "
            f"mean_abs={self.mean_abs_error:.2e}  "
            f"max_rel={self.max_rel_error:.2e}  "
            f"p99_rel={self.rel_error_pct_99:.2e}\n"
            f"       tolerances: atol={self.atol:.2e}  rtol={self.rtol:.2e}"
            + (f"\n       note: {self.note}" if self.note else "")
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Reference implementations
#
# We compare THREE implementations:
#   1. naive_attention_fp32: naive attention computed in FP32 — our "ground truth"
#   2. naive_attention_dtype: naive attention in the test dtype (FP16/BF16)
#   3. flash_attention: PyTorch SDPA (FlashAttention when conditions are met)
#
# Comparing FA vs naive_fp32 tells us: "how close is FA to true math?"
# Comparing FA vs naive_dtype tells us: "are they equivalent in the same dtype?"
# ─────────────────────────────────────────────────────────────────────────────

def naive_attention_fp32(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                         causal: bool = False) -> torch.Tensor:
    """
    Naive attention computed in FP32, regardless of input dtype.
    Used as the "true" reference since FP32 has ~7 decimal digits of precision.

    Steps:
      1. Upcast Q, K, V to FP32
      2. Compute full N×N attention matrix
      3. Softmax over rows
      4. Weighted sum with V
      5. Return in original dtype

    Args:
        Q, K, V: [batch, heads, seq, head_dim]  (any dtype)
        causal:  apply causal mask
    Returns:
        output: [batch, heads, seq, head_dim] in same dtype as Q
    """
    orig_dtype = Q.dtype

    # Upcast for precision — this is our "true" answer
    Q = Q.float()
    K = K.float()
    V = V.float()

    d_head = Q.shape[-1]
    scale  = 1.0 / math.sqrt(d_head)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale   # [B, H, N, N]

    if causal:
        N = Q.shape[-2]
        mask = torch.triu(
            torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float('-inf'))

    probs  = torch.softmax(scores, dim=-1)         # [B, H, N, N] in FP32
    output = torch.matmul(probs, V)                # [B, H, N, d] in FP32

    return output.to(orig_dtype)                   # back to original dtype


def naive_attention_native(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                            causal: bool = False) -> torch.Tensor:
    """
    Naive attention in the native dtype of the inputs.
    In FP16: accumulates errors due to limited precision.
    Used to show that FA's errors are comparable to naive's errors vs FP32 truth.

    Args:
        Q, K, V: [batch, heads, seq, head_dim]
        causal:  apply causal mask
    Returns:
        output: [batch, heads, seq, head_dim]
    """
    d_head = Q.shape[-1]
    scale  = 1.0 / math.sqrt(d_head)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale   # in native dtype

    if causal:
        N = Q.shape[-2]
        mask = torch.triu(
            torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float('-inf'))

    probs  = torch.softmax(scores.float(), dim=-1).to(Q.dtype)  # softmax in FP32 → cast back
    output = torch.matmul(probs, V)

    return output


def flash_attention_sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                          causal: bool = False) -> torch.Tensor:
    """
    FlashAttention via PyTorch's scaled_dot_product_attention.

    PyTorch dispatches to the FlashAttention backend when:
      - dtype in {float16, bfloat16}
      - device is CUDA
      - no custom mask (or is_causal=True)
      - dropout_p = 0

    When conditions aren't met, falls back to a memory-efficient implementation
    or the math fallback. We check which backend was selected below.

    Args:
        Q, K, V: [batch, heads, seq, head_dim]
        causal:  use causal attention mask
    Returns:
        output: [batch, heads, seq, head_dim]
    """
    return F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Core comparison function
#
# How to correctly compare two tensors that "should be the same":
#
#   BAD:  output_fa == output_naive         (exact equality — almost never true)
#
#   GOOD: torch.allclose(a, b, atol, rtol)
#         Returns True if |a - b| ≤ atol + rtol × |b|  for all elements.
#
#         atol: absolute tolerance — handles small values near zero
#         rtol: relative tolerance — handles large values proportionally
#
# Choosing tolerances:
#   FP32: atol=1e-5, rtol=1e-5   (FP32 has ~7 sig figs, allows 1-2 rounding errors)
#   FP16: atol=1e-3, rtol=1e-3   (FP16 has ~3 sig figs, allows proportionally more)
#   BF16: atol=1e-2, rtol=1e-2   (BF16 has same range as FP32 but ~2 sig figs)
# ─────────────────────────────────────────────────────────────────────────────

DTYPE_TOLERANCES = {
    torch.float32:  (1e-5, 1e-5),
    torch.float16:  (1e-3, 1e-3),
    torch.bfloat16: (2e-2, 2e-2),
}


def compare_outputs(
    output_ref:  torch.Tensor,
    output_test: torch.Tensor,
    test_name:   str,
    note:        str = "",
) -> EquivalenceResult:
    """
    Compute all error statistics between two attention output tensors.

    Args:
        output_ref:  reference output (e.g., naive FP32 or naive native)
        output_test: test output (e.g., FlashAttention)
        test_name:   label for the result
        note:        optional note

    Returns:
        EquivalenceResult with all statistics and pass/fail verdict
    """
    dtype = output_ref.dtype

    # Cast both to FP32 for error computation
    # (can't compute meaningful differences in FP16 itself due to rounding)
    ref_f32  = output_ref.float()
    test_f32 = output_test.float()

    diff = (test_f32 - ref_f32).abs()  # absolute differences

    max_abs  = diff.max().item()
    mean_abs = diff.mean().item()

    # Relative error: |diff| / (|ref| + ε)
    # ε prevents division by zero when ref is exactly 0
    rel_err  = diff / (ref_f32.abs() + 1e-8)
    max_rel  = rel_err.max().item()
    p99_rel  = rel_err.flatten().quantile(0.99).item()  # 99th percentile

    atol, rtol = DTYPE_TOLERANCES.get(dtype, (1e-3, 1e-3))
    passed = torch.allclose(test_f32, ref_f32, atol=atol, rtol=rtol)

    return EquivalenceResult(
        test_name=test_name,
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        max_rel_error=max_rel,
        rel_error_pct_99=p99_rel,
        atol=atol,
        rtol=rtol,
        passed=passed,
        dtype=str(dtype).replace("torch.", ""),
        shape=tuple(output_ref.shape),
        note=note,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Test suite
#
# We verify equivalence across:
#   1. Different dtypes (FP32, FP16, BF16)
#   2. Different sequence lengths (where FA-1 vs FA-2 tiling varies)
#   3. With and without causal mask
#   4. Extreme input values (large scores → hard softmax; small scores)
#   5. Batch size > 1 and multiple heads
#   6. GQA configuration (fewer KV heads than Q heads)
# ─────────────────────────────────────────────────────────────────────────────

def test_basic_equivalence(device: str = "cuda") -> list:
    """
    Test 1: Basic equivalence across dtypes and sequence lengths.
    """
    results = []

    configs = [
        # (dtype, batch, heads, seq, d_head, causal, note)
        (torch.float32, 1, 8,  64,  64,  False, "small, non-causal, FP32"),
        (torch.float16, 1, 8,  64,  64,  False, "small, non-causal, FP16"),
        (torch.bfloat16,1, 8,  64,  64,  False, "small, non-causal, BF16"),
        (torch.float16, 1, 8,  64,  64,  True,  "small, causal, FP16"),
        (torch.float16, 1, 8,  128, 64,  False, "medium seq, FP16"),
        (torch.float16, 1, 8,  256, 64,  False, "longer seq, FP16"),
        (torch.float16, 1, 8,  512, 64,  True,  "seq=512, causal, FP16"),
        (torch.float16, 1, 8,  1024,64,  True,  "seq=1024, causal, FP16"),
        (torch.float16, 2, 8,  256, 64,  False, "batch=2, FP16"),
        (torch.float16, 1, 32, 256, 128, True,  "32 heads, d=128, FP16"),
    ]

    print("\n" + "=" * 70)
    print("Test 1: Basic Equivalence (FlashAttention vs Naive FP32 Reference)")
    print("=" * 70)

    for dtype, batch, heads, seq, d_head, causal, note in configs:
        torch.manual_seed(42)   # reproducibility

        Q = torch.randn(batch, heads, seq, d_head, device=device, dtype=dtype)
        K = torch.randn(batch, heads, seq, d_head, device=device, dtype=dtype)
        V = torch.randn(batch, heads, seq, d_head, device=device, dtype=dtype)

        with torch.no_grad():
            ref_out = naive_attention_fp32(Q, K, V, causal=causal)
            fa_out  = flash_attention_sdpa(Q, K, V, causal=causal)

        result = compare_outputs(
            ref_out, fa_out,
            test_name="FA-SDPA vs Naive-FP32",
            note=note,
        )
        results.append(result)
        print(result)

    return results


def test_online_softmax_equivalence(device: str = "cuda") -> list:
    """
    Test 2: Verify online softmax produces the same result as standard softmax.

    This is the mathematical core of FlashAttention. We implement a simplified
    tiled attention to show the online softmax update is exact.
    """
    results = []

    print("\n" + "=" * 70)
    print("Test 2: Online Softmax Equivalence")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # Implement a Python-level tiled attention that uses online softmax.
    # This is NOT fast (no CUDA kernel) but proves the math is correct.
    # It's the algorithmic skeleton that FA implements in CUDA.
    # ─────────────────────────────────────────────────────────────────────────

    def online_softmax_attention(Q: torch.Tensor, K: torch.Tensor,
                                 V: torch.Tensor, tile_size: int = 32) -> torch.Tensor:
        """
        Tiled attention with online softmax.
        Processes K, V in tiles of `tile_size` while maintaining running
        (max, sum) statistics for exact softmax normalization.

        This is conceptually what FlashAttention does (but in Python for clarity).

        Returns exactly the same result as naive_attention_fp32 (modulo float ordering).
        """
        Q = Q.float()   # work in FP32 for clarity
        K = K.float()
        V = V.float()

        B, H, N, d = Q.shape
        scale = 1.0 / math.sqrt(d)

        # Output accumulator — will be divided by l at the end
        O = torch.zeros(B, H, N, d, device=Q.device)

        # Running statistics (per query position)
        m = torch.full((B, H, N), float('-inf'), device=Q.device)  # running max
        l = torch.zeros(B, H, N, device=Q.device)                  # running sum

        # Process K, V in tiles
        for j in range(0, N, tile_size):
            K_j = K[:, :, j : j + tile_size, :]   # [B, H, tile, d]
            V_j = V[:, :, j : j + tile_size, :]   # [B, H, tile, d]

            # Score tile: [B, H, N, tile]
            S_j = torch.matmul(Q, K_j.transpose(-2, -1)) * scale

            # Online softmax update
            # m_j: max of THIS tile's scores, shape [B, H, N]
            m_j   = S_j.max(dim=-1).values

            # New global max after seeing this tile
            m_new = torch.maximum(m, m_j)    # elementwise max

            # Rescale old running sum to new reference maximum
            # exp(m_old - m_new) adjusts for the change in reference point
            # When m_new == m_old: exp(0) = 1 → no change
            # When m_new  > m_old: exp(-positive) < 1 → old sum shrinks
            l_new = torch.exp(m - m_new) * l \
                  + torch.exp(S_j - m_new.unsqueeze(-1)).sum(dim=-1)

            # Update output accumulator
            # Old O contribution: rescaled to new reference max
            # New O contribution: weighted sum of V_j using this tile's attention
            O_new = torch.exp(m - m_new).unsqueeze(-1) * O \
                  + torch.matmul(
                      torch.exp(S_j - m_new.unsqueeze(-1)),   # [B, H, N, tile]
                      V_j                                       # [B, H, tile, d]
                  )                                            # → [B, H, N, d]

            # Update state
            m = m_new
            l = l_new
            O = O_new

        # Final normalization: divide by cumulative sum to get true softmax weights
        # O currently contains: Σ_j exp(S_ij - m) × V_j
        # l currently contains: Σ_j exp(S_ij - m)
        # Dividing gives: Σ_j (exp(S_ij - m) / l) × V_j = Σ_j softmax(S_i)_j × V_j
        output = O / l.unsqueeze(-1)

        return output

    # Compare online softmax attention vs standard attention
    seq_lengths  = [64, 128, 256, 512]
    tile_sizes   = [16, 32, 64]   # different tile sizes should give same result

    for seq in seq_lengths:
        torch.manual_seed(0)
        Q = torch.randn(1, 4, seq, 32, device=device)
        K = torch.randn(1, 4, seq, 32, device=device)
        V = torch.randn(1, 4, seq, 32, device=device)

        ref = naive_attention_fp32(Q, K, V, causal=False)

        for tile in tile_sizes:
            # Skip if tile size >= seq (no tiling needed)
            if tile >= seq:
                continue

            tiled_out = online_softmax_attention(Q, K, V, tile_size=tile)

            # Compare in FP32 — should be very close (FP32 rounding only)
            dtype = torch.float32
            atol = DTYPE_TOLERANCES[dtype][0]
            rtol = DTYPE_TOLERANCES[dtype][1]
            passed = torch.allclose(ref.float(), tiled_out.float(), atol=atol, rtol=rtol)

            diff = (ref.float() - tiled_out.float()).abs()
            result = EquivalenceResult(
                test_name=f"OnlineSoftmax tile={tile}",
                max_abs_error=diff.max().item(),
                mean_abs_error=diff.mean().item(),
                max_rel_error=(diff / (ref.float().abs() + 1e-8)).max().item(),
                rel_error_pct_99=(diff / (ref.float().abs() + 1e-8)).flatten().quantile(0.99).item(),
                atol=atol, rtol=rtol,
                passed=passed,
                dtype="float32",
                shape=tuple(Q.shape),
                note=f"seq={seq}, tile={tile}",
            )
            results.append(result)
            print(result)

    return results


def test_extreme_inputs(device: str = "cuda") -> list:
    """
    Test 3: Equivalence under numerically challenging inputs.

    Extreme inputs stress-test numerical stability:
      - Very large scores → softmax near one-hot
      - Very small scores → softmax near uniform
      - All-zero scores → uniform attention (should be exactly 1/N)
      - Mix of large positive and large negative scores
      - Inf/nan guard: causal mask produces -inf in scores
    """
    results = []

    print("\n" + "=" * 70)
    print("Test 3: Extreme Inputs — Numerical Stability")
    print("=" * 70)

    B, H, N, d = 1, 4, 64, 32
    dtype = torch.float16

    extreme_cases = []

    # Case 1: Large positive scores → sharply peaked softmax
    # Scores very large → one token gets ~1.0, rest get ~0
    Q_large = torch.randn(B, H, N, d, device=device, dtype=dtype) * 10.0
    K_large = torch.randn(B, H, N, d, device=device, dtype=dtype) * 10.0
    V_large = torch.randn(B, H, N, d, device=device, dtype=dtype)
    extreme_cases.append((Q_large, K_large, V_large, False, "large scores (×10)"))

    # Case 2: Very small scores → near-uniform softmax
    Q_small = torch.randn(B, H, N, d, device=device, dtype=dtype) * 0.001
    K_small = torch.randn(B, H, N, d, device=device, dtype=dtype) * 0.001
    V_small = torch.randn(B, H, N, d, device=device, dtype=dtype)
    extreme_cases.append((Q_small, K_small, V_small, False, "small scores (×0.001)"))

    # Case 3: All-zero Q — should produce uniform attention, output = mean(V)
    Q_zero = torch.zeros(B, H, N, d, device=device, dtype=dtype)
    K_zero = torch.randn(B, H, N, d, device=device, dtype=dtype)
    V_zero = torch.randn(B, H, N, d, device=device, dtype=dtype)
    extreme_cases.append((Q_zero, K_zero, V_zero, False, "Q=zeros (uniform attention)"))

    # Case 4: Causal mask at different sequence lengths
    # Tests that masking interacts correctly with online softmax
    Q_causal = torch.randn(B, H, N, d, device=device, dtype=dtype)
    K_causal = torch.randn(B, H, N, d, device=device, dtype=dtype)
    V_causal = torch.randn(B, H, N, d, device=device, dtype=dtype)
    extreme_cases.append((Q_causal, K_causal, V_causal, True, "causal mask, random inputs"))

    # Case 5: Very asymmetric V values — tests that weighted sum is correct
    V_asym = torch.zeros(B, H, N, d, device=device, dtype=dtype)
    V_asym[:, :, 0, :] = 1000.0    # one very large value
    V_asym[:, :, 1:, :] = 0.001    # rest near zero
    Q_asym = torch.randn(B, H, N, d, device=device, dtype=dtype)
    K_asym = torch.randn(B, H, N, d, device=device, dtype=dtype)
    extreme_cases.append((Q_asym, K_asym, V_asym, False, "asymmetric V (one outlier)"))

    for Q, K, V, causal, note in extreme_cases:
        with torch.no_grad():
            ref_out = naive_attention_fp32(Q, K, V, causal=causal)
            fa_out  = flash_attention_sdpa(Q, K, V, causal=causal)

            # Also check: verify all-zero Q gives mean(V)
            if "uniform" in note:
                expected_uniform = V.float().mean(dim=2, keepdim=True).expand_as(V.float())
                deviation = (ref_out.float() - expected_uniform).abs().max().item()
                print(f"  [Sanity] All-zero Q → output should be mean(V). Deviation: {deviation:.2e}")

        result = compare_outputs(ref_out, fa_out, test_name="FA vs Naive", note=note)
        results.append(result)
        print(result)

    return results


def test_gqa_equivalence(device: str = "cuda") -> list:
    """
    Test 4: GQA (Grouped Query Attention) equivalence.

    In GQA, Q has H_q heads but K, V only have H_kv heads (H_kv < H_q).
    PyTorch SDPA handles GQA by broadcasting K, V across query groups.

    We verify the GQA output matches the naive implementation
    (which explicitly expands K, V before computing attention).
    """
    results = []

    print("\n" + "=" * 70)
    print("Test 4: GQA Equivalence (Fewer KV Heads than Q Heads)")
    print("=" * 70)

    def naive_gqa_attention(Q: torch.Tensor, K: torch.Tensor,
                             V: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """
        Naive GQA via explicit expansion of K, V.
        K, V are repeated to match the number of Q heads.

        Args:
            Q: [B, H_q, N, d]
            K: [B, H_kv, N, d]    H_kv < H_q, H_q % H_kv == 0
            V: [B, H_kv, N, d]
        Returns:
            output: [B, H_q, N, d]
        """
        H_q  = Q.shape[1]
        H_kv = K.shape[1]
        assert H_q % H_kv == 0, "H_q must be divisible by H_kv for GQA"
        groups = H_q // H_kv

        # Expand K, V: each KV head serves `groups` query heads
        # [B, H_kv, N, d] → [B, H_kv, 1, N, d] → [B, H_kv, groups, N, d] → [B, H_q, N, d]
        K_expanded = K.unsqueeze(2).expand(-1, -1, groups, -1, -1).reshape(
            K.shape[0], H_q, K.shape[2], K.shape[3]
        )
        V_expanded = V.unsqueeze(2).expand(-1, -1, groups, -1, -1).reshape(
            V.shape[0], H_q, V.shape[2], V.shape[3]
        )

        return naive_attention_fp32(Q, K_expanded, V_expanded, causal=causal)

    gqa_configs = [
        # (B, H_q, H_kv, N, d, causal, note)
        (1, 32, 8,  256, 128, True,  "32 Q heads, 8 KV heads (GQA G=4)"),
        (1, 32, 1,  256, 128, True,  "32 Q heads, 1 KV head  (MQA)"),
        (1, 16, 4,  512, 64,  True,  "16 Q heads, 4 KV heads (GQA G=4)"),
        (2, 32, 8,  128, 128, False, "batch=2, 32/8 heads, non-causal"),
    ]

    for B, H_q, H_kv, N, d, causal, note in gqa_configs:
        torch.manual_seed(1)
        dtype = torch.bfloat16

        Q = torch.randn(B, H_q,  N, d, device=device, dtype=dtype)
        K = torch.randn(B, H_kv, N, d, device=device, dtype=dtype)
        V = torch.randn(B, H_kv, N, d, device=device, dtype=dtype)

        with torch.no_grad():
            ref_out = naive_gqa_attention(Q, K, V, causal=causal)
            fa_out  = flash_attention_sdpa(Q, K, V, causal=causal)

        result = compare_outputs(ref_out, fa_out,
                                  test_name="GQA: FA-SDPA vs Naive-expand",
                                  note=note)
        results.append(result)
        print(result)

    return results


def test_gradient_equivalence(device: str = "cuda") -> list:
    """
    Test 5: Backward pass (gradient) equivalence.

    During training, FlashAttention recomputes the attention matrix
    during the backward pass instead of storing it. This should produce
    equivalent gradients. We verify dL/dQ, dL/dK, dL/dV are equivalent.
    """
    results = []

    print("\n" + "=" * 70)
    print("Test 5: Gradient Equivalence (Backward Pass)")
    print("=" * 70)

    # Gradients require float32 for the backward pass itself to be stable.
    # We test in float32 (exact) and float16 (approximate).
    configs = [
        (torch.float32, 1, 4, 64,  32, "FP32"),
        (torch.float16, 1, 4, 128, 64, "FP16"),
    ]

    for dtype, B, H, N, d, dtype_name in configs:
        torch.manual_seed(42)

        # requires_grad=True so backward pass works
        Q_ref = torch.randn(B, H, N, d, device=device, dtype=dtype, requires_grad=True)
        K_ref = torch.randn(B, H, N, d, device=device, dtype=dtype, requires_grad=True)
        V_ref = torch.randn(B, H, N, d, device=device, dtype=dtype, requires_grad=True)

        # Clone for FA (same values, separate computation graph)
        Q_fa  = Q_ref.detach().clone().requires_grad_(True)
        K_fa  = K_ref.detach().clone().requires_grad_(True)
        V_fa  = V_ref.detach().clone().requires_grad_(True)

        # Dummy upstream gradient (simulates gradient from the next layer)
        grad_out = torch.randn(B, H, N, d, device=device, dtype=dtype)

        # Forward + backward for naive attention
        out_ref = naive_attention_fp32(Q_ref, K_ref, V_ref)
        out_ref.backward(grad_out)

        # Forward + backward for FlashAttention
        out_fa = flash_attention_sdpa(Q_fa, K_fa, V_fa)
        out_fa.backward(grad_out)

        # Compare gradients for each input
        for tensor_name, grad_ref, grad_fa in [
            ("dL/dQ", Q_ref.grad, Q_fa.grad),
            ("dL/dK", K_ref.grad, K_fa.grad),
            ("dL/dV", V_ref.grad, V_fa.grad),
        ]:
            if grad_ref is None or grad_fa is None:
                print(f"  ⚠ Gradient for {tensor_name} is None — check requires_grad")
                continue

            result = compare_outputs(
                grad_ref, grad_fa,
                test_name=f"Gradient {tensor_name}",
                note=f"dtype={dtype_name}, N={N}"
            )
            results.append(result)
            print(result)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Summary and diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: list):
    """Print a final summary table of all test results."""
    passed = [r for r in all_results if r.passed]
    failed = [r for r in all_results if not r.passed]

    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(passed)}/{len(all_results)} tests passed")
    print("=" * 70)

    if failed:
        print(f"\n⚠ FAILED TESTS ({len(failed)}):")
        for r in failed:
            print(f"  ✗ {r.test_name} | {r.note}")
            print(f"    max_abs={r.max_abs_error:.2e}  tolerance={r.atol:.2e}")
    else:
        print("\n✓ All tests passed.")

    # Print error magnitude summary
    print("\nError magnitude summary (across all passing tests):")
    print(f"  {'Test':<40} {'max_abs':>10} {'max_rel':>10}")
    print(f"  {'─'*40} {'─'*10} {'─'*10}")
    for r in all_results:
        status = "✓" if r.passed else "✗"
        short_name = f"{r.test_name[:28]}... {r.dtype}" if len(r.test_name) > 30 else r.test_name
        print(f"  {status} {short_name:<40} {r.max_abs_error:>10.2e} {r.max_rel_error:>10.2e}")

    # Educational note on error magnitudes
    print("""
What the errors mean:
  FP32 max_abs ~ 1e-6 to 1e-5  →  Normal. One floating point rounding error.
  FP16 max_abs ~ 1e-3 to 1e-2  →  Normal. FP16 has ~3 decimal digits of precision.
  BF16 max_abs ~ 1e-2 to 5e-2  →  Normal. BF16 has ~2 decimal digits of precision.

  FA and naive compute the SAME mathematical function.
  The differences you see are purely from floating point summation order.
  They are NOT bugs in FlashAttention.
  They are NOT approximation errors.
  They are the same type of error you'd see between:
    sum([1, 2, 3, 4])  vs  sum([4, 3, 2, 1])   in floating point.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Backend detection utility
#
# Useful debugging tool: which attention backend did PyTorch actually use?
# SDPA can dispatch to flash_attention, mem_efficient, or math backends.
# ─────────────────────────────────────────────────────────────────────────────

def check_active_backend(dtype=torch.float16, device="cuda"):
    """
    Report which attention backend PyTorch SDPA will use for given dtype/device.

    PyTorch 2.0+ chooses backends in this priority order:
      1. FlashAttention  (requires: FP16/BF16, CUDA, no dropout, no custom mask)
      2. Memory-Efficient (xFormers-style, broader compatibility)
      3. Math            (naive reference, CPU or unsupported GPU)

    You can force a backend with:
      with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
          output = F.scaled_dot_product_attention(q, k, v)
    """
    print("\n" + "=" * 70)
    print("Active SDPA Backend Detection")
    print("=" * 70)

    Q = torch.randn(1, 8, 64, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, 64, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, 64, 64, device=device, dtype=dtype)

    backends = {
        "FlashAttention": dict(enable_flash=True,  enable_mem_efficient=False, enable_math=False),
        "MemEfficient":   dict(enable_flash=False, enable_mem_efficient=True,  enable_math=False),
        "Math (Naive)":   dict(enable_flash=False, enable_mem_efficient=False, enable_math=True),
    }

    for name, flags in backends.items():
        try:
            with torch.backends.cuda.sdp_kernel(**flags):
                out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
                status = "✓ Available"
        except RuntimeError as e:
            status = f"✗ Not available: {str(e)[:60]}"
        print(f"  {name:<20} {status}")

    print(f"\n  (Testing with dtype={dtype}, device={device})")
    print(f"  Note: FlashAttention requires FP16 or BF16 dtype on CUDA.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    assert torch.cuda.is_available(), "CUDA GPU required for FlashAttention"
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:   {torch.version.cuda}")

    device = "cuda"

    # Detect which backends are available
    check_active_backend(dtype=torch.float16, device=device)

    all_results = []

    # Run all test suites
    all_results += test_basic_equivalence(device)
    all_results += test_online_softmax_equivalence(device)
    all_results += test_extreme_inputs(device)
    all_results += test_gqa_equivalence(device)
    all_results += test_gradient_equivalence(device)

    # Final summary
    print_summary(all_results)


if __name__ == "__main__":
    main()