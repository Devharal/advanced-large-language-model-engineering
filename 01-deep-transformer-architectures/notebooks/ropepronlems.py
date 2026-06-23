"""
pe_lab.py  ─  Positional Encoding Laboratory
═══════════════════════════════════════════════════════════════════════════════
Problem 1 │ RoPE from scratch – rotate Q, K in complex space before attention
Problem 2 │ YaRN – NTK-by-parts ramp + temperature correction,
            tested on sequences 2× the training length
Problem 3 │ Position-generalisation comparison:
            Vanilla RoPE  ·  ALiBi  ·  Learned APE
═══════════════════════════════════════════════════════════════════════════════
Run:
    python pe_lab.py

Outputs (saved to working directory):
    p1_rope_verification.png
    p2_yarn_analysis.png
    p3_generalisation.png
"""

# ─────────────────── Standard imports ───────────────────────────────────────
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEP = "═" * 72
print(f"[Lab]  device = {DEVICE}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SHARED ROPE PRIMITIVES  (used by all three problems)                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    base: float = 10_000.0,
    device: torch.device = torch.device("cpu"),
    custom_inv_freq: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute cos/sin rotation tables for RoPE.

    Theory recap
    ─────────────
    Chop head_dim into d/2 pairs.  Pair i gets frequency θ_i:
        θ_i = base^{-2i/d}          (geometric, from fast to slow)
    For position m, pair i rotates by angle  m·θ_i.

    Returned tables use the 'split-half' pairing convention
    (first d/2 cols handle x[0..d/2-1], last d/2 handle x[d/2..d-1])
    which vectorises better than the interleaved (adjacent) convention.

    Parameters
    ──────────
    head_dim        : per-head dimensionality (must be even)
    max_seq_len     : maximum number of positions to pre-compute
    base            : RoPE base frequency  (default 10 000)
    device          : target device
    custom_inv_freq : (d/2,) tensor — supply pre-computed frequencies
                      (used by YaRN / NTK-aware to override the defaults)

    Returns
    ───────
    cos_table : (max_seq_len, head_dim)
    sin_table : (max_seq_len, head_dim)
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    if custom_inv_freq is not None:
        inv_freq = custom_inv_freq.to(device)               # (d/2,)
    else:
        # θ_i = base^{-2i/d}   for i = 0, 1, …, d/2-1
        i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (i / head_dim))           # (d/2,)

    # m·θ_i  for all positions m and all frequencies θ_i
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)                 # (S, d/2)

    # Duplicate so shape becomes (S, d) — split-half convention
    freqs = torch.cat([freqs, freqs], dim=-1)                # (S, d)

    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    The 'rotate-half' permutation used to implement 2×2 rotation efficiently.

    For each complex pair (x_i, x_{i+d/2}) this computes:
        (-x_{i+d/2},  x_i)
    which is exactly the (-sin·x₁ + cos·x₂, sin·x₁ + cos·x₂) part
    when combined with the x*cos term in apply_rotary_pos_emb.

    Input  : (..., head_dim)
    Output : (..., head_dim)
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]    # first half  (real component of each pair)
    x2 = x[..., d // 2 :]    # second half (imaginary component)
    # Rotation: new_real = -x2, new_imag = x1
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    q, k       : (batch, n_heads, seq_len, head_dim)
    cos, sin   : (max_seq_len, head_dim)  — precomputed tables
    position_ids : (seq_len,) int — which rows of cos/sin to use
                   (allows non-contiguous positions, e.g. during KV-cache decode)

    RoPE formula per token at position m:
        x_rotated = x * cos(m·θ) + rotate_half(x) * sin(m·θ)

    This is exactly equivalent to multiplying the complex number
    (x_{2i} + j·x_{2i+1})  by  e^{j·m·θ_i}  for each pair i.
    """
    S = q.shape[2]
    if position_ids is None:
        position_ids = torch.arange(S, device=q.device)

    # Index the precomputed table by position_ids: (S, d)
    cos_pos = cos[position_ids]  # (S, head_dim)
    sin_pos = sin[position_ids]  # (S, head_dim)

    # Broadcast to (1, 1, S, head_dim) for (batch, heads, seq, dim)
    cos_pos = cos_pos.unsqueeze(0).unsqueeze(0)
    sin_pos = sin_pos.unsqueeze(0).unsqueeze(0)

    # Apply rotation to Q and K  (V is left untouched — position only
    # needs to influence *where* attention looks, not *what* is aggregated)
    q_rot = q * cos_pos + rotate_half(q) * sin_pos
    k_rot = k * cos_pos + rotate_half(k) * sin_pos

    return q_rot, k_rot


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PROBLEM 1 — RoPE Attention Module & Verification                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class RoPEMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with Rotary Position Embeddings.

    Key design points
    ──────────────────
    • RoPE is applied AFTER the Q and K linear projections, not to the token
      embedding — so it acts fresh at every layer/head independently.
    • V is never rotated.
    • The cos/sin tables are registered as non-parameter buffers: they move
      with the module (.to(device)) but are not trained.
    • Causal (autoregressive) masking is applied before softmax.

    Arguments
    ─────────
    d_model       : total model dimension
    n_heads       : number of attention heads
    max_seq_len   : maximum sequence length to pre-compute tables for
    base          : RoPE frequency base  (default 10 000)
    custom_freqs  : (d/2,) tensor — override inv_freq (for YaRN etc.)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 512,
        base: float = 10_000.0,
        custom_freqs: Optional[torch.Tensor] = None,
        attn_scale_correction: float = 1.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads   = n_heads
        self.head_dim  = d_model // n_heads
        self.d_model   = d_model
        # 1/√d  (the standard scaling; YaRN multiplies this by mscale)
        self.scale     = (self.head_dim ** -0.5) * attn_scale_correction

        # Learned projection matrices (no bias — standard in modern LLMs)
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Pre-compute rotation tables and register as buffers
        cos, sin = precompute_rope_freqs(
            self.head_dim, max_seq_len, base,
            custom_inv_freq=custom_freqs,
        )
        self.register_buffer("cos_cached", cos)  # (max_seq_len, head_dim)
        self.register_buffer("sin_cached", sin)

    # ── forward ─────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x : (batch, seq_len, d_model)
        Returns: (output, attn_weights or None)
        """
        B, S, D = x.shape

        # ── 1. Project to Q, K, V ──────────────────────────────────────────
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v : (B, H, S, head_dim)

        # ── 2. Apply RoPE to Q and K (NOT to V) ───────────────────────────
        q, k = apply_rotary_pos_emb(
            q, k,
            self.cos_cached,
            self.sin_cached,
            position_ids,
        )

        # ── 3. Scaled dot-product attention ───────────────────────────────
        # scores_{b,h,i,j} = (q_i · k_j) / √d
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,S,S)

        # Causal mask: token i can only attend to tokens j ≤ i
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)          # (B, H, S, S)

        # ── 4. Aggregate values ────────────────────────────────────────────
        out = torch.matmul(attn_weights, v)               # (B, H, S, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)                          # (B, S, D)

        if return_attn_weights:
            return out, attn_weights
        return out, None


# ─── Verification helpers ────────────────────────────────────────────────────

def p1_verify_relative_position_property(head_dim: int = 8, base: float = 10_000.0):
    """
    Prove that <rotate(q, m), rotate(k, n)> depends only on (m - n),
    not on m or n individually.

    We check: dot(q_m, k_n) == dot(q_{m+Δ}, k_{n+Δ})  for all Δ.
    """
    print("\n[P1] Verifying relative-position property of RoPE dot product …")
    cos, sin = precompute_rope_freqs(head_dim, max_seq_len=200, base=base)

    torch.manual_seed(0)
    q_raw = torch.randn(head_dim)
    k_raw = torch.randn(head_dim)

    # Make them (1,1,1,d) for apply_rotary_pos_emb
    q_4d = q_raw.view(1, 1, 1, head_dim)
    k_4d = k_raw.view(1, 1, 1, head_dim)

    results = []
    for m, n in [(5, 3), (50, 48), (100, 98), (10, 3), (20, 13), (75, 68)]:
        delta = m - n
        # Rotate q at position m, k at position n
        q_m, k_n = apply_rotary_pos_emb(
            q_4d, k_4d, cos, sin,
            position_ids=torch.tensor([m]),
        )
        dot_mn = (q_m * k_n).sum().item()

        # Shift both by +Δ — result should be identical (relative pos = m-n stays same)
        shift = 30
        q_ms, k_ns = apply_rotary_pos_emb(
            q_4d, k_4d, cos, sin,
            position_ids=torch.tensor([m + shift]),
        )
        # But now compare with k at n + shift → relative offset still m - n
        k_ns_key, _ = apply_rotary_pos_emb(
            k_4d, k_4d, cos, sin,
            position_ids=torch.tensor([n + shift]),
        )
        dot_shifted = (q_ms * k_ns_key).sum().item()

        error = abs(dot_mn - dot_shifted)
        results.append((m, n, delta, dot_mn, dot_shifted, error))
        print(f"   m={m:3d}, n={n:3d} | rel={delta:+3d} | "
              f"dot(q_m,k_n)={dot_mn:+.6f}  dot(q_{{m+Δ}},k_{{n+Δ}})={dot_shifted:+.6f}  "
              f"error={error:.2e}")

    max_error = max(r[5] for r in results)
    status = "✓ PASSED" if max_error < 1e-5 else "✗ FAILED"
    print(f"   {status}  (max |error| = {max_error:.2e})")
    return results


def p1_verify_norm_preservation(head_dim: int = 16):
    """
    RoPE is a rotation → it must preserve the L2 norm of Q and K vectors.
    Verify: ||q|| = ||rotate(q, m)|| for all m.
    """
    print("\n[P1] Verifying rotation preserves vector norms …")
    cos, sin = precompute_rope_freqs(head_dim, max_seq_len=1000)
    torch.manual_seed(7)
    q_raw = torch.randn(1, 1, 1, head_dim)

    norms_before, norms_after = [], []
    for m in [0, 1, 50, 100, 500, 999]:
        q_rot, _ = apply_rotary_pos_emb(q_raw, q_raw, cos, sin,
                                         position_ids=torch.tensor([m]))
        nb = q_raw.norm().item()
        na = q_rot.norm().item()
        norms_before.append(nb)
        norms_after.append(na)
        print(f"   m={m:4d} | ||q||={nb:.6f}  ||rotate(q,m)||={na:.6f}  "
              f"delta={abs(nb-na):.2e}")

    max_delta = max(abs(a - b) for a, b in zip(norms_before, norms_after))
    print(f"   {'✓ PASSED' if max_delta < 1e-5 else '✗ FAILED'}  "
          f"(max delta = {max_delta:.2e})")


def p1_compute_attention_decay(head_dim: int = 64, d_model: int = 64, n_heads: int = 1):
    """
    Compute the expected attention score magnitude as a function of
    relative distance |m - n|, averaged over many random (q, k) pairs.
    This demonstrates the long-term decay property.
    """
    attn = RoPEMultiHeadAttention(d_model=d_model, n_heads=n_heads,
                                   max_seq_len=512)
    attn.eval()
    cos, sin = attn.cos_cached, attn.sin_cached

    distances = list(range(0, 200, 5))
    mean_dots = []
    n_samples  = 500

    with torch.no_grad():
        for dist in distances:
            m, n = 100, 100 - dist if dist <= 100 else 0
            dots = []
            for _ in range(n_samples):
                q = torch.randn(1, 1, 1, head_dim)
                k = torch.randn(1, 1, 1, head_dim)
                qr, kr = apply_rotary_pos_emb(
                    q, k, cos, sin,
                    position_ids=torch.tensor([m]),
                )
                _, kr2 = apply_rotary_pos_emb(
                    q, k, cos, sin,
                    position_ids=torch.tensor([abs(m - dist)]),
                )
                dots.append((qr * kr2).sum().item())
            mean_dots.append(float(np.mean(np.abs(dots))))

    return distances, mean_dots


# ─── Problem 1 Figure ────────────────────────────────────────────────────────

def p1_make_figure(rel_pos_results, decay_distances, decay_means,
                   head_dim: int = 64, base: float = 10_000.0):
    cos, sin = precompute_rope_freqs(head_dim, max_seq_len=256, base=base)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    TITLE_C = "#e6edf3"
    LABEL_C = "#8b949e"
    GRID_C  = "#21262d"
    ACCENT  = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]

    ax_style = dict(facecolor="#161b22", grid_color=GRID_C)

    def styled_ax(ax):
        ax.set_facecolor(ax_style["facecolor"])
        ax.tick_params(colors=LABEL_C, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)
        ax.grid(True, color=GRID_C, linewidth=0.6, alpha=0.7)
        return ax

    # ── Panel A: frequency spectrum θ_i across pair indices ──────────────
    ax1 = styled_ax(fig.add_subplot(gs[0, 0]))
    pair_idx = np.arange(head_dim // 2)
    i_t      = torch.arange(0, head_dim, 2, dtype=torch.float32)
    theta_i  = (1.0 / (base ** (i_t / head_dim))).numpy()
    ax1.semilogy(pair_idx, theta_i, color=ACCENT[0], linewidth=2, marker="o",
                 markersize=3)
    ax1.set_title("Frequency spectrum  θᵢ = base^{−2i/d}", color=TITLE_C,
                  fontsize=9, pad=6)
    ax1.set_xlabel("Pair index  i", color=LABEL_C, fontsize=8)
    ax1.set_ylabel("θᵢ  (log scale)", color=LABEL_C, fontsize=8)
    ax1.axhline(1.0, color=ACCENT[2], linestyle="--", linewidth=0.8,
                alpha=0.6, label="fast (local)")
    ax1.axhline(base**-1, color=ACCENT[3], linestyle="--", linewidth=0.8,
                alpha=0.6, label="slow (global)")
    ax1.legend(fontsize=7, facecolor="#161b22", labelcolor=LABEL_C,
               edgecolor=GRID_C)

    # ── Panel B: rotation angle m·θᵢ for selected positions ─────────────
    ax2 = styled_ax(fig.add_subplot(gs[0, 1]))
    for m_pos, col in zip([1, 8, 32, 128], ACCENT):
        angles = (m_pos * theta_i)           # m·θᵢ for this position
        ax2.plot(pair_idx, angles % (2 * math.pi), color=col, linewidth=1.5,
                 label=f"m={m_pos}")
    ax2.set_title("Rotation angle  m·θᵢ mod 2π", color=TITLE_C, fontsize=9, pad=6)
    ax2.set_xlabel("Pair index  i", color=LABEL_C, fontsize=8)
    ax2.set_ylabel("Angle (radians)", color=LABEL_C, fontsize=8)
    ax2.legend(fontsize=7, facecolor="#161b22", labelcolor=LABEL_C, edgecolor=GRID_C)

    # ── Panel C: Long-term decay ──────────────────────────────────────────
    ax3 = styled_ax(fig.add_subplot(gs[0, 2]))
    ax3.plot(decay_distances, decay_means, color=ACCENT[0], linewidth=2)
    ax3.fill_between(decay_distances, decay_means, alpha=0.15, color=ACCENT[0])
    ax3.set_title("Long-term decay: E[|⟨q_m, k_n⟩|] vs |m−n|",
                  color=TITLE_C, fontsize=9, pad=6)
    ax3.set_xlabel("|m − n|  (relative distance)", color=LABEL_C, fontsize=8)
    ax3.set_ylabel("Mean |dot product|", color=LABEL_C, fontsize=8)
    ax3.text(0.95, 0.92, "High-freq pairs\ndecohere → decay",
             transform=ax3.transAxes, ha="right", va="top",
             color=LABEL_C, fontsize=7, style="italic")

    # ── Panel D: Relative position property (numeric proof) ──────────────
    ax4 = styled_ax(fig.add_subplot(gs[1, 0]))
    deltas  = [r[2] for r in rel_pos_results]
    dot_mn  = [r[3] for r in rel_pos_results]
    dot_sh  = [r[4] for r in rel_pos_results]
    x_ticks = range(len(rel_pos_results))
    ax4.bar([x - 0.2 for x in x_ticks], dot_mn, width=0.35,
            color=ACCENT[0], label="dot(q_m, k_n)")
    ax4.bar([x + 0.2 for x in x_ticks], dot_sh, width=0.35,
            color=ACCENT[1], alpha=0.8, label="dot(q_{m+Δ}, k_{n+Δ})")
    ax4.set_xticks(x_ticks)
    ax4.set_xticklabels([f"rel={d:+d}" for d in deltas], fontsize=6,
                         rotation=30, color=LABEL_C)
    ax4.set_title("Relative-position property\n(bars should be identical)",
                  color=TITLE_C, fontsize=9, pad=6)
    ax4.set_ylabel("Dot product value", color=LABEL_C, fontsize=8)
    ax4.legend(fontsize=7, facecolor="#161b22", labelcolor=LABEL_C, edgecolor=GRID_C)

    # ── Panel E: Attention heat-map (RoPE applied, small example) ─────────
    ax5 = styled_ax(fig.add_subplot(gs[1, 1]))
    S_demo = 24
    attn_module = RoPEMultiHeadAttention(d_model=32, n_heads=1, max_seq_len=128)
    attn_module.eval()
    with torch.no_grad():
        x_demo = torch.randn(1, S_demo, 32)
        _, W = attn_module(x_demo, return_attn_weights=True)
    W_np = W[0, 0].numpy()   # (S, S) head 0
    im = ax5.imshow(W_np, cmap="Blues", aspect="auto",
                    interpolation="nearest", vmin=0)
    plt.colorbar(im, ax=ax5, fraction=0.04, pad=0.02)
    ax5.set_title("RoPE attention weights (1 head, causal)",
                  color=TITLE_C, fontsize=9, pad=6)
    ax5.set_xlabel("Key position", color=LABEL_C, fontsize=8)
    ax5.set_ylabel("Query position", color=LABEL_C, fontsize=8)

    # ── Panel F: cos/sin patterns for two extreme frequency pairs ─────────
    ax6 = styled_ax(fig.add_subplot(gs[1, 2]))
    positions_np = np.arange(128)
    # fastest pair (i=0): cos(m·θ_0) where θ_0 ≈ 1
    cos_fast = cos[:128, 0].numpy()
    # slowest pair (i=d/2-1): cos(m·θ_{d/2-1})
    cos_slow = cos[:128, head_dim // 2 - 1].numpy()
    ax6.plot(positions_np, cos_fast, color=ACCENT[2], linewidth=1.2,
             label=f"pair i=0  (θ≈{theta_i[0]:.2f}, fast/local)")
    ax6.plot(positions_np, cos_slow, color=ACCENT[3], linewidth=1.2,
             label=f"pair i={head_dim//2-1}  (θ≈{theta_i[-1]:.4f}, slow/global)")
    ax6.set_title("Cosine patterns: fast vs slow frequency pair",
                  color=TITLE_C, fontsize=9, pad=6)
    ax6.set_xlabel("Position m", color=LABEL_C, fontsize=8)
    ax6.set_ylabel("cos(m·θᵢ)", color=LABEL_C, fontsize=8)
    ax6.legend(fontsize=7, facecolor="#161b22", labelcolor=LABEL_C, edgecolor=GRID_C)

    fig.suptitle(
        "Problem 1 — RoPE from Scratch: Rotation in Complex Space",
        color=TITLE_C, fontsize=13, fontweight="bold", y=0.98,
    )
    path = "/mnt/user-data/outputs/p1_rope_verification.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[P1] Figure saved → {path}")
    return path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PROBLEM 2 — YaRN & Context-Length Extrapolation                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─── Scaling strategies: each returns a scaled inv_freq (d/2,) tensor ────────

def vanilla_rope_inv_freq(head_dim: int, base: float = 10_000.0) -> torch.Tensor:
    """Baseline: standard RoPE frequencies, no scaling."""
    i = torch.arange(0, head_dim, 2, dtype=torch.float32)
    return 1.0 / (base ** (i / head_dim))


def position_interpolation_inv_freq(
    head_dim: int,
    original_max_len: int,
    new_max_len: int,
    base: float = 10_000.0,
) -> torch.Tensor:
    """
    Position Interpolation (Chen et al., 2023).

    Uniformly compress position indices: m → m / scale_factor.
    Equivalent to dividing ALL frequencies by the scale factor.

    Strength : guarantees no out-of-distribution rotation angles.
    Weakness : compresses high-frequency (local) pairs equally, blurring
               fine-grained position distinctions between nearby tokens.
    """
    scale = new_max_len / original_max_len
    inv_freq = vanilla_rope_inv_freq(head_dim, base)
    return inv_freq / scale          # every frequency shrunk uniformly


def ntk_aware_inv_freq(
    head_dim: int,
    original_max_len: int,
    new_max_len: int,
    base: float = 10_000.0,
) -> torch.Tensor:
    """
    NTK-aware scaling (kaiokendev / bloc97, 2023).

    Instead of compressing positions, stretch the frequency BASE by α,
    so that low-frequency (long-wavelength) pairs absorb most of the
    extrapolation burden while high-frequency pairs are nearly untouched.

    α = scale_factor^{d / (d-2)}
    new_base = base * α
    θ'_i = new_base^{-2i/d}

    Effect on θ_i:
      i=0    (fastest) : θ'_0 = new_base^0 = 1 → unchanged ✓
      i=d/2-1 (slowest): θ' << θ → much smaller → long wavelength ✓
    """
    scale = new_max_len / original_max_len
    # α derived from requiring the slowest pair to cover the new length
    alpha = scale ** (head_dim / (head_dim - 2))
    new_base = base * alpha
    i = torch.arange(0, head_dim, 2, dtype=torch.float32)
    return 1.0 / (new_base ** (i / head_dim))


def yarn_inv_freq(
    head_dim: int,
    original_max_len: int,
    new_max_len: int,
    base: float = 10_000.0,
    beta_fast: float = 32.0,   # high-freq boundary (wavelength units)
    beta_slow: float = 1.0,    # low-freq boundary
) -> Tuple[torch.Tensor, float]:
    """
    YaRN: NTK-by-parts interpolation (Peng et al., 2023).

    Classifies each of the d/2 frequency pairs into one of three zones
    based on its wavelength  λ_i = 2π / θ_i:

      Zone 1 (high-freq, λ < high_freq_wavelen):
          γ = 1  →  θ'_i = θ_i          (no scaling, local resolution kept)

      Zone 2 (low-freq, λ > low_freq_wavelen):
          γ = 0  →  θ'_i = θ_i / s      (full interpolation, like PI)

      Zone 3 (mid-band):
          γ ∈ (0,1)  →  θ'_i = θ_i * [(1-γ)/s + γ]   (smooth blend)

    Blending formula:
        γ(i) = (L_orig / λ_i  −  β_slow) / (β_fast − β_slow)
    clamped to [0, 1].

    Additionally returns mscale, a temperature-correction factor that
    compensates for the attention-score magnitude shift caused by
    frequency rescaling:
        mscale = 0.1 · ln(s) + 1.0
    """
    s = new_max_len / original_max_len

    # Base vanilla frequencies
    i   = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv = 1.0 / (base ** (i / head_dim))          # θ_i

    # Wavelength per frequency pair  λ_i = 2π / θ_i
    wavelens = 2 * math.pi / inv

    # Zone boundaries (in tokens)
    low_freq_wavelen  = original_max_len / beta_slow   # long   (low-freq  threshold)
    high_freq_wavelen = original_max_len / beta_fast   # short  (high-freq threshold)

    # Smooth ramp γ(i):
    #   γ = 1  if λ < high_freq_wavelen   (high freq → no scaling)
    #   γ = 0  if λ > low_freq_wavelen    (low  freq → full PI)
    #   γ = smooth blend in between
    smooth = torch.clamp(
        (wavelens / low_freq_wavelen - 1.0 / beta_fast) / (1.0 / beta_slow - 1.0 / beta_fast),
        min=0.0, max=1.0
    )
    # Equivalent but more explicit:
    gamma = torch.where(
        wavelens < high_freq_wavelen,
        torch.ones_like(inv),
        torch.where(
            wavelens > low_freq_wavelen,
            torch.zeros_like(inv),
            # linear ramp: 1 at high-freq end, 0 at low-freq end
            (original_max_len / wavelens - beta_slow) / (beta_fast - beta_slow),
        )
    )
    gamma = gamma.clamp(0.0, 1.0)

    # Blended inverse frequency
    inv_scaled = inv * ((1.0 - gamma) / s + gamma)

    # ── mscale: attention temperature correction ───────────────────────────
    # Frequency rescaling slightly flattens the attention distribution.
    # Multiply attention logits by mscale to restore sharpness.
    mscale = 0.1 * math.log(s) + 1.0 if s > 1.0 else 1.0

    return inv_scaled, mscale, gamma


# ─── Comparison analysis ─────────────────────────────────────────────────────

def p2_run_analysis(
    head_dim: int = 64,
    original_max_len: int = 128,   # ← "training length"
    new_max_len: int = 256,        # ← "target length" = 2 × training
    base: float = 10_000.0,
):
    """
    Compare vanilla RoPE, PI, NTK-aware, and YaRN by examining:
      1. Per-dimension scaling factors  θ'_i / θ_i
      2. Effective wavelengths after scaling
      3. Attention score patterns for a simple Q,K pair over long ranges
      4. mscale temperature factor

    The 'test on 2× training length' means new_max_len = 2 × original_max_len.
    """
    print(f"\n[P2] Comparing RoPE extension methods")
    print(f"     original_max_len={original_max_len}  new_max_len={new_max_len}  "
          f"scale_factor={new_max_len/original_max_len:.1f}×  head_dim={head_dim}")

    vanilla_freq = vanilla_rope_inv_freq(head_dim, base)
    pi_freq      = position_interpolation_inv_freq(head_dim, original_max_len, new_max_len, base)
    ntk_freq     = ntk_aware_inv_freq(head_dim, original_max_len, new_max_len, base)
    yarn_freq, mscale, gamma = yarn_inv_freq(head_dim, original_max_len, new_max_len, base)

    scale_pi  = (pi_freq  / vanilla_freq).numpy()
    scale_ntk = (ntk_freq / vanilla_freq).numpy()
    scale_yarn= (yarn_freq/ vanilla_freq).numpy()

    print(f"\n     mscale (YaRN attention temperature correction) = {mscale:.4f}")
    print(f"     Expected: ~{0.1 * math.log(new_max_len/original_max_len) + 1:.4f}")

    pair_idx = np.arange(head_dim // 2)
    # Wavelengths for vanilla
    wavelens_vanilla = (2 * math.pi / vanilla_freq).numpy()

    # Fraction of dimensions in each YaRN zone
    n_high  = (gamma == 1.0).sum().item()
    n_low   = (gamma == 0.0).sum().item()
    n_mid   = head_dim // 2 - n_high - n_low
    print(f"\n     YaRN zone breakdown  (out of {head_dim//2} pairs):")
    print(f"       High-freq (γ=1, no scale) : {n_high:3d}  ({100*n_high/(head_dim//2):.0f}%)")
    print(f"       Mid-band  (0<γ<1)         : {n_mid:3d}  ({100*n_mid/(head_dim//2):.0f}%)")
    print(f"       Low-freq  (γ=0, full PI)  : {n_low:3d}  ({100*n_low/(head_dim//2):.0f}%)")

    # ── Attention score decay curves for each method ───────────────────────
    # Use a fixed q,k pair; compute dot product at relative distances 0..new_max_len
    torch.manual_seed(1)
    q_raw = torch.randn(1, 1, 1, head_dim)
    k_raw = torch.randn(1, 1, 1, head_dim)

    methods = {
        "Vanilla RoPE": vanilla_freq,
        "Pos. Interp.": pi_freq,
        "NTK-aware":    ntk_freq,
        "YaRN":         yarn_freq,
    }
    distances   = list(range(0, new_max_len + 1, 4))
    decay_curves = {}

    for name, freq in methods.items():
        cos, sin = precompute_rope_freqs(head_dim, new_max_len + 10,
                                          custom_inv_freq=freq)
        dots = []
        ref_pos = new_max_len // 2   # anchor query position in the middle
        for dist in distances:
            n_pos = max(0, ref_pos - dist)
            q_rot, _ = apply_rotary_pos_emb(
                q_raw, q_raw, cos, sin, position_ids=torch.tensor([ref_pos]))
            _, k_rot = apply_rotary_pos_emb(
                q_raw, k_raw, cos, sin, position_ids=torch.tensor([n_pos]))
            dots.append(abs((q_rot * k_rot).sum().item()))
        decay_curves[name] = dots
        print(f"     {name:20s}  |  dot at dist=0: {dots[0]:.3f}  "
              f"dist={distances[-1]}: {dots[-1]:.4f}")

    return {
        "pair_idx": pair_idx,
        "vanilla_freq": vanilla_freq.numpy(),
        "pi_freq":      pi_freq.numpy(),
        "ntk_freq":     ntk_freq.numpy(),
        "yarn_freq":    yarn_freq.numpy(),
        "gamma":        gamma.numpy(),
        "wavelens_vanilla": wavelens_vanilla,
        "scale_pi":     scale_pi,
        "scale_ntk":    scale_ntk,
        "scale_yarn":   scale_yarn,
        "distances":    distances,
        "decay_curves": decay_curves,
        "mscale":       mscale,
        "original_max_len": original_max_len,
        "new_max_len":  new_max_len,
    }


def p2_make_figure(data: dict):
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.40)

    TITLE_C = "#e6edf3"
    LABEL_C = "#8b949e"
    GRID_C  = "#21262d"
    COLORS  = {"Vanilla RoPE": "#58a6ff",
                "Pos. Interp.": "#3fb950",
                "NTK-aware":    "#ffa657",
                "YaRN":         "#d2a8ff"}

    def styled_ax(ax):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=LABEL_C, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)
        ax.grid(True, color=GRID_C, linewidth=0.6, alpha=0.7)
        return ax

    pair_idx = data["pair_idx"]

    # ── A: Per-dimension scaling ratio θ'_i / θ_i ────────────────────────
    ax1 = styled_ax(fig.add_subplot(gs[0, 0]))
    ax1.plot(pair_idx, data["scale_pi"],   color=COLORS["Pos. Interp."], lw=2,
             label="PI  (uniform)")
    ax1.plot(pair_idx, data["scale_ntk"],  color=COLORS["NTK-aware"],    lw=2,
             label="NTK-aware")
    ax1.plot(pair_idx, data["scale_yarn"], color=COLORS["YaRN"],         lw=2,
             label="YaRN (NTK-by-parts)")
    ax1.axhline(1.0, color="#8b949e", linestyle="--", lw=1, label="no scaling")
    ax1.axhline(data["original_max_len"] / data["new_max_len"],
                color="#f78166", linestyle=":", lw=1, label=f"full PI (÷{data['new_max_len']//data['original_max_len']})")
    ax1.set_title("Per-dimension scaling factor  θ'ᵢ / θᵢ",
                  color=TITLE_C, fontsize=9, pad=6)
    ax1.set_xlabel("Pair index  i", color=LABEL_C, fontsize=8)
    ax1.set_ylabel("Scale  (1.0 = no change)", color=LABEL_C, fontsize=8)
    ax1.legend(fontsize=7, facecolor="#161b22", labelcolor=LABEL_C, edgecolor=GRID_C)

    # ── B: YaRN ramp function γ(i) ───────────────────────────────────────
    ax2 = styled_ax(fig.add_subplot(gs[0, 1]))
    ax2.fill_between(pair_idx, data["gamma"], alpha=0.25, color=COLORS["YaRN"])
    ax2.plot(pair_idx, data["gamma"], color=COLORS["YaRN"], lw=2.5)
    ax2.axhline(0.5, color=LABEL_C, linestyle="--", lw=0.8, alpha=0.5)
    ax2.set_title("YaRN ramp function  γ(i)\n"
                  "1 = high-freq (no scale)  |  0 = low-freq (full PI)",
                  color=TITLE_C, fontsize=9, pad=6)
    ax2.set_xlabel("Pair index  i", color=LABEL_C, fontsize=8)
    ax2.set_ylabel("γ  (interpolation weight)", color=LABEL_C, fontsize=8)
    ax2.set_ylim(-0.05, 1.05)

    # ── C: Wavelength distribution (log scale) with zone boundaries ──────
    ax3 = styled_ax(fig.add_subplot(gs[0, 2]))
    ax3.semilogy(pair_idx, data["wavelens_vanilla"],
                 color=COLORS["Vanilla RoPE"], lw=2, label="Vanilla wavelength λᵢ")
    L = data["original_max_len"]
    ax3.axhline(L / 1.0,  color="#f78166", lw=1.2, linestyle="--",
                label=f"low-freq threshold  (L={L})")
    ax3.axhline(L / 32.0, color="#3fb950", lw=1.2, linestyle=":",
                label=f"high-freq threshold (L/32={L//32})")
    ax3.fill_between(pair_idx,
                     [L / 32.0] * len(pair_idx), [L] * len(pair_idx),
                     alpha=0.08, color=COLORS["YaRN"], label="YaRN mid-band zone")
    ax3.set_title("Wavelength  λᵢ = 2π/θᵢ  with YaRN zone boundaries",
                  color=TITLE_C, fontsize=9, pad=6)
    ax3.set_xlabel("Pair index  i", color=LABEL_C, fontsize=8)
    ax3.set_ylabel("Wavelength (tokens, log scale)", color=LABEL_C, fontsize=8)
    ax3.legend(fontsize=7, facecolor="#161b22", labelcolor=LABEL_C, edgecolor=GRID_C)

    # ── D: Effective inv_freq comparison (log scale) ──────────────────────
    ax4 = styled_ax(fig.add_subplot(gs[1, 0]))
    ax4.semilogy(pair_idx, data["vanilla_freq"], color=COLORS["Vanilla RoPE"],
                 lw=2, label="Vanilla")
    ax4.semilogy(pair_idx, data["pi_freq"],      color=COLORS["Pos. Interp."],
                 lw=2, linestyle="--", label="PI")
    ax4.semilogy(pair_idx, data["ntk_freq"],     color=COLORS["NTK-aware"],
                 lw=2, linestyle="-.", label="NTK-aware")
    ax4.semilogy(pair_idx, data["yarn_freq"],    color=COLORS["YaRN"],
                 lw=2.5, label="YaRN")
    ax4.set_title("Effective inv_freq  θ'ᵢ  after scaling  (log scale)",
                  color=TITLE_C, fontsize=9, pad=6)
    ax4.set_xlabel("Pair index  i", color=LABEL_C, fontsize=8)
    ax4.set_ylabel("θ'ᵢ  (log scale)", color=LABEL_C, fontsize=8)
    ax4.legend(fontsize=7, facecolor="#161b22", labelcolor=LABEL_C, edgecolor=GRID_C)

    # ── E: Attention-score decay at 2× length ─────────────────────────────
    ax5 = styled_ax(fig.add_subplot(gs[1, 1]))
    for name, curve in data["decay_curves"].items():
        ax5.plot(data["distances"], curve,
                 color=COLORS[name], lw=2, label=name)
    ax5.axvline(data["original_max_len"], color="#f78166",
                linestyle="--", lw=1.2, label="Training length L")
    ax5.set_title(f"Attention score |⟨q_m, k_n⟩| vs distance\n"
                  f"(test sequences: {data['new_max_len']} = 2×{data['original_max_len']} tokens)",
                  color=TITLE_C, fontsize=9, pad=6)
    ax5.set_xlabel("Relative distance |m − n|", color=LABEL_C, fontsize=8)
    ax5.set_ylabel("|Dot product|", color=LABEL_C, fontsize=8)
    ax5.legend(fontsize=7, facecolor="#161b22", labelcolor=LABEL_C, edgecolor=GRID_C)

    # ── F: Summary table ──────────────────────────────────────────────────
    ax6 = styled_ax(fig.add_subplot(gs[1, 2]))
    ax6.axis("off")
    table_data = [
        ["Method", "Local res.", "Extrapolation", "Fine-tune?", "mscale"],
        ["Vanilla RoPE", "✓ Full", "✗ Fails >L", "N/A", "1.00"],
        ["Pos. Interp.", "✗ Blurred", "✓ Stable", "Yes (more)", "1.00"],
        ["NTK-aware",    "≈ Full",  "✓ Good",   "Less needed", "1.00"],
        ["YaRN",         "✓ Full",  "✓ Best",   "Minimal",  f"{data['mscale']:.3f}"],
    ]
    tbl = ax6.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f6feb")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#161b22")
            cell.set_text_props(color=TITLE_C)
        cell.set_edgecolor(GRID_C)
    ax6.set_title("Method comparison", color=TITLE_C, fontsize=9, pad=6)

    fig.suptitle(
        f"Problem 2 — YaRN & Context Extension  "
        f"({data['original_max_len']}→{data['new_max_len']} tokens, "
        f"{data['new_max_len']//data['original_max_len']}× scale)",
        color=TITLE_C, fontsize=13, fontweight="bold", y=0.98,
    )
    path = "/mnt/user-data/outputs/p2_yarn_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[P2] Figure saved → {path}")
    return path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PROBLEM 3 — Generalisation Comparison: RoPE · ALiBi · Learned APE    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─── ALiBi slopes ────────────────────────────────────────────────────────────

def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """
    Compute ALiBi head-specific distance penalty slopes.

    From Press et al. (2022):
        slope_h = 2^{−8h/H}  for h = 1, …, H

    Heads with larger h get a steeper penalty → attend more locally.
    Heads with smaller h have a gentler slope → can attend more globally.

    For non-power-of-2 H, the paper suggests interpolating; here we
    use the simple formula directly for clarity.

    Returns: (n_heads,) tensor of positive slope values.
    """
    # slope_h = 2^{-8h/H} = exp(-8h ln2 / H) for h = 1..H
    h     = torch.arange(1, n_heads + 1, dtype=torch.float32)
    slopes = torch.pow(2.0, -8.0 * h / n_heads)
    return slopes   # shape: (H,)


def compute_alibi_bias(slopes: torch.Tensor, seq_len: int,
                        device: torch.device) -> torch.Tensor:
    """
    Compute the ALiBi attention bias matrix.

    For query position i and key position j:
        bias[h, i, j] = −slope_h · |i − j|

    (negative because this penalises distance; added to raw attention scores)

    Returns: (n_heads, seq_len, seq_len) bias tensor.
    """
    slopes = slopes.to(device)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    # |i - j|  for all query-key pairs
    rel_dist = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))  # (S, S)
    # Apply per-head slope:  (H, 1, 1) * (1, S, S)  →  (H, S, S)
    bias = -slopes.view(-1, 1, 1) * rel_dist.unsqueeze(0)
    return bias


# ─── Attention modules ───────────────────────────────────────────────────────

class PlainMultiHeadAttention(nn.Module):
    """Standard dot-product attention, no built-in positional encoding.
    Used by Learned APE (position info comes from the input embedding)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, **_):
        B, S, D = x.shape
        H, hd   = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, S, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, hd).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(causal, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out), weights


class ALiBiMultiHeadAttention(nn.Module):
    """
    ALiBi (Attention with Linear Biases) — Press et al., 2022.

    No positional embeddings are added to the token representation at all.
    Instead, the attention score for each head h is modified as:
        score[h, i, j] = (q_i · k_j) / √d  −  slope_h · |i − j|

    Key properties:
    • slopes are fixed (not learned) and differ per head
    • the bias is a LINEAR function of distance (not sinusoidal)
    • no upper bound on sequence length — works at any length out-of-the-box
    • because the bias only depends on relative distance |i-j|, it is
      perfectly translation-invariant by construction
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024):
        super().__init__()
        self.n_heads    = n_heads
        self.head_dim   = d_model // n_heads
        self.scale      = self.head_dim ** -0.5
        self.q_proj     = nn.Linear(d_model, d_model, bias=False)
        self.k_proj     = nn.Linear(d_model, d_model, bias=False)
        self.v_proj     = nn.Linear(d_model, d_model, bias=False)
        self.out_proj   = nn.Linear(d_model, d_model, bias=False)

        # Slopes: fixed, one per head, registered as buffer (not a parameter)
        slopes = get_alibi_slopes(n_heads)
        self.register_buffer("slopes", slopes)          # (H,)

    def forward(self, x, **_):
        B, S, D = x.shape
        H, hd   = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, S, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, hd).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, S, S)

        # Add ALiBi bias:  (H, S, S) broadcast over batch dimension
        alibi = compute_alibi_bias(self.slopes, S, x.device)         # (H, S, S)
        scores = scores + alibi.unsqueeze(0)                          # (B, H, S, S)

        # Causal mask
        causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(causal, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out), weights


# ─── Transformer building blocks ─────────────────────────────────────────────

@dataclass
class ModelConfig:
    vocab_size:  int   = 64
    d_model:     int   = 64
    n_heads:     int   = 4
    n_layers:    int   = 3
    d_ff:        int   = 128
    train_len:   int   = 32    # training sequence length
    max_seq_len: int   = 128   # buffer for cos/sin tables
    pos_encoding: str  = "rope" # "rope" | "alibi" | "learned_ape"


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-LN transformer block wrapping any attention module."""

    def __init__(self, d_model: int, d_ff: int, attn_module: nn.Module):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = attn_module
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = FFN(d_model, d_ff)

    def forward(self, x, **kwargs):
        attn_out, weights = self.attn(self.norm1(x), **kwargs)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, weights


class MiniLM(nn.Module):
    """
    Tiny autoregressive LM. The only difference between the three variants
    is which attention module is used:

    • rope        → RoPEMultiHeadAttention  (position encoded in Q, K rotation)
    • alibi       → ALiBiMultiHeadAttention (position as distance penalty bias)
    • learned_ape → PlainMultiHeadAttention (position as learned input embedding)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg   = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Optional learned position embeddings (Learned APE only)
        # NOTE: table is sized to max_seq_len, but positions 0..train_len-1
        #       are the ONLY ones ever seen during training.
        #       Positions >= train_len get random-init embeddings → degraded signal.
        self.pos_embed = (
            nn.Embedding(cfg.max_seq_len, cfg.d_model)
            if cfg.pos_encoding == "learned_ape"
            else None
        )

        # Build one attention module per layer
        def make_attn():
            if cfg.pos_encoding == "rope":
                return RoPEMultiHeadAttention(
                    cfg.d_model, cfg.n_heads, cfg.max_seq_len)
            elif cfg.pos_encoding == "alibi":
                return ALiBiMultiHeadAttention(
                    cfg.d_model, cfg.n_heads, cfg.max_seq_len)
            else:  # learned_ape
                return PlainMultiHeadAttention(cfg.d_model, cfg.n_heads)

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.d_ff, make_attn())
             for _ in range(cfg.n_layers)]
        )
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.lm_head  = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight   # weight tying

    def forward(self, tokens: torch.Tensor,
                return_attn: bool = False) -> tuple:
        B, S = tokens.shape
        x = self.embed(tokens)                    # (B, S, D)

        if self.pos_embed is not None:
            # Clamp so we don't go out-of-bounds for the embedding table
            # (this is what deployed systems do with Learned APE at inference)
            pos_ids = torch.arange(S, device=tokens.device).clamp(
                max=self.cfg.max_seq_len - 1)
            x = x + self.pos_embed(pos_ids)

        all_weights = []
        for block in self.blocks:
            x, w = block(x)
            if return_attn:
                all_weights.append(w)

        x      = self.norm_out(x)
        logits = self.lm_head(x)                  # (B, S, V)
        return (logits, all_weights) if return_attn else (logits, None)

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Next-token prediction loss over a full sequence."""
        logits, _ = self(tokens[:, :-1])          # (B, S-1, V)
        targets   = tokens[:, 1:]                  # (B, S-1)
        return F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            targets.reshape(-1),
        )


# ─── Data generation ─────────────────────────────────────────────────────────

def make_batch(batch_size: int, seq_len: int, vocab_size: int,
               device: torch.device) -> torch.Tensor:
    """
    Arithmetic sequences:  a_i = (start + step × i) mod V

    Why this task tests position generalisation:
    • A model that tracks position correctly can exploit the constant-step
      rule at any distance from the start.
    • Learned APE at positions ≥ train_len receives untrained random embeddings
      → wrong positional signal → cannot correctly predict offsets beyond
      what was learned during training at those indices.
    • RoPE and ALiBi encode position through rotation / distance penalty
      → generalise naturally beyond training length.
    """
    starts = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    steps  = torch.randint(1, vocab_size, (batch_size, 1), device=device)
    pos    = torch.arange(seq_len, device=device).unsqueeze(0)
    return (starts + steps * pos) % vocab_size      # (B, S)


# ─── Training ────────────────────────────────────────────────────────────────

def train_model(model: MiniLM, cfg: ModelConfig,
                n_steps: int = 2000, batch_size: int = 64) -> List[float]:
    """
    Train on sequences of length cfg.train_len (not cfg.max_seq_len).
    Only positions 0 … train_len-1 are ever presented during training.
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps,
                                                         eta_min=1e-4)
    losses = []
    for step in range(n_steps):
        tokens = make_batch(batch_size, cfg.train_len, cfg.vocab_size, DEVICE)
        loss   = model.loss(tokens)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % 500 == 0:
            print(f"     step {step+1:4d}/{n_steps}  loss={loss.item():.4f}")
    return losses


# ─── Evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_per_position_loss(
    model: MiniLM, cfg: ModelConfig,
    eval_len: int, n_batches: int = 50, batch_size: int = 32,
) -> np.ndarray:
    """
    Compute average cross-entropy at each token position 0 … eval_len-2.

    Positions  < cfg.train_len : in-distribution for ALL models.
    Positions >= cfg.train_len : out-of-distribution for Learned APE.
    """
    model.eval()
    pos_losses = np.zeros(eval_len - 1)
    pos_counts = np.zeros(eval_len - 1)

    for _ in range(n_batches):
        tokens = make_batch(batch_size, eval_len, cfg.vocab_size, DEVICE)
        logits, _ = model(tokens[:, :-1])          # (B, S-1, V)
        targets   = tokens[:, 1:]                   # (B, S-1)

        for pos in range(eval_len - 1):
            loss = F.cross_entropy(logits[:, pos, :], targets[:, pos])
            pos_losses[pos] += loss.item()
            pos_counts[pos] += 1

    model.train()
    return pos_losses / pos_counts


@torch.no_grad()
def get_attention_maps(model: MiniLM, cfg: ModelConfig,
                       seq_len: int) -> List[np.ndarray]:
    """Return average attention weight maps (one per layer) for a single batch."""
    model.eval()
    tokens = make_batch(1, seq_len, cfg.vocab_size, DEVICE)
    _, weights_list = model(tokens, return_attn=True)
    # Average over heads for each layer
    maps = []
    for w in weights_list:
        # w: (B=1, H, S, S)  →  average over heads  →  (S, S)
        maps.append(w[0].mean(0).cpu().numpy())
    model.train()
    return maps


# ─── Problem 3 runner ────────────────────────────────────────────────────────

def p3_run():
    N_STEPS    = 3000
    TRAIN_LEN  = 32
    EVAL_LEN   = 64    # 2× training length
    BATCH      = 64
    VOCAB      = 64

    cfg_base = ModelConfig(
        vocab_size=VOCAB, d_model=64, n_heads=4, n_layers=3,
        d_ff=128, train_len=TRAIN_LEN, max_seq_len=128,
    )

    models = {}
    train_curves = {}
    pos_losses   = {}
    attn_maps    = {}

    for pe in ["rope", "alibi", "learned_ape"]:
        print(f"\n[P3] Training  pos_encoding={pe}  "
              f"({N_STEPS} steps, train_len={TRAIN_LEN}, eval_len={EVAL_LEN})")
        cfg        = ModelConfig(**{**cfg_base.__dict__, "pos_encoding": pe})
        model      = MiniLM(cfg).to(DEVICE)
        n_params   = sum(p.numel() for p in model.parameters())
        print(f"     Parameters: {n_params:,}")

        t0              = time.time()
        train_curves[pe] = train_model(model, cfg, N_STEPS, BATCH)
        elapsed          = time.time() - t0
        print(f"     Training time: {elapsed:.1f}s  "
              f"final_loss={train_curves[pe][-1]:.4f}")

        # Evaluate at both train length and 2× train length
        pos_losses[pe]  = eval_per_position_loss(model, cfg, EVAL_LEN)
        attn_maps[pe]   = get_attention_maps(model, cfg, EVAL_LEN)
        models[pe]      = model

        mean_in  = pos_losses[pe][:TRAIN_LEN-1].mean()
        mean_out = pos_losses[pe][TRAIN_LEN-1:].mean()
        print(f"     Avg loss pos [0,{TRAIN_LEN-1}] (in-dist):  {mean_in:.4f}")
        print(f"     Avg loss pos [{TRAIN_LEN},{EVAL_LEN-1}] (out-of-dist): {mean_out:.4f}")
        print(f"     Generalisation gap: {mean_out - mean_in:.4f}")

    return {
        "models":       models,
        "train_curves": train_curves,
        "pos_losses":   pos_losses,
        "attn_maps":    attn_maps,
        "train_len":    TRAIN_LEN,
        "eval_len":     EVAL_LEN,
    }


def p3_make_figure(data: dict):
    train_len = data["train_len"]
    eval_len  = data["eval_len"]

    COLORS = {
        "rope":        "#58a6ff",
        "alibi":       "#3fb950",
        "learned_ape": "#ffa657",
    }
    LABELS = {
        "rope":        "Vanilla RoPE",
        "alibi":       "ALiBi",
        "learned_ape": "Learned APE",
    }
    TITLE_C = "#e6edf3"
    LABEL_C = "#8b949e"
    GRID_C  = "#21262d"

    def styled_ax(ax):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=LABEL_C, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)
        ax.grid(True, color=GRID_C, linewidth=0.6, alpha=0.7)
        return ax

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

    # ── A: Training loss curves ───────────────────────────────────────────
    ax1 = styled_ax(fig.add_subplot(gs[0, :2]))
    for pe, curve in data["train_curves"].items():
        xs = np.arange(len(curve))
        # Smooth with rolling mean
        smooth = np.convolve(curve, np.ones(40)/40, mode="valid")
        ax1.plot(xs, curve, color=COLORS[pe], alpha=0.2, linewidth=0.7)
        ax1.plot(xs[19:-20], smooth, color=COLORS[pe], linewidth=2,
                 label=LABELS[pe])
    ax1.set_title(f"Training loss (sequence length = {train_len})",
                  color=TITLE_C, fontsize=9, pad=6)
    ax1.set_xlabel("Step", color=LABEL_C, fontsize=8)
    ax1.set_ylabel("Cross-entropy loss", color=LABEL_C, fontsize=8)
    ax1.legend(fontsize=8, facecolor="#161b22", labelcolor=LABEL_C, edgecolor=GRID_C)

    # ── B: Per-position loss at 2× length ─────────────────────────────────
    ax2 = styled_ax(fig.add_subplot(gs[1, :2]))
    positions = np.arange(eval_len - 1)
    for pe, losses in data["pos_losses"].items():
        ax2.plot(positions, losses, color=COLORS[pe], linewidth=2,
                 label=LABELS[pe])
    ax2.axvline(train_len - 1, color="#f78166", linestyle="--", linewidth=1.5,
                label=f"Training length boundary (pos {train_len-1})")
    ax2.fill_betweenx([0, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 5],
                       train_len - 1, eval_len - 2,
                       alpha=0.05, color="#f78166", label="Out-of-distribution zone")
    ax2.set_title(
        f"Per-position loss on sequences of length {eval_len} = 2×{train_len}\n"
        f"(trained only on length {train_len} — right of dashed line is OOD)",
        color=TITLE_C, fontsize=9, pad=6,
    )
    ax2.set_xlabel("Token position", color=LABEL_C, fontsize=8)
    ax2.set_ylabel("Cross-entropy loss", color=LABEL_C, fontsize=8)
    ax2.legend(fontsize=7.5, facecolor="#161b22", labelcolor=LABEL_C, edgecolor=GRID_C)
    ax2.set_xlim(0, eval_len - 2)
    ax2.set_ylim(bottom=0)

    # ── C: Summary bar chart (in-dist vs out-of-dist mean loss) ──────────
    ax3 = styled_ax(fig.add_subplot(gs[0:2, 2]))
    pe_list = list(data["pos_losses"].keys())
    x       = np.arange(len(pe_list))
    width   = 0.35
    in_means  = [data["pos_losses"][pe][:train_len-1].mean() for pe in pe_list]
    out_means = [data["pos_losses"][pe][train_len-1:].mean() for pe in pe_list]

    bars1 = ax3.bar(x - width/2, in_means, width,
                    color=[COLORS[pe] for pe in pe_list], alpha=0.9,
                    label="In-distribution (≤ train_len)")
    bars2 = ax3.bar(x + width/2, out_means, width,
                    color=[COLORS[pe] for pe in pe_list], alpha=0.4,
                    edgecolor=[COLORS[pe] for pe in pe_list], linewidth=1.5,
                    label="Out-of-distribution (> train_len)")

    for bar, val in zip(list(bars1) + list(bars2), in_means + out_means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", color=TITLE_C, fontsize=7)

    ax3.set_xticks(x)
    ax3.set_xticklabels([LABELS[pe] for pe in pe_list], color=LABEL_C, fontsize=8)
    ax3.set_title("Avg loss: in-dist vs out-of-dist\n(smaller is better)",
                  color=TITLE_C, fontsize=9, pad=6)
    ax3.set_ylabel("Mean cross-entropy", color=LABEL_C, fontsize=8)
    ax3.legend(fontsize=7, facecolor="#161b22", labelcolor=LABEL_C, edgecolor=GRID_C)

    # ── D-F: Attention maps at OOD positions (last layer) ────────────────
    for col_idx, pe in enumerate(["rope", "alibi", "learned_ape"]):
        ax = styled_ax(fig.add_subplot(gs[2, col_idx]))
        # Show the last layer's attention at OOD positions (second half of seq)
        # Crop to last 32 positions of the 64-token eval sequence
        attn_map = data["attn_maps"][pe][-1]   # last layer: (eval_len, eval_len)
        # Show the query positions from train_len to eval_len-1 (OOD range)
        crop_start = train_len
        crop_end   = eval_len
        sub_map = attn_map[crop_start:crop_end, :crop_end]
        im = ax.imshow(sub_map, cmap="Blues", aspect="auto",
                       interpolation="nearest", vmin=0,
                       vmax=sub_map.max() + 1e-8)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        ax.axvline(train_len - crop_start - 0.5, color="#f78166",
                   linewidth=1.5, linestyle="--", alpha=0.8)
        ax.set_title(
            f"{LABELS[pe]}\nattn at OOD positions [{crop_start}:{crop_end}]",
            color=TITLE_C, fontsize=8, pad=4,
        )
        ax.set_xlabel("Key position (within crop)", color=LABEL_C, fontsize=7)
        ax.set_ylabel("Query position (OOD)", color=LABEL_C, fontsize=7)

    fig.suptitle(
        f"Problem 3 — Position Generalisation: RoPE · ALiBi · Learned APE\n"
        f"Train length = {train_len}  |  Test length = {eval_len} (2×)",
        color=TITLE_C, fontsize=13, fontweight="bold", y=0.99,
    )
    path = "/mnt/user-data/outputs/p3_generalisation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[P3] Figure saved → {path}")
    return path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":

    # ─── Problem 1 ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PROBLEM 1 — RoPE from Scratch")
    print(SEP)
    rel_pos_results  = p1_verify_relative_position_property(head_dim=32)
    p1_verify_norm_preservation(head_dim=64)
    print("\n[P1] Computing long-term decay curve …")
    decay_distances, decay_means = p1_compute_attention_decay(
        head_dim=64, d_model=64, n_heads=1)
    p1_path = p1_make_figure(rel_pos_results, decay_distances, decay_means,
                              head_dim=64)

    # ─── Problem 2 ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PROBLEM 2 — YaRN & Context Extension")
    print(SEP)
    p2_data = p2_run_analysis(
        head_dim=64,
        original_max_len=128,   # training length
        new_max_len=256,        # 2× training length
        base=10_000.0,
    )
    p2_path = p2_make_figure(p2_data)

    # ─── Problem 3 ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PROBLEM 3 — Position Generalisation Comparison")
    print(SEP)
    p3_data = p3_run()
    p3_path = p3_make_figure(p3_data)

    # ─── Summary ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  All problems complete. Outputs:")
    print(f"    {p1_path}")
    print(f"    {p2_path}")
    print(f"    {p3_path}")
    print(SEP)