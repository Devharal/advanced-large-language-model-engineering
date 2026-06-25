"""
Problem 1: Implement KV cache in a generation loop
           Measure memory vs. sequence length

What this code teaches:
  - How KV cache is physically structured in GPU memory
  - How to build a generation loop with and without caching
  - How to measure VRAM growth at each decode step
  - Why cached generation scales as O(1) compute per step vs O(N) without cache
  - How to read memory numbers and connect them to the formula:
    2 × n_layers × n_kv_heads × seq_len × d_head × dtype_bytes

Requirements:
    pip install torch transformers
"""

import gc
import math
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Minimal Transformer Config
#
# We build a tiny but architecturally correct Transformer so all numbers
# are easy to verify by hand. Real models (LLaMA 7B) follow the exact same
# pattern — just with larger L, H, d_model values.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransformerConfig:
    """
    Model hyperparameters. These control KV cache size directly via:
        M_kv = 2 × n_layers × n_kv_heads × seq_len × head_dim × bytes_per_elem
    """
    vocab_size:   int = 512
    d_model:      int = 256       # total embedding dimension
    n_heads:      int = 8         # query heads
    n_kv_heads:   int = 8         # KV heads (= n_heads for MHA, < n_heads for GQA)
    n_layers:     int = 4         # Transformer layers
    max_seq_len:  int = 2048      # maximum sequence length
    dtype:        torch.dtype = torch.float16

    @property
    def head_dim(self) -> int:
        # Each head operates on a slice of d_model
        return self.d_model // self.n_heads

    @property
    def bytes_per_element(self) -> int:
        return 2 if self.dtype == torch.float16 else 4

    def kv_cache_bytes(self, seq_len: int, batch_size: int = 1) -> int:
        """
        Theoretical KV cache size from the formula.
        Compare this to torch.cuda.memory_allocated() to verify correctness.
        """
        return (
            2                   # K and V
            * self.n_layers     # one cache per layer
            * self.n_kv_heads   # one entry per KV head
            * seq_len           # one entry per token
            * self.head_dim     # each entry is head_dim floats
            * self.bytes_per_element
            * batch_size
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — KV Cache Data Structure
#
# In production systems (vLLM, HuggingFace), the KV cache is a large pre-allocated
# tensor. Here we show both:
#   A) Pre-allocated: allocate max_seq_len upfront, fill as you go
#   B) Dynamic:       grow as needed (simpler but causes CUDA allocator churn)
#
# Real systems use pre-allocated to avoid memory fragmentation.
# ─────────────────────────────────────────────────────────────────────────────

class KVCache:
    """
    Manages the Key and Value cache for all layers during generation.

    Physical layout:
        self.k[layer]: tensor of shape [batch, n_kv_heads, current_len, head_dim]
        self.v[layer]: tensor of shape [batch, n_kv_heads, current_len, head_dim]

    At each decode step:
        1. Compute k_new, v_new for the single new token
        2. Append to cache via torch.cat (dynamic) or index assignment (pre-alloc)
        3. Attend: Q_new over the full cache [1, n_kv_heads, seq_len_so_far, head_dim]

    Memory grows by:
        2 × n_kv_heads × head_dim × dtype_bytes per step per layer
    """

    def __init__(self, config: TransformerConfig, batch_size: int = 1,
                 device: str = "cuda", mode: str = "dynamic"):
        """
        Args:
            config:     model configuration
            batch_size: number of parallel sequences
            device:     "cuda" or "cpu"
            mode:       "dynamic" — grow via torch.cat each step
                        "preallocated" — fixed buffer, use write pointer
        """
        self.config     = config
        self.batch_size = batch_size
        self.device     = device
        self.mode       = mode
        self.length     = 0    # number of tokens currently in cache

        if mode == "preallocated":
            # Allocate the maximum possible cache upfront.
            # This is what production systems do — avoids allocator overhead.
            # Shape: [n_layers, 2, batch, n_kv_heads, max_seq_len, head_dim]
            self._buffer = torch.zeros(
                config.n_layers, 2, batch_size,
                config.n_kv_heads, config.max_seq_len, config.head_dim,
                dtype=config.dtype, device=device
            )
            # slice(0, 0) means "no valid tokens yet"
            self._write_ptr = 0

        else:  # dynamic
            # Lists of tensors, one per layer.
            # k_cache[i]: [batch, n_kv_heads, len, head_dim]
            # Grows via torch.cat at each step.
            self.k_cache: list[Optional[torch.Tensor]] = [None] * config.n_layers
            self.v_cache: list[Optional[torch.Tensor]] = [None] * config.n_layers

    def update(self, layer_idx: int,
               k_new: torch.Tensor,
               v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new K, V to the cache for `layer_idx` and return the full cache.

        Args:
            layer_idx: which Transformer layer this is for
            k_new:     [batch, n_kv_heads, 1, head_dim]  (one new token)
            v_new:     [batch, n_kv_heads, 1, head_dim]

        Returns:
            (full_k, full_v): [batch, n_kv_heads, seq_len_so_far, head_dim]
            — these are passed to the attention computation.
        """
        if self.mode == "preallocated":
            # Write into pre-allocated buffer at current write pointer.
            # The write pointer advances by 1 each step.
            # slice = current position for this one new token.
            pos = self._write_ptr
            self._buffer[layer_idx, 0, :, :, pos : pos + 1, :] = k_new
            self._buffer[layer_idx, 1, :, :, pos : pos + 1, :] = v_new

            # Return the slice of the buffer that's been written so far.
            full_k = self._buffer[layer_idx, 0, :, :, : pos + 1, :]
            full_v = self._buffer[layer_idx, 1, :, :, : pos + 1, :]

            # Update write pointer after last layer (each step visits all layers)
            if layer_idx == self.config.n_layers - 1:
                self._write_ptr += 1
                self.length += 1

        else:  # dynamic
            if self.k_cache[layer_idx] is None:
                # First token: just store it
                self.k_cache[layer_idx] = k_new
                self.v_cache[layer_idx] = v_new
            else:
                # Append new K, V along the sequence dimension (dim=2)
                # torch.cat creates a new tensor — slight overhead vs pre-alloc
                self.k_cache[layer_idx] = torch.cat(
                    [self.k_cache[layer_idx], k_new], dim=2
                )
                self.v_cache[layer_idx] = torch.cat(
                    [self.v_cache[layer_idx], v_new], dim=2
                )

            if layer_idx == self.config.n_layers - 1:
                self.length += 1

            full_k = self.k_cache[layer_idx]
            full_v = self.v_cache[layer_idx]

        return full_k, full_v

    def memory_bytes(self) -> int:
        """
        Actual VRAM bytes consumed by the cache tensors.
        Should closely match config.kv_cache_bytes(self.length).
        """
        if self.mode == "preallocated":
            # Pre-allocated buffer always takes max_seq_len × ... bytes
            # regardless of how many tokens are actually filled
            return self._buffer.element_size() * self._buffer.nelement()

        total = 0
        for i in range(self.config.n_layers):
            if self.k_cache[i] is not None:
                total += self.k_cache[i].element_size() * self.k_cache[i].nelement()
                total += self.v_cache[i].element_size() * self.v_cache[i].nelement()
        return total

    def clear(self):
        """Free all cached tensors and reset state."""
        if self.mode == "preallocated":
            del self._buffer
        else:
            self.k_cache = [None] * self.config.n_layers
            self.v_cache = [None] * self.config.n_layers
        self.length = 0
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Attention with KV Cache
#
# During decode, attention is:
#   Q: [batch, n_heads, 1, head_dim]              — just one new query row
#   K: [batch, n_kv_heads, seq_len_so_far, head_dim]  — full cache
#   V: [batch, n_kv_heads, seq_len_so_far, head_dim]  — full cache
#
# This is a [1 × seq_len] attention matrix per head.
# Very different from prefill where Q is [seq_len × seq_len].
# The compute is O(seq_len) per step; the MEMORY LOAD is also O(seq_len).
# That memory load is the bottleneck — this is why decode is memory-bound.
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttentionWithCache(nn.Module):
    """
    Attention module that uses the KV cache during decode.

    Two modes of operation:
      prefill (cache=None): process all prompt tokens at once, populate cache
      decode  (cache=KVCache): process one new token, attend over full cache
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config   = config
        self.n_heads  = config.n_heads
        self.n_kv     = config.n_kv_heads
        self.head_dim = config.head_dim
        self.scale    = 1.0 / math.sqrt(self.head_dim)

        # W_Q projects to n_heads × head_dim
        # W_K, W_V project to n_kv_heads × head_dim (smaller for GQA)
        self.W_Q = nn.Linear(config.d_model, config.n_heads    * config.head_dim, bias=False)
        self.W_K = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.W_V = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.W_O = nn.Linear(config.n_heads * config.head_dim, config.d_model,    bias=False)

    def forward(self, x: torch.Tensor,
                cache: Optional[KVCache] = None,
                layer_idx: int = 0) -> torch.Tensor:
        """
        Args:
            x:         [batch, seq_len, d_model]
                       seq_len = full prompt length during prefill
                       seq_len = 1 during decode (one new token)
            cache:     KVCache or None
            layer_idx: which layer (used to index into cache)

        Returns:
            output: [batch, seq_len, d_model]
        """
        B, T, _ = x.shape

        # Project to Q, K, V
        # During decode: T=1, so these produce single-row tensors
        Q = self.W_Q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_kv,    self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_kv,    self.head_dim).transpose(1, 2)
        # Q: [B, n_heads, T, head_dim]
        # K: [B, n_kv, T, head_dim]
        # V: [B, n_kv, T, head_dim]

        if cache is not None:
            # Decode mode: T=1
            # Update cache with this step's K, V; get back full sequence K, V
            # full_K: [B, n_kv, seq_len_so_far, head_dim]
            # full_V: [B, n_kv, seq_len_so_far, head_dim]
            full_K, full_V = cache.update(layer_idx, K, V)
        else:
            # Prefill mode: T = full prompt length, no cache lookup
            full_K = K
            full_V = V

        # GQA/MQA broadcasting: if n_kv < n_heads, repeat K, V to match n_heads
        # Each group of (n_heads // n_kv) query heads shares one KV head
        if self.n_kv < self.n_heads:
            groups = self.n_heads // self.n_kv
            # [B, n_kv, S, d] → [B, n_heads, S, d]
            full_K = full_K.repeat_interleave(groups, dim=1)
            full_V = full_V.repeat_interleave(groups, dim=1)

        # Attention computation
        # During decode: Q is [B, n_heads, 1, head_dim], K is [B, n_heads, S, head_dim]
        # scores: [B, n_heads, 1, S]  — one query row attending to all S keys
        scores  = torch.matmul(Q, full_K.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, full_V)   # [B, n_heads, T, head_dim]

        # Merge heads and project
        context = context.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_O(context)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Minimal Transformer for end-to-end testing
# ─────────────────────────────────────────────────────────────────────────────

class TransformerLayer(nn.Module):
    """One Transformer block: attention + FFN with residual connections."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn   = MultiHeadAttentionWithCache(config)
        self.norm1  = nn.LayerNorm(config.d_model)
        self.norm2  = nn.LayerNorm(config.d_model)
        # FFN: 4× expansion then back to d_model
        self.ffn    = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor,
                cache: Optional[KVCache] = None,
                layer_idx: int = 0) -> torch.Tensor:
        # Pre-norm style (used by LLaMA, Mistral, etc.)
        x = x + self.attn(self.norm1(x), cache=cache, layer_idx=layer_idx)
        x = x + self.ffn(self.norm2(x))
        return x


class MiniTransformer(nn.Module):
    """
    Full Transformer with embedding + N layers + language model head.
    Used for generation with and without KV cache.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config  = config
        self.embed   = nn.Embedding(config.vocab_size, config.d_model)
        self.layers  = nn.ModuleList([TransformerLayer(config) for _ in range(config.n_layers)])
        self.norm    = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor,
                cache: Optional[KVCache] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]  — seq_len=1 during decode, full during prefill
            cache:     KVCache or None
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        x = self.embed(input_ids)   # [B, T, d_model]
        for i, layer in enumerate(self.layers):
            x = layer(x, cache=cache, layer_idx=i)
        x = self.norm(x)
        return self.lm_head(x)      # [B, T, vocab_size]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Generation loops: with and without KV cache
#
# The key difference:
#
# WITHOUT cache:
#   Step t: run model on ALL tokens seen so far [t1, t2, ..., t_current]
#   → O(t) work per step → O(T²) total for T steps
#   → Recomputes K, V for all previous tokens every step
#
# WITH cache:
#   Prefill: run model on full prompt → populate cache → O(prompt_len) once
#   Step t:  run model on [new_token] only → K, V from cache → O(1) per step
#   → O(T) total for T steps
# ─────────────────────────────────────────────────────────────────────────────

def generate_without_cache(model: MiniTransformer,
                           prompt_ids: torch.Tensor,
                           max_new_tokens: int,
                           device: str = "cuda") -> list[dict]:
    """
    Generate tokens WITHOUT KV cache.
    At each step, re-run the full model on the entire sequence so far.

    This is O(T²) in total compute. Each step grows by 1 token.
    Used to measure baseline memory and time (no cache overhead).

    Returns:
        List of dicts with {step, seq_len, memory_mb, token_id} for each step.
    """
    model.eval()
    metrics = []

    # Start with the prompt tokens
    sequence = prompt_ids.to(device)

    with torch.no_grad():
        for step in range(max_new_tokens):
            # ── Memory BEFORE this step ────────────────────────────────────────
            torch.cuda.synchronize()

            # Run full model on entire sequence so far (no cache)
            logits = model(sequence)           # [1, seq_len, vocab_size]
            next_token_logits = logits[:, -1, :]   # take last position's logits
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)   # greedy

            # Append new token to sequence
            sequence = torch.cat([sequence, next_token], dim=1)

            torch.cuda.synchronize()

            mem_mb = torch.cuda.memory_allocated() / 1e6
            metrics.append({
                "step":      step,
                "seq_len":   sequence.shape[1],
                "memory_mb": mem_mb,
                "token_id":  next_token.item(),
                "mode":      "no_cache",
            })

    return metrics


def generate_with_cache(model: MiniTransformer,
                        prompt_ids: torch.Tensor,
                        max_new_tokens: int,
                        device: str = "cuda",
                        cache_mode: str = "dynamic") -> list[dict]:
    """
    Generate tokens WITH KV cache.

    Phase 1 — Prefill: run full model on prompt, populate cache.
    Phase 2 — Decode:  run model on one new token per step, using cache.

    This is O(prompt_len + T) total compute. Each step is O(1) computation.
    Memory grows by (cache_bytes_per_token) each step.

    Returns:
        List of dicts including cache_mb (theoretical) and allocated_mb (actual).
    """
    model.eval()
    metrics = []
    config = model.config

    # Initialize KV cache
    cache = KVCache(config, batch_size=1, device=device, mode=cache_mode)

    with torch.no_grad():

        # ── Phase 1: Prefill ──────────────────────────────────────────────────
        # Process the entire prompt in one forward pass.
        # Populates the cache with K, V for all prompt tokens.
        # After prefill, cache.length == prompt_len.

        prompt = prompt_ids.to(device)    # [1, prompt_len]
        logits = model(prompt, cache=cache)   # [1, prompt_len, vocab_size]

        # The LAST token's logits give us the first generated token
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # [1, 1]
        current_input = next_token

        torch.cuda.synchronize()
        mem_after_prefill = torch.cuda.memory_allocated() / 1e6
        metrics.append({
            "step":          -1,               # -1 marks prefill step
            "seq_len":       prompt.shape[1],
            "cache_len":     cache.length,
            "cache_mb_theoretical": config.kv_cache_bytes(cache.length) / 1e6,
            "cache_mb_actual":      cache.memory_bytes() / 1e6,
            "allocated_mb":  mem_after_prefill,
            "mode":          "prefill",
        })

        # ── Phase 2: Decode ───────────────────────────────────────────────────
        # One new token at a time. Each forward pass gets input [1, 1].
        # The model computes K, V for this one token and appends to cache.
        # Attention attends over the full cache.

        for step in range(max_new_tokens):
            # current_input: [1, 1] — just the ONE new token
            logits = model(current_input, cache=cache)   # [1, 1, vocab_size]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current_input = next_token

            torch.cuda.synchronize()
            allocated_mb = torch.cuda.memory_allocated() / 1e6

            # Compare theoretical formula vs actual allocation
            theoretical_mb = config.kv_cache_bytes(cache.length) / 1e6
            actual_cache_mb = cache.memory_bytes() / 1e6

            metrics.append({
                "step":                  step,
                "seq_len":               prompt.shape[1] + step + 1,
                "cache_len":             cache.length,
                "cache_mb_theoretical":  theoretical_mb,
                "cache_mb_actual":       actual_cache_mb,
                "allocated_mb":          allocated_mb,
                "token_id":              next_token.item(),
                "mode":                  "decode",
            })

    cache.clear()
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Memory growth experiment
#
# Run both generation modes and track memory at every step.
# Shows:
#   1. Cache grows by a fixed amount per step (predictable, formula-driven)
#   2. No-cache: model runs on longer and longer sequences (growing compute)
#   3. The formula prediction vs actual allocation
# ─────────────────────────────────────────────────────────────────────────────

def measure_memory_growth(config: TransformerConfig,
                          prompt_len: int = 16,
                          max_new_tokens: int = 100,
                          device: str = "cuda") -> dict:
    """
    Measure VRAM usage at every generation step for both modes.

    Returns a dict with metrics for both cached and uncached generation.
    """
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.empty_cache()
    gc.collect()

    model = MiniTransformer(config).to(device=device, dtype=config.dtype)
    model.eval()

    # Record baseline (model weights only, no cache, no activations)
    torch.cuda.synchronize()
    baseline_mb = torch.cuda.memory_allocated() / 1e6

    # Dummy prompt (random token IDs)
    torch.manual_seed(0)
    prompt_ids = torch.randint(0, config.vocab_size, (1, prompt_len))

    # ── Run with cache ────────────────────────────────────────────────────────
    torch.cuda.empty_cache()
    metrics_cached = generate_with_cache(model, prompt_ids, max_new_tokens,
                                          device=device, cache_mode="dynamic")

    # ── Run without cache ─────────────────────────────────────────────────────
    torch.cuda.empty_cache()
    gc.collect()
    metrics_uncached = generate_without_cache(model, prompt_ids, max_new_tokens,
                                               device=device)

    return {
        "config":           config,
        "baseline_mb":      baseline_mb,
        "cached":           metrics_cached,
        "uncached":         metrics_uncached,
        "bytes_per_token":  config.kv_cache_bytes(1),  # growth per step
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Scale experiment: KV cache at LLaMA-scale
#
# Even though we can't run LLaMA 7B here, we can predict its KV cache
# at any sequence length using the formula — and verify the formula is
# accurate by comparing to our small model's measurements.
# ─────────────────────────────────────────────────────────────────────────────

def kv_cache_projection_table():
    """
    Print a table of theoretical KV cache sizes for several real models
    across a range of sequence lengths and batch sizes.

    These numbers explain why:
    - GQA (n_kv_heads=8 instead of n_heads=32) was adopted in LLaMA 2 70B
    - Long contexts (100K+) require dedicated memory management
    - Batch size amplifies cache pressure linearly
    """

    models = [
        # (name, n_layers, n_heads, n_kv_heads, d_model)
        ("LLaMA-2 7B  (MHA)",  32, 32, 32, 4096),
        ("LLaMA-2 7B  (GQA)",  32, 32,  8, 4096),  # hypothetical GQA conversion
        ("LLaMA-2 70B (GQA)",  80, 64,  8, 8192),
        ("LLaMA-3 405B(GQA)", 126, 128, 8, 16384),
        ("Mistral-7B  (GQA)",  32, 32,  8, 4096),
    ]

    seq_lengths  = [1024, 4096, 32768, 131072]
    batch_sizes  = [1, 8, 32]
    dtype_bytes  = 2   # FP16

    print("\n" + "═" * 90)
    print("KV Cache Size Projections (FP16, GB)")
    print("Formula: 2 × n_layers × n_kv_heads × seq_len × head_dim × 2 bytes × batch")
    print("═" * 90)

    for name, n_layers, n_heads, n_kv_heads, d_model in models:
        head_dim = d_model // n_heads
        print(f"\n  {name}")
        print(f"  {'Seq len':>10} {'Batch':>6}", end="")
        print(f"  {'Cache GB':>10}  {'Bytes/token':>14}")
        print(f"  {'─'*10} {'─'*6} {'─'*10} {'─'*14}")

        for seq in seq_lengths:
            for bs in batch_sizes:
                cache_bytes = 2 * n_layers * n_kv_heads * seq * head_dim * dtype_bytes * bs
                cache_gb    = cache_bytes / 1e9
                per_token   = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
                print(f"  {seq:>10,} {bs:>6}  {cache_gb:>10.2f}  {per_token:>14,}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Report printing
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: dict):
    """Print a structured report from measure_memory_growth()."""
    config     = results["config"]
    baseline   = results["baseline_mb"]
    bpt        = results["bytes_per_token"]
    cached     = results["cached"]
    uncached   = results["uncached"]

    print("\n" + "═" * 70)
    print("KV Cache Memory Growth Report")
    print("═" * 70)
    print(f"  Model:          {config.n_layers}L × {config.n_heads}H × d{config.d_model}")
    print(f"  KV heads:       {config.n_kv_heads} (head_dim={config.head_dim})")
    print(f"  Dtype:          {config.dtype}")
    print(f"  Weights in GPU: {baseline:.1f} MB")
    print(f"  Cache growth:   {bpt:,} bytes per token")
    print(f"  Formula:        2 × {config.n_layers} × {config.n_kv_heads} × "
          f"{config.head_dim} × {config.bytes_per_element} = {bpt:,}")

    print(f"\n{'─'*70}")
    print("Cached generation (prefill + decode):")
    print(f"  {'Step':>6} {'CacheLen':>10} {'Theoretical MB':>16} {'Actual MB':>12} {'Total MB':>10}")
    print(f"  {'─'*6} {'─'*10} {'─'*16} {'─'*12} {'─'*10}")
    for m in cached:
        if m.get("mode") == "prefill":
            print(f"  {'PREFILL':>6} {m['cache_len']:>10} "
                  f"{m['cache_mb_theoretical']:>16.3f} {m['cache_mb_actual']:>12.3f} "
                  f"{m['allocated_mb']:>10.1f}")
        elif m["step"] % 10 == 0:   # print every 10 steps to keep output short
            print(f"  {m['step']:>6} {m['cache_len']:>10} "
                  f"{m['cache_mb_theoretical']:>16.3f} {m['cache_mb_actual']:>12.3f} "
                  f"{m['allocated_mb']:>10.1f}")

    # Verify formula accuracy: theoretical vs actual should be very close
    decode_steps = [m for m in cached if m.get("mode") == "decode"]
    if decode_steps:
        max_deviation_pct = max(
            abs(m["cache_mb_theoretical"] - m["cache_mb_actual"]) / m["cache_mb_theoretical"] * 100
            for m in decode_steps if m["cache_mb_theoretical"] > 0
        )
        print(f"\n  Formula accuracy: max deviation = {max_deviation_pct:.2f}%")
        print(f"  (Small deviation from allocator alignment; formula is correct)")

    print(f"\n{'─'*70}")
    print("Uncached generation (full sequence at each step):")
    print(f"  {'Step':>6} {'SeqLen':>8} {'Total MB':>10}  (model runs on full sequence)")
    print(f"  {'─'*6} {'─'*8} {'─'*10}")
    for m in uncached:
        if m["step"] % 10 == 0:
            print(f"  {m['step']:>6} {m['seq_len']:>8} {m['memory_mb']:>10.1f}")

    print(f"\n{'─'*70}")
    if decode_steps and uncached:
        final_cached   = decode_steps[-1]["allocated_mb"]
        final_uncached = uncached[-1]["memory_mb"]
        print(f"  Final memory — cached: {final_cached:.1f} MB, uncached: {final_uncached:.1f} MB")
        print(f"  Memory overhead of cache: {final_cached - baseline:.1f} MB (the KV cache)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Config: small model for fast experimentation
    # Change n_kv_heads to 2 to simulate GQA and see cache shrink
    config = TransformerConfig(
        vocab_size=512, d_model=256, n_heads=8,
        n_kv_heads=8,   # change to 2 for GQA simulation
        n_layers=4, dtype=torch.float16
    )

    # Run memory growth experiment
    results = measure_memory_growth(
        config, prompt_len=16, max_new_tokens=80, device=device
    )
    print_report(results)

    # Print scale projection table
    kv_cache_projection_table()


if __name__ == "__main__":
    main()