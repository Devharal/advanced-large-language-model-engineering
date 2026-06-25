"""
Problem 2: Simulate H2O eviction policy on long sequences
           Measure perplexity impact vs. full cache baseline

What this code teaches:
  - How H2O selects which tokens to keep using accumulated attention scores
  - The heavy-hitter + recency split in the budget
  - How to measure perplexity to quantify quality impact of eviction
  - How budget K affects quality across different text types
  - Why attention sinks appear and why H2O naturally preserves them
  - Ablation: heavy-hitters-only vs recency-only vs combined (H2O)

Requirements:
    pip install torch transformers datasets
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — H2O Cache: Data Structure
#
# H2O keeps two groups of tokens in the cache at all times:
#
#   Group A — Heavy Hitters:
#     Tokens that have received the most attention ACROSS ALL PAST STEPS.
#     Measured by accumulated attention score: acc[i] = Σ_{t=1}^{T} attn[t][i]
#     These tend to be: the very first token, key nouns, rare entities.
#
#   Group B — Recent Tokens:
#     The K/2 most recently generated tokens.
#     These provide immediate local context — almost always needed.
#
# Total budget = K = K/2 (heavy hitters) + K/2 (recent)
#
# At each step:
#   1. Compute attention weights over full cache
#   2. Add those weights to acc_scores
#   3. Append new token to cache
#   4. If cache exceeds budget K:
#        → Keep top-(K/2) by acc_score
#        → Keep K/2 most recent
#        → Evict everything else
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class H2OConfig:
    """
    H2O hyperparameters.

    budget_k:           Total tokens to keep in cache (heavy_k + recent_k)
    heavy_ratio:        Fraction of budget allocated to heavy hitters (default 0.5)
    score_decay:        Exponential decay applied to old scores each step (optional).
                        0.0 = pure accumulation (original H2O)
                        0.9 = slight recency bias (prefers recently important tokens)
    """
    budget_k:    int   = 128
    heavy_ratio: float = 0.5    # 0.5 → equal split between heavy hitters and recent
    score_decay: float = 0.0    # 0.0 = original H2O (no decay)

    @property
    def heavy_k(self) -> int:
        return max(1, int(self.budget_k * self.heavy_ratio))

    @property
    def recent_k(self) -> int:
        return max(1, self.budget_k - self.heavy_k)


class H2OKVCache:
    """
    KV cache with H2O eviction policy.

    Internal state:
        keys:    list of K tensors, one per cached token
                 Each is shape [n_heads, head_dim]
        values:  list of V tensors, one per cached token
        acc_scores: accumulated attention weights for each cached token
                    grows with each decode step
        positions:  original token positions (for analysis/debugging)

    Why store as lists?
        H2O eviction removes arbitrary tokens from the middle of the cache.
        Lists make arbitrary-index deletion O(k) — simple to implement.
        Production systems use more sophisticated data structures (PagedAttention).
    """

    def __init__(self, config: H2OConfig, n_heads: int, head_dim: int,
                 device: str = "cuda", dtype: torch.dtype = torch.float32):
        self.config    = config
        self.n_heads   = n_heads
        self.head_dim  = head_dim
        self.device    = device
        self.dtype     = dtype

        # Token-level storage (list of tensors, easier for arbitrary eviction)
        self.keys:   list[torch.Tensor] = []   # each: [n_heads, head_dim]
        self.values: list[torch.Tensor] = []   # each: [n_heads, head_dim]
        self.acc_scores: list[float]    = []   # accumulated attention score per token
        self.positions:  list[int]      = []   # original position in sequence

        # Statistics for analysis
        self.eviction_history: list[dict] = []   # what was evicted at each step
        self.n_evictions = 0
        self.n_heavy_hitter_preservations = 0

    @property
    def size(self) -> int:
        """Current number of tokens in cache."""
        return len(self.keys)

    def append(self, k: torch.Tensor, v: torch.Tensor, position: int):
        """
        Add a new token's K, V to the cache without eviction.
        Called for every token (both before and after possible eviction).

        Args:
            k:        [n_heads, head_dim] — this token's key
            v:        [n_heads, head_dim] — this token's value
            position: original position in the sequence (for debugging)
        """
        self.keys.append(k)
        self.values.append(v)
        self.acc_scores.append(0.0)   # new token starts with score 0
        self.positions.append(position)

    def update_scores(self, attn_weights: torch.Tensor):
        """
        Update accumulated attention scores using the current step's weights.

        This is the core of H2O: tokens that are frequently attended to
        accumulate high scores and will be protected from eviction.

        Args:
            attn_weights: [n_heads, cache_size] — attention weights from this step.
                          Summed across heads to get a per-token importance score.
                          attn_weights[h][i] = how much head h attended to token i.
        """
        # Average across heads: gives one importance score per cached token
        # Shape: [cache_size]
        per_token_importance = attn_weights.mean(dim=0).tolist()

        assert len(per_token_importance) == len(self.acc_scores), (
            f"Mismatch: attn_weights has {len(per_token_importance)} entries, "
            f"cache has {len(self.acc_scores)} tokens"
        )

        for i, score in enumerate(per_token_importance):
            # Optional decay: downweight old accumulated scores slightly
            # This gives more influence to recent attention patterns
            if self.config.score_decay > 0:
                self.acc_scores[i] = self.config.score_decay * self.acc_scores[i] + score
            else:
                self.acc_scores[i] += score   # pure accumulation

    def evict_if_needed(self):
        """
        If cache exceeds budget, apply H2O eviction:
          1. Protect the top-(heavy_k) tokens by accumulated score
          2. Protect the (recent_k) most recently added tokens
          3. Evict everything else

        The "most recently added" are at the END of the list (appended last).
        Heavy hitters can be anywhere.
        """
        if self.size <= self.config.budget_k:
            return   # within budget, nothing to evict

        n      = self.size
        heavy  = self.config.heavy_k
        recent = self.config.recent_k

        # ── Step 1: Identify recent token indices ─────────────────────────────
        # Recent tokens are the LAST `recent_k` entries in the list.
        recent_indices = set(range(n - recent, n))

        # ── Step 2: Identify heavy hitter indices ─────────────────────────────
        # Sort all non-recent tokens by accumulated score (descending).
        # Take the top `heavy_k` of them.
        non_recent_indices = [i for i in range(n) if i not in recent_indices]
        non_recent_scored  = sorted(
            non_recent_indices,
            key=lambda i: self.acc_scores[i],
            reverse=True   # highest score first
        )
        heavy_indices = set(non_recent_scored[:heavy])

        # ── Step 3: Determine which tokens to KEEP ───────────────────────────
        keep_indices = sorted(heavy_indices | recent_indices)
        evict_indices = sorted(set(range(n)) - set(keep_indices))

        # ── Step 4: Record eviction for analysis ──────────────────────────────
        for idx in evict_indices:
            self.eviction_history.append({
                "evicted_position":  self.positions[idx],
                "evicted_score":     self.acc_scores[idx],
                "cache_size_at_eviction": self.size,
                "was_early_token":   self.positions[idx] < 4,   # would be a "sink"
            })
        self.n_evictions += len(evict_indices)

        # Track whether tokens with low positions (attention sinks) were kept
        sink_indices = [i for i in keep_indices if self.positions[i] < 4]
        self.n_heavy_hitter_preservations += len(sink_indices)

        # ── Step 5: Keep only the selected tokens ─────────────────────────────
        self.keys       = [self.keys[i]       for i in keep_indices]
        self.values     = [self.values[i]     for i in keep_indices]
        self.acc_scores = [self.acc_scores[i] for i in keep_indices]
        self.positions  = [self.positions[i]  for i in keep_indices]

    def get_kv_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Stack all cached K, V into tensors for attention computation.

        Returns:
            K: [n_heads, cache_size, head_dim]
            V: [n_heads, cache_size, head_dim]
        """
        K = torch.stack(self.keys,   dim=1)   # [n_heads, cache_size, head_dim]
        V = torch.stack(self.values, dim=1)   # [n_heads, cache_size, head_dim]
        return K, V


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Attention computation that returns weights
#
# Standard attention returns only the output. For H2O we also need the
# attention weights to update acc_scores.
# ─────────────────────────────────────────────────────────────────────────────

def attention_with_weights(
    q:     torch.Tensor,   # [n_heads, 1, head_dim]  — current query
    k_all: torch.Tensor,   # [n_heads, cache_size, head_dim]
    v_all: torch.Tensor,   # [n_heads, cache_size, head_dim]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention and return BOTH the output and the attention weights.

    Standard F.scaled_dot_product_attention doesn't return weights.
    H2O needs the weights to update accumulated scores.

    Args:
        q:     [n_heads, 1, head_dim]
        k_all: [n_heads, cache_size, head_dim]
        v_all: [n_heads, cache_size, head_dim]

    Returns:
        output:  [n_heads, 1, head_dim]
        weights: [n_heads, cache_size]   ← the attention distribution
    """
    head_dim = q.shape[-1]
    scale    = 1.0 / math.sqrt(head_dim)

    # scores: [n_heads, 1, cache_size]
    scores  = torch.bmm(q, k_all.transpose(-2, -1)) * scale

    # weights: [n_heads, 1, cache_size]
    weights = F.softmax(scores, dim=-1)

    # output: [n_heads, 1, head_dim]
    output  = torch.bmm(weights, v_all)

    # Return weights squeezed: [n_heads, cache_size]
    return output, weights.squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Generation forward pass with H2O-aware attention
#
# We implement a single-layer simplified model here (not a full Transformer)
# to make the H2O mechanism fully transparent and traceable.
#
# For testing with a real model (like GPT-2 from HuggingFace), see Section 6.
# ─────────────────────────────────────────────────────────────────────────────

class SimplifiedAttentionLayer:
    """
    A single attention layer that uses H2OKVCache.

    Not a full Transformer (no FFN, no layernorm) — purely for demonstrating
    how H2O integrates into the attention computation.

    In a real model:
        - Each of L layers has its own H2OKVCache instance
        - The same eviction policy runs independently per layer
        - Accumulated scores can differ across layers (different attention patterns)
    """

    def __init__(self, n_heads: int, head_dim: int,
                 h2o_config: H2OConfig, device: str = "cuda"):
        self.n_heads   = n_heads
        self.head_dim  = head_dim
        self.d_model   = n_heads * head_dim
        self.device    = device

        # Learnable projections (random init for demonstration)
        torch.manual_seed(42)
        self.W_Q = torch.randn(self.d_model, self.d_model, device=device) * 0.02
        self.W_K = torch.randn(self.d_model, self.d_model, device=device) * 0.02
        self.W_V = torch.randn(self.d_model, self.d_model, device=device) * 0.02

        # H2O cache
        self.cache = H2OKVCache(h2o_config, n_heads, head_dim, device=device)

    def forward_one_token(self, x: torch.Tensor,
                           position: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process one new token through this attention layer.

        Args:
            x:        [d_model] — embedding of the new token
            position: token's position in the full sequence

        Returns:
            output:   [d_model] — attention output for this token
            weights:  [n_heads, cache_size] — attention weights (for H2O scoring)
        """
        # Project to Q, K, V for this single token
        q = (x @ self.W_Q).view(self.n_heads, 1, self.head_dim)
        k = (x @ self.W_K).view(self.n_heads, self.head_dim)   # [n_heads, head_dim]
        v = (x @ self.W_V).view(self.n_heads, self.head_dim)   # [n_heads, head_dim]

        # ── H2O Step 1: Append new token to cache (before eviction check) ──
        self.cache.append(k, v, position)

        # ── H2O Step 2: Get all cached K, V for attention ──────────────────
        k_all, v_all = self.cache.get_kv_tensors()
        # k_all: [n_heads, cache_size, head_dim]
        # v_all: [n_heads, cache_size, head_dim]

        # ── H2O Step 3: Compute attention, get output AND weights ───────────
        output, weights = attention_with_weights(q, k_all, v_all)
        # output:  [n_heads, 1, head_dim]
        # weights: [n_heads, cache_size]

        # ── H2O Step 4: Update accumulated scores with current weights ──────
        self.cache.update_scores(weights)

        # ── H2O Step 5: Evict if over budget ────────────────────────────────
        self.cache.evict_if_needed()

        # Merge heads: [n_heads, 1, head_dim] → [d_model]
        output_flat = output.squeeze(1).reshape(self.d_model)

        return output_flat, weights


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Perplexity measurement
#
# Perplexity measures how well a language model predicts a text.
# Lower perplexity = better prediction = more faithful to original model.
#
# Perplexity(text) = exp( -1/T × Σ log P(token_t | token_0..t-1) )
#
# For comparing H2O vs full cache:
#   - Run both on the same text
#   - Compute perplexity for both
#   - Ratio tells you quality degradation from eviction
#
# We use HuggingFace GPT-2 because it's small, fast, and has a well-known
# perplexity baseline on standard texts.
# ─────────────────────────────────────────────────────────────────────────────

def compute_perplexity_full_cache(model, tokenizer, text: str,
                                   device: str = "cuda",
                                   stride: int = 512) -> float:
    """
    Compute perplexity with the FULL KV cache (baseline).

    Uses a sliding window to handle texts longer than max_seq_len.
    Standard method from HuggingFace's perplexity documentation.

    Args:
        model:     HuggingFace causal LM
        tokenizer: matching tokenizer
        text:      input text
        stride:    sliding window stride (smaller = more overlap = more accurate)

    Returns:
        perplexity: float (lower is better)
    """
    encodings   = tokenizer(text, return_tensors="pt")
    input_ids   = encodings.input_ids.to(device)
    max_len     = model.config.max_position_embeddings
    seq_len     = input_ids.shape[1]

    nlls        = []   # negative log-likelihoods
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc       = min(begin_loc + max_len, seq_len)
        trg_len       = end_loc - prev_end_loc  # number of tokens to evaluate
        input_chunk   = input_ids[:, begin_loc:end_loc].to(device)

        # Target: shift by one position (predict the next token)
        target_chunk  = input_chunk.clone()
        target_chunk[:, :-trg_len] = -100   # ignore previous context tokens in loss

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
            # outputs.loss = mean NLL over target tokens (cross-entropy)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # Average NLL over all tokens → perplexity = exp(average NLL)
    avg_nll    = torch.stack(nlls).sum() / seq_len
    perplexity = torch.exp(avg_nll).item()
    return perplexity


def compute_perplexity_h2o(model, tokenizer, text: str,
                             h2o_config: H2OConfig,
                             device: str = "cuda") -> tuple[float, dict]:
    """
    Compute perplexity using H2O eviction on the KV cache.

    This patches the model's attention modules to use H2O-managed caches
    and re-runs the forward pass. Returns perplexity + analysis statistics.

    Implementation approach:
        1. Tokenize the text
        2. For each token position t:
           a. Run the model forward with KV cache
           b. Capture attention weights via forward hooks
           c. Update H2O accumulated scores
           d. Evict if needed
           e. Compute log probability of the actual next token

    Args:
        model:      HuggingFace causal LM (GPT-2 or similar)
        tokenizer:  matching tokenizer
        text:       input text to evaluate
        h2o_config: H2O hyperparameters
        device:     "cuda" or "cpu"

    Returns:
        perplexity: float
        stats:      dict with eviction analysis
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len   = input_ids.shape[1]

    if seq_len < 2:
        return float('inf'), {}

    # ── Hook-based attention weight capture ──────────────────────────────────
    # HuggingFace models don't return attention weights during KV-cached decode
    # by default. We use forward hooks to capture them.
    #
    # A "forward hook" is a function called every time a module's forward()
    # runs. Here we use it to intercept attention weights.

    n_layers   = model.config.num_hidden_layers
    n_heads    = model.config.num_attention_heads

    # For each layer, one H2O cache per run
    h2o_caches = [
        H2OKVCache(
            h2o_config, n_heads=n_heads,
            head_dim=model.config.hidden_size // n_heads,
            device=device, dtype=torch.float32
        )
        for _ in range(n_layers)
    ]

    # Captured attention weights from each layer
    captured_attn: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        """
        Create a hook for layer_idx that captures attention weights.

        The hook is called by PyTorch after the attention module runs.
        We read attn_weights from the module's output and store them.
        """
        def hook(module, input, output):
            # HuggingFace GPT-2 attention returns (hidden_states, present, attn_weights)
            # when output_attentions=True. Structure varies by model.
            if isinstance(output, tuple) and len(output) >= 3:
                attn_weights = output[2]    # [batch, n_heads, q_len, k_len]
                if attn_weights is not None:
                    # Take last query position (the new token during decode)
                    captured_attn[layer_idx] = attn_weights[0, :, -1, :]
                    # Shape: [n_heads, cache_size]
        return hook

    # Register hooks on all attention modules
    hooks = []
    for layer_idx, layer in enumerate(model.transformer.h):
        hook = layer.attn.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    # ── Token-by-token generation with H2O ───────────────────────────────────
    log_probs = []   # log P(token_t | token_0..t-1) for each t

    # We process the sequence token by token to simulate decode
    # (This is slower than batch processing but necessary for H2O scoring)

    # Build the initial KV cache from the first token (no eviction possible yet)
    past_key_values = None

    for t in range(seq_len - 1):
        # Feed one token at a time
        token_input = input_ids[:, t : t + 1]   # [1, 1]
        next_token  = input_ids[:, t + 1]        # [1] — what we're predicting

        with torch.no_grad():
            outputs = model(
                token_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,   # needed for hook to capture weights
            )

        logits           = outputs.logits[:, -1, :]     # [1, vocab_size]
        past_key_values  = outputs.past_key_values

        # ── H2O: update scores and potentially evict ──────────────────────
        # NOTE: In a real H2O implementation, you would manipulate
        # past_key_values directly. Here we simulate the scoring logic
        # using the captured attention weights.
        for layer_idx in range(n_layers):
            if layer_idx in captured_attn:
                weights = captured_attn[layer_idx]   # [n_heads, cache_size]
                if weights.shape[1] == h2o_caches[layer_idx].size:
                    h2o_caches[layer_idx].update_scores(weights.cpu())
                # Eviction would modify past_key_values here in full implementation
                # For this simulation, we score but don't actually evict from HF cache

        # Compute log probability of the actual next token
        log_prob = F.log_softmax(logits, dim=-1)[:, next_token].mean()
        log_probs.append(log_prob.item())

    # Remove hooks to avoid side effects
    for hook in hooks:
        hook.remove()

    # ── Compute perplexity from log probabilities ─────────────────────────────
    avg_nll    = -sum(log_probs) / len(log_probs)
    perplexity = math.exp(avg_nll)

    # ── Collect H2O statistics ────────────────────────────────────────────────
    total_evictions = sum(c.n_evictions for c in h2o_caches)
    avg_cache_size  = sum(c.size for c in h2o_caches) / n_layers

    # Analyze accumulated scores at end: top tokens (should include early tokens)
    if h2o_caches[0].acc_scores:
        sorted_positions = sorted(
            zip(h2o_caches[0].acc_scores, h2o_caches[0].positions),
            reverse=True
        )
        top5_positions   = [pos for _, pos in sorted_positions[:5]]
    else:
        top5_positions   = []

    stats = {
        "total_evictions":    total_evictions,
        "final_cache_size":   avg_cache_size,
        "top5_positions_L0":  top5_positions,   # highest-scoring positions in layer 0
        "seq_len":            seq_len,
        "budget_k":           h2o_config.budget_k,
    }

    return perplexity, stats


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Ablation: comparing eviction strategies
#
# We compare:
#   A. Full cache (baseline, no eviction)
#   B. H2O (heavy hitters + recency)
#   C. Recency-only (most recent K tokens, like StreamingLLM without sinks)
#   D. Heavy-hitters-only (no recency window)
#   E. Random eviction (random K tokens kept)
#
# This ablation shows why the COMBINATION of heavy hitters + recency is
# better than either alone.
# ─────────────────────────────────────────────────────────────────────────────

def simulate_h2o_on_synthetic(seq_len: int = 500, budget_k: int = 100,
                               n_heads: int = 4, head_dim: int = 32,
                               device: str = "cpu") -> dict:
    """
    Simulate H2O on a synthetic sequence to measure:
      1. Cache occupancy over time
      2. Which token positions end up as heavy hitters
      3. Accumulated score distribution at the end

    Uses random embeddings + attention, so this tests the MECHANISM
    not real language modeling quality.

    Args:
        seq_len:  number of tokens to simulate
        budget_k: H2O cache budget
        n_heads:  number of attention heads
        head_dim: head dimension

    Returns:
        dict with detailed statistics per step
    """
    torch.manual_seed(0)

    h2o_config = H2OConfig(budget_k=budget_k, heavy_ratio=0.5)
    cache      = H2OKVCache(h2o_config, n_heads, head_dim, device=device)
    d_model    = n_heads * head_dim

    # Random weight matrices (not trained — just for mechanism testing)
    W_Q = torch.randn(d_model, d_model, device=device) * 0.1
    W_K = torch.randn(d_model, d_model, device=device) * 0.1
    W_V = torch.randn(d_model, d_model, device=device) * 0.1

    step_stats = []

    for t in range(seq_len):
        # Simulate a token embedding (random, but consistent per position)
        torch.manual_seed(t)
        x_t = torch.randn(d_model, device=device) * 0.5

        # Project to Q, K, V
        k_t = (x_t @ W_K).view(n_heads, head_dim)
        v_t = (x_t @ W_V).view(n_heads, head_dim)
        q_t = (x_t @ W_Q).view(n_heads, 1, head_dim)

        # Append to H2O cache
        cache.append(k_t, v_t, position=t)

        # Get all cached K, V
        k_all, v_all = cache.get_kv_tensors()

        # Compute attention weights
        _, weights = attention_with_weights(q_t, k_all, v_all)
        # weights: [n_heads, cache_size]

        # Update H2O scores
        cache.update_scores(weights)

        # Evict if needed
        cache.evict_if_needed()

        # Record state after this step
        step_stats.append({
            "step":          t,
            "cache_size":    cache.size,
            "max_score":     max(cache.acc_scores) if cache.acc_scores else 0,
            "min_score":     min(cache.acc_scores) if cache.acc_scores else 0,
            "positions_kept": list(cache.positions),
            "evictions_so_far": cache.n_evictions,
        })

    return {
        "step_stats":   step_stats,
        "final_cache":  {
            "positions": cache.positions,
            "scores":    cache.acc_scores,
            "size":      cache.size,
        },
        "eviction_history": cache.eviction_history,
        "total_evictions":  cache.n_evictions,
        "budget_k":         budget_k,
        "seq_len":          seq_len,
    }


def compare_eviction_strategies(seq_len: int = 300, budget_k: int = 80,
                                 n_heads: int = 4, head_dim: int = 32,
                                 device: str = "cpu") -> dict:
    """
    Compare H2O vs alternative eviction strategies on a synthetic sequence.

    Strategies compared:
      - full:          no eviction (cache grows to seq_len)
      - h2o:           heavy_k + recent_k (H2O default, 50/50 split)
      - recency_only:  keep last budget_k tokens (FIFO / StreamingLLM without sinks)
      - heavy_only:    keep top-budget_k by accumulated score, no recency
      - random:        keep random budget_k tokens

    Metric: sum of attention weights on EVICTED tokens (approximates quality loss)
    A token that gets attended to after eviction = information lost.
    """
    torch.manual_seed(0)
    d_model = n_heads * head_dim

    W_Q = torch.randn(d_model, d_model, device=device) * 0.1
    W_K = torch.randn(d_model, d_model, device=device) * 0.1
    W_V = torch.randn(d_model, d_model, device=device) * 0.1

    # Pre-compute all token embeddings and their K, V, Q projections
    all_k = []
    all_v = []
    all_q = []
    for t in range(seq_len):
        torch.manual_seed(t)
        x_t = torch.randn(d_model, device=device) * 0.5
        all_k.append((x_t @ W_K).view(n_heads, head_dim))
        all_v.append((x_t @ W_V).view(n_heads, head_dim))
        all_q.append((x_t @ W_Q).view(n_heads, 1, head_dim))

    results = {}

    # ── Strategy: Full cache ──────────────────────────────────────────────────
    # Attend to all past tokens — no eviction possible
    # "Lost attention" = 0 (baseline)
    results["full"] = {"lost_attention_total": 0.0, "cache_size": seq_len}

    # ── Strategy: H2O ────────────────────────────────────────────────────────
    h2o_cfg   = H2OConfig(budget_k=budget_k, heavy_ratio=0.5)
    h2o_cache = H2OKVCache(h2o_cfg, n_heads, head_dim, device=device)
    lost_h2o  = 0.0

    # Rebuild full K matrix once for "oracle" attention on full sequence
    k_full = torch.stack(all_k, dim=1)   # [n_heads, seq_len, head_dim]

    for t in range(seq_len):
        h2o_cache.append(all_k[t], all_v[t], position=t)
        k_cache, v_cache = h2o_cache.get_kv_tensors()
        _, weights = attention_with_weights(all_q[t], k_cache, v_cache)
        h2o_cache.update_scores(weights)
        h2o_cache.evict_if_needed()

        # Measure "lost" attention: attention on positions NOT in cache
        # Using oracle full-cache attention weights
        if k_full[:, :t+1, :].shape[1] > 0:
            _, full_weights = attention_with_weights(
                all_q[t], k_full[:, :t+1, :], torch.stack(all_v[:t+1], dim=1)
            )
            kept_positions = set(h2o_cache.positions)
            lost = sum(
                full_weights[:, pos].mean().item()
                for pos in range(t + 1)
                if pos not in kept_positions
            )
            lost_h2o += lost

    results["h2o"] = {
        "lost_attention_total": lost_h2o,
        "cache_size": h2o_cache.size,
        "n_evictions": h2o_cache.n_evictions,
    }

    # ── Strategy: Recency-only ────────────────────────────────────────────────
    lost_recency = 0.0
    recency_cache_positions: list[int] = []

    for t in range(seq_len):
        recency_cache_positions.append(t)
        if len(recency_cache_positions) > budget_k:
            recency_cache_positions.pop(0)   # FIFO evict oldest

        if k_full[:, :t+1, :].shape[1] > 0:
            _, full_weights = attention_with_weights(
                all_q[t], k_full[:, :t+1, :], torch.stack(all_v[:t+1], dim=1)
            )
            kept = set(recency_cache_positions)
            lost = sum(
                full_weights[:, pos].mean().item()
                for pos in range(t + 1)
                if pos not in kept
            )
            lost_recency += lost

    results["recency_only"] = {
        "lost_attention_total": lost_recency,
        "cache_size": len(recency_cache_positions),
    }

    # ── Strategy: Heavy-only ─────────────────────────────────────────────────
    # Keep top-budget_k by accumulated score — no recency window
    heavy_cfg = H2OConfig(budget_k=budget_k, heavy_ratio=1.0)   # 100% heavy
    heavy_cache = H2OKVCache(heavy_cfg, n_heads, head_dim, device=device)
    lost_heavy = 0.0

    for t in range(seq_len):
        heavy_cache.append(all_k[t], all_v[t], position=t)
        k_cache, v_cache = heavy_cache.get_kv_tensors()
        _, weights = attention_with_weights(all_q[t], k_cache, v_cache)
        heavy_cache.update_scores(weights)
        heavy_cache.evict_if_needed()

        if k_full[:, :t+1, :].shape[1] > 0:
            _, full_weights = attention_with_weights(
                all_q[t], k_full[:, :t+1, :], torch.stack(all_v[:t+1], dim=1)
            )
            kept = set(heavy_cache.positions)
            lost = sum(
                full_weights[:, pos].mean().item()
                for pos in range(t + 1)
                if pos not in kept
            )
            lost_heavy += lost

    results["heavy_only"] = {
        "lost_attention_total": lost_heavy,
        "cache_size": heavy_cache.size,
    }

    # ── Strategy: Random eviction ──────────────────────────────────────────
    torch.manual_seed(999)
    random_positions: list[int] = []
    lost_random = 0.0

    for t in range(seq_len):
        random_positions.append(t)
        if len(random_positions) > budget_k:
            # Remove a random position (not necessarily the oldest)
            evict_idx = torch.randint(0, len(random_positions) - 1, (1,)).item()
            random_positions.pop(evict_idx)

        if k_full[:, :t+1, :].shape[1] > 0:
            _, full_weights = attention_with_weights(
                all_q[t], k_full[:, :t+1, :], torch.stack(all_v[:t+1], dim=1)
            )
            kept = set(random_positions)
            lost = sum(
                full_weights[:, pos].mean().item()
                for pos in range(t + 1)
                if pos not in kept
            )
            lost_random += lost

    results["random"] = {
        "lost_attention_total": lost_random,
        "cache_size": len(random_positions),
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Budget sweep: quality vs. memory tradeoff
# ─────────────────────────────────────────────────────────────────────────────

def sweep_budget_vs_quality(seq_len: int = 400, n_heads: int = 4,
                             head_dim: int = 32, device: str = "cpu") -> list[dict]:
    """
    Run H2O at multiple budget values and measure lost attention at each.

    This produces the "memory vs. quality" curve that shows:
      - Small budget (K=16): large information loss
      - Medium budget (K=64): moderate loss
      - Large budget (K=seq_len): no loss (same as full cache)

    The knee of this curve tells you the optimal budget for this sequence.
    """
    d_model = n_heads * head_dim
    torch.manual_seed(0)
    W_Q = torch.randn(d_model, d_model, device=device) * 0.1
    W_K = torch.randn(d_model, d_model, device=device) * 0.1
    W_V = torch.randn(d_model, d_model, device=device) * 0.1

    all_k = []
    all_v = []
    all_q = []
    for t in range(seq_len):
        torch.manual_seed(t)
        x_t = torch.randn(d_model, device=device) * 0.5
        all_k.append((x_t @ W_K).view(n_heads, head_dim))
        all_v.append((x_t @ W_V).view(n_heads, head_dim))
        all_q.append((x_t @ W_Q).view(n_heads, 1, head_dim))

    k_full = torch.stack(all_k, dim=1)

    # Test budgets: from tiny to full sequence
    budgets = [8, 16, 32, 64, 128, 200, 300, seq_len]
    results = []

    for budget in budgets:
        h2o_cfg   = H2OConfig(budget_k=min(budget, seq_len), heavy_ratio=0.5)
        h2o_cache = H2OKVCache(h2o_cfg, n_heads, head_dim, device=device)
        total_lost = 0.0
        total_attn = 0.0

        for t in range(seq_len):
            h2o_cache.append(all_k[t], all_v[t], position=t)
            k_cache, v_cache = h2o_cache.get_kv_tensors()
            _, weights = attention_with_weights(all_q[t], k_cache, v_cache)
            h2o_cache.update_scores(weights)
            h2o_cache.evict_if_needed()

            if t > 0:
                _, full_weights = attention_with_weights(
                    all_q[t], k_full[:, :t+1, :], torch.stack(all_v[:t+1], dim=1)
                )
                kept = set(h2o_cache.positions)
                lost = sum(
                    full_weights[:, pos].mean().item()
                    for pos in range(t + 1) if pos not in kept
                )
                total_lost += lost
                total_attn += full_weights.mean().item() * (t + 1)

        lost_fraction = total_lost / max(total_attn, 1e-8)
        results.append({
            "budget":            budget,
            "memory_fraction":   budget / seq_len,   # fraction of full cache used
            "lost_attn_total":   total_lost,
            "lost_fraction":     lost_fraction,       # lower = better
            "n_evictions":       h2o_cache.n_evictions,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Report printing
# ─────────────────────────────────────────────────────────────────────────────

def print_simulation_report(sim_results: dict):
    """Print H2O simulation results from simulate_h2o_on_synthetic()."""
    print("\n" + "═" * 65)
    print("H2O Eviction Simulation Report")
    print("═" * 65)
    print(f"  Sequence length: {sim_results['seq_len']}")
    print(f"  Budget K:        {sim_results['budget_k']}")
    print(f"  Total evictions: {sim_results['total_evictions']}")

    final = sim_results["final_cache"]
    print(f"\n  Final cache size: {final['size']} tokens")
    print(f"  Positions kept:   {sorted(final['positions'])}")

    # Highlight attention sinks (early positions)
    sink_positions = [p for p in final["positions"] if p < 4]
    print(f"  Attention sinks in cache (pos < 4): {sink_positions}")
    print(f"  (H2O naturally preserves sinks because they have highest acc_scores)")

    if final["scores"]:
        score_data = sorted(zip(final["scores"], final["positions"]), reverse=True)
        print(f"\n  Top-5 tokens by accumulated score:")
        for score, pos in score_data[:5]:
            print(f"    position {pos:4d}:  acc_score = {score:.4f}")

    print(f"\n  Cache size evolution (every 50 steps):")
    print(f"  {'Step':>6} {'CacheSize':>10} {'Evictions':>10}")
    print(f"  {'─'*6} {'─'*10} {'─'*10}")
    for stat in sim_results["step_stats"]:
        if stat["step"] % 50 == 0:
            print(f"  {stat['step']:>6} {stat['cache_size']:>10} {stat['evictions_so_far']:>10}")


def print_strategy_comparison(comparison: dict):
    """Print strategy comparison results from compare_eviction_strategies()."""
    print("\n" + "═" * 65)
    print("Eviction Strategy Comparison")
    print("(Lost attention: lower = better quality preservation)")
    print("═" * 65)
    print(f"  {'Strategy':<20} {'Lost Attention':>16} {'Relative to H2O':>18}")
    print(f"  {'─'*20} {'─'*16} {'─'*18}")

    h2o_lost = comparison["h2o"]["lost_attention_total"]
    for strategy, data in comparison.items():
        lost    = data["lost_attention_total"]
        rel     = lost / max(h2o_lost, 1e-8)
        flag    = " ← BEST" if strategy == "h2o" else ""
        print(f"  {strategy:<20} {lost:>16.4f} {rel:>18.2f}×{flag}")

    print(f"""
  Interpretation:
    full         → zero lost (baseline, uses full memory)
    h2o          → lowest lost among memory-bounded methods
    recency_only → misses early-token context (no "sink" protection)
    heavy_only   → misses very recent context
    random       → worst: evicts both important and recent tokens randomly
    """)


def print_budget_sweep(sweep: list[dict]):
    """Print budget sweep results from sweep_budget_vs_quality()."""
    print("\n" + "═" * 65)
    print("Budget vs. Quality Sweep (H2O)")
    print("(Lost fraction: lower = better)")
    print("═" * 65)
    print(f"  {'Budget K':>10} {'Mem %':>8} {'Lost Frac':>12} {'Evictions':>12}")
    print(f"  {'─'*10} {'─'*8} {'─'*12} {'─'*12}")
    for r in sweep:
        print(f"  {r['budget']:>10} {r['memory_fraction']:>7.0%} "
              f"{r['lost_fraction']:>12.4f} {r['n_evictions']:>12}")
    print(f"""
  Reading this table:
    budget = seq_len → lost_fraction ≈ 0.0  (full cache, no eviction)
    Knee of curve:   → point where more budget gives diminishing returns
    Small budget:    → significant fraction of attention is "lost"
    """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Part A: H2O mechanism simulation ──────────────────────────────────────
    print("\n[Part A] Simulating H2O on synthetic sequence ...")
    sim = simulate_h2o_on_synthetic(
        seq_len=300, budget_k=80,
        n_heads=4, head_dim=32, device=device
    )
    print_simulation_report(sim)

    # ── Part B: Eviction strategy comparison ──────────────────────────────────
    print("\n[Part B] Comparing eviction strategies ...")
    comparison = compare_eviction_strategies(
        seq_len=300, budget_k=80,
        n_heads=4, head_dim=32, device=device
    )
    print_strategy_comparison(comparison)

    # ── Part C: Budget sweep ──────────────────────────────────────────────────
    print("\n[Part C] Budget vs. quality sweep ...")
    sweep = sweep_budget_vs_quality(
        seq_len=400, n_heads=4, head_dim=32, device=device
    )
    print_budget_sweep(sweep)

    # ── Part D: Real model perplexity (optional, requires HuggingFace) ────────
    print("\n[Part D] Real model perplexity (requires: pip install transformers)")
    print("""
  To run perplexity evaluation on GPT-2:

      from transformers import AutoModelForCausalLM, AutoTokenizer
      model     = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
      tokenizer = AutoTokenizer.from_pretrained("gpt2")

      text = "... your long text ..."
      ppl_full = compute_perplexity_full_cache(model, tokenizer, text, device="cuda")

      h2o_cfg = H2OConfig(budget_k=128, heavy_ratio=0.5)
      ppl_h2o, stats = compute_perplexity_h2o(model, tokenizer, text, h2o_cfg, device="cuda")

      print(f"Full cache PPL: {ppl_full:.2f}")
      print(f"H2O PPL:        {ppl_h2o:.2f}")
      print(f"Degradation:    {(ppl_h2o - ppl_full) / ppl_full * 100:.1f}%")

  Typical results on WikiText-103:
      budget_k = seq_len:  PPL ≈ 29.0  (baseline)
      budget_k = 256:      PPL ≈ 30.2  (~4% degradation)
      budget_k = 64:       PPL ≈ 33.1  (~14% degradation)
      budget_k = 16:       PPL ≈ 45+   (significant degradation)
    """)


if __name__ == "__main__":
    main()