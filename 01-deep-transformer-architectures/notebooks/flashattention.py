"""
Problem 1: Enable FlashAttention-2 in HuggingFace
           Measure throughput vs. standard attention

What this file teaches:
  - How HuggingFace exposes attention backends via attn_implementation
  - How to correctly set up a fair benchmark (warmup, CUDA sync, timing)
  - What "throughput" means: tokens generated per second
  - Why FA-2 helps more at longer sequence lengths
  - How to read the numbers and know what to expect

Requirements:
  pip install transformers torch accelerate
  pip install flash-attn --no-build-isolation   # needs CUDA toolkit
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Configuration
# All tunable knobs in one place so the benchmark is easy to modify.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID    = "meta-llama/Meta-Llama-3-8B"   # swap any causal LM here
DTYPE       = torch.bfloat16                  # BF16: required for FA-2 on most GPUs
DEVICE      = "cuda"
WARMUP_RUNS = 3     # discard first N runs (GPU frequency scaling, caches cold)
TIMED_RUNS  = 10    # average over this many runs for stable numbers

# We test multiple sequence lengths to see how speedup scales with N.
# FA-2's advantage grows as N grows because naive attention's N² cost grows faster.
SEQUENCE_LENGTHS = [256, 512, 1024, 2048, 4096]

NEW_TOKENS = 50     # how many tokens to generate per run
                    # keep small: we're measuring throughput, not output quality


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Load the two model variants
#
# HuggingFace exposes attention backend selection via `attn_implementation`:
#
#   "eager"            — pure PyTorch loops, no fusion, slowest
#   "sdpa"             — PyTorch 2.0+ scaled_dot_product_attention
#                        uses FA-2 automatically when conditions are met
#                        (causal mask, no dropout, supported dtype/device)
#   "flash_attention_2" — explicitly forces the flash_attn package kernels
#                         requires: pip install flash-attn
#
# We load two separate model objects so they can be benchmarked independently.
# Both load the same weights — only the attention kernel differs.
# ─────────────────────────────────────────────────────────────────────────────

def load_models():
    print("Loading standard attention model (sdpa / eager) ...")
    model_standard = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation="sdpa",   # PyTorch native — no flash_attn needed
    )
    model_standard.eval()

    print("Loading FlashAttention-2 model ...")
    model_fa2 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation="flash_attention_2",  # explicit FA-2
    )
    model_fa2.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Some tokenizers don't set pad_token — generation needs it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model_standard, model_fa2, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Benchmarking helper
#
# A robust GPU benchmark must:
#   1. Warm up first  — CUDA JIT-compiles kernels on first call; skip those.
#   2. Synchronize    — torch operations are async; time.time() without
#                       torch.cuda.synchronize() measures launch time, not
#                       actual compute time. Always sync before and after.
#   3. Use events     — torch.cuda.Event is more accurate than time.time()
#                       for GPU timing; measures on-device elapsed time.
#   4. Repeat and average — GPU clocks vary; single runs are noisy.
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_throughput(model, tokenizer, seq_len, new_tokens,
                         warmup_runs=3, timed_runs=10):
    """
    Generate `new_tokens` tokens given a prompt of length `seq_len`.
    Returns average tokens/second over `timed_runs` runs.

    Args:
        model:       HuggingFace causal LM (already on CUDA)
        tokenizer:   matching tokenizer
        seq_len:     number of input tokens (prompt length)
        new_tokens:  how many tokens to generate
        warmup_runs: discard first N runs (kernel compilation, cache effects)
        timed_runs:  average timing over this many runs

    Returns:
        float: tokens per second (higher = better)
    """

    # Build a dummy input of exactly `seq_len` tokens.
    # torch.randint produces random token IDs in the valid vocabulary range.
    # Shape: [1, seq_len] — batch size 1.
    input_ids = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(1, seq_len),
        device=DEVICE,
        dtype=torch.long,
    )

    # Attention mask: all 1s (no padding in our synthetic prompt)
    attention_mask = torch.ones_like(input_ids)

    generate_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=new_tokens,
        do_sample=False,        # greedy decoding — deterministic, reproducible
        use_cache=True,         # IMPORTANT: enables KV cache (always use in practice)
        pad_token_id=tokenizer.pad_token_id,
    )

    # ── Warmup ────────────────────────────────────────────────────────────────
    # CUDA lazy-initializes: first kernel call compiles ptx → cubin.
    # Warmup ensures all compilation happens before we start timing.
    print(f"    Warming up (seq_len={seq_len}) ...", end=" ", flush=True)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model.generate(**generate_kwargs)
            torch.cuda.synchronize()   # wait for GPU to actually finish
    print("done")

    # ── Timed runs ────────────────────────────────────────────────────────────
    # torch.cuda.Event records timestamps on the GPU timeline.
    # Avoids CPU-GPU synchronization overhead of time.time().
    elapsed_seconds = []

    with torch.no_grad():
        for _ in range(timed_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)

            start_event.record()                   # stamp GPU clock before
            _ = model.generate(**generate_kwargs)
            end_event.record()                     # stamp GPU clock after

            torch.cuda.synchronize()               # wait for GPU to complete
            # elapsed_time returns milliseconds; convert to seconds
            elapsed_seconds.append(start_event.elapsed_time(end_event) / 1000.0)

    avg_time = sum(elapsed_seconds) / len(elapsed_seconds)
    tokens_per_second = new_tokens / avg_time   # throughput metric

    return tokens_per_second, avg_time


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Memory usage helper
#
# GPU memory matters because FA-2's main benefit beyond speed is memory.
# At training time: FA-2 avoids the N×N attention matrix → O(N) memory.
# At inference time: the dominant memory cost is KV cache, not attention matrix.
# So memory difference at inference is smaller than during training.
#
# torch.cuda.memory_allocated()   — bytes currently allocated by tensors
# torch.cuda.memory_reserved()    — bytes reserved by the allocator (may be higher)
# torch.cuda.max_memory_allocated()— peak usage since last reset_peak_stats()
# ─────────────────────────────────────────────────────────────────────────────

def measure_peak_memory(model, tokenizer, seq_len, new_tokens):
    """
    Run one generation and return peak VRAM usage in MB.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()  # reset peak counter

    input_ids = torch.randint(
        low=0, high=tokenizer.vocab_size,
        size=(1, seq_len), device=DEVICE, dtype=torch.long,
    )

    with torch.no_grad():
        _ = model.generate(
            input_ids=input_ids,
            max_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return peak_mb


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Confirm CUDA is available — FA-2 requires a CUDA GPU
    assert torch.cuda.is_available(), "CUDA GPU required. FA-2 does not run on CPU."
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    model_standard, model_fa2, tokenizer = load_models()

    # ── Results table ────────────────────────────────────────────────────────
    # We'll collect results and print them together at the end for easy reading.
    results = []

    print("\n" + "=" * 70)
    print("Benchmarking throughput (tokens/sec) — higher is better")
    print("=" * 70)

    for seq_len in SEQUENCE_LENGTHS:
        print(f"\n[seq_len = {seq_len}]")

        # Standard attention
        print("  Standard attention:")
        tps_std, t_std = benchmark_throughput(
            model_standard, tokenizer,
            seq_len=seq_len,
            new_tokens=NEW_TOKENS,
            warmup_runs=WARMUP_RUNS,
            timed_runs=TIMED_RUNS,
        )
        mem_std = measure_peak_memory(model_standard, tokenizer, seq_len, NEW_TOKENS)
        print(f"    {tps_std:.1f} tok/s  |  avg {t_std*1000:.1f} ms  |  peak {mem_std:.0f} MB")

        # FlashAttention-2
        print("  FlashAttention-2:")
        tps_fa2, t_fa2 = benchmark_throughput(
            model_fa2, tokenizer,
            seq_len=seq_len,
            new_tokens=NEW_TOKENS,
            warmup_runs=WARMUP_RUNS,
            timed_runs=TIMED_RUNS,
        )
        mem_fa2 = measure_peak_memory(model_fa2, tokenizer, seq_len, NEW_TOKENS)
        print(f"    {tps_fa2:.1f} tok/s  |  avg {t_fa2*1000:.1f} ms  |  peak {mem_fa2:.0f} MB")

        speedup = tps_fa2 / tps_std
        mem_reduction = (mem_std - mem_fa2) / mem_std * 100
        print(f"  → Speedup: {speedup:.2f}×  |  Memory reduction: {mem_reduction:.1f}%")

        results.append({
            "seq_len": seq_len,
            "tps_std": tps_std,
            "tps_fa2": tps_fa2,
            "speedup": speedup,
            "mem_std_mb": mem_std,
            "mem_fa2_mb": mem_fa2,
        })

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'seq_len':>10} {'Std tok/s':>12} {'FA2 tok/s':>12} {'Speedup':>10} {'Mem std MB':>12} {'Mem FA2 MB':>12}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['seq_len']:>10} "
            f"{r['tps_std']:>12.1f} "
            f"{r['tps_fa2']:>12.1f} "
            f"{r['speedup']:>10.2f}× "
            f"{r['mem_std_mb']:>12.0f} "
            f"{r['mem_fa2_mb']:>12.0f}"
        )

    # ── Expected observations ─────────────────────────────────────────────────
    # seq_len=256:  Speedup ~1.0–1.2×  — N² too small, overhead dominates
    # seq_len=1024: Speedup ~1.3–1.8×  — FA-2 starts winning
    # seq_len=4096: Speedup ~2–3×      — FA-2's O(N) IO cost vs O(N²) is clear
    #
    # Memory difference is smaller at inference than training because:
    # - Inference doesn't store the N×N attention matrix for backward pass
    # - Dominant memory is model weights + KV cache, both same in both variants
    # FA-2 memory savings are larger during training (see verify script for that).


if __name__ == "__main__":
    main()