# Core Engineering Project — Custom Transformer Block from Scratch

## Objective
Build a complete Transformer block using zero library attention layers, then profile
and validate it against production techniques.

## Deliverables
1. **Transformer block implementation** (`src/transformer_block.py`):
   GQA + RoPE + RMSNorm + SwiGLU + causal mask, pure PyTorch.
2. **FlashAttention-2 vs naive attention benchmark**: throughput, memory, latency
   across sequence lengths 512–8192. Output: `results/fa2_vs_naive.md` + plots.
3. **KV cache generation loop**: measure TTFT vs decoding latency at 4K and 16K
   tokens. Output: `results/kv_cache_latency.md`.
4. **Pre-LN vs Post-LN gradient study**: instrument gradient norms across 24 layers,
   reproduce the stability difference empirically. Output: `results/pre_ln_vs_post_ln.md`.

## Acceptance Checklist
- [ ] Causal mask verified — no future-token leakage (unit test)
- [ ] GQA/MQA/MHA all implemented with correct KV cache memory accounting
- [ ] RoPE applied correctly to Q/K before attention; YaRN scaling reproduced
- [ ] RMSNorm + SwiGLU numerically validated against reference formulas
- [ ] FA-2 output matches naive attention within tolerance (numerical equivalence)
- [ ] FLOPs vs seqlen plot shows quadratic growth for naive attention
- [ ] KV cache memory formula validated against measured memory usage
- [ ] Gradient norm plots clearly show Pre-LN stability advantage at depth

## Results
Place final report, plots, and tables in `results/`.
