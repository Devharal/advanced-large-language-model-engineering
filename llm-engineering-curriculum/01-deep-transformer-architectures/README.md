# Module 1 — Deep Transformer Architectures

> Master the mathematical foundations, memory layouts, and hardware execution patterns
> of the modern Transformer backbone.

## Status
- [ ] Readiness map reviewed (`notes/00-readiness-map.md`)
- [ ] Part 1 notes + implementations
- [ ] Part 2 notes + implementations
- [ ] Part 3 notes + implementations
- [ ] Core Engineering Project complete

## Folder Structure
```
01-deep-transformer-architectures/
├── README.md
├── notes/
│   ├── 00-readiness-map.md
│   ├── 01-attention-mha-gqa-mqa.md
│   ├── 02-flashattention.md
│   ├── 03-rope-positional-encoding.md
│   ├── 04-normalization-activations.md
│   └── 05-kv-cache-long-context.md
├── src/
│   ├── attention.py            # scaled dot-product, MHA, GQA, MQA
│   ├── flash_attention_bench.py
│   ├── rope.py
│   ├── rmsnorm_swiglu.py
│   ├── transformer_block.py    # full block: GQA + RoPE + RMSNorm + SwiGLU
│   └── kv_cache.py
├── notebooks/
│   ├── flops_vs_seqlen.ipynb
│   ├── pre_ln_vs_post_ln_gradients.ipynb
│   └── h2o_streamingllm_eviction.ipynb
├── benchmarks/
│   └── results.md
└── project/
    ├── README.md
    └── results/
```

## Topics & Resource Directory

### Part 1 — Attention Mechanisms

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Scaled Dot-Product & Multi-Head Attention | Attention(Q,K,V)=softmax(QKᵀ/√dₖ)V; causal masking; O(n²d) complexity | Pure PyTorch SDPA + causal mask + FLOPs vs seqlen benchmark |
| MHA vs GQA vs MQA | KV cache growth; head-group sharing; MLA (DeepSeek V3) | Implement all 3; convert MHA→GQA checkpoint; profile latency |
| FlashAttention 1/2/3 | IO-aware tiling, online softmax, FA-3 async Tensor Core/TMA | Enable FA-2 in HF; Nsight profiling; verify numerical equivalence |

**Resources:**
- Paper: *Attention Is All You Need* (Vaswani et al. 2017) | *FlashAttention-2* (Dao 2023)
- Course: Stanford CS224N L8 | CMU 11-868 GPU Memory & Attention Kernels
- Repo: `karpathy/nanoGPT` | `Dao-AILab/flash-attention`
- Blog: Jay Alammar *The Illustrated Transformer* | Tri Dao FlashAttention blog

### Part 2 — Positional Encoding & Normalization

| Topic | Key Concepts | What to Implement |
|---|---|---|
| RoPE | Complex-space rotation, relative position, YaRN/SuMA extrapolation | RoPE from scratch; YaRN scaling; RoPE vs ALiBi vs learned embeddings |
| Normalization & Activations | Pre-LN vs Post-LN, RMSNorm, SwiGLU | RMSNorm + SwiGLU FFN; gradient norm instrumentation across 32 layers |

**Resources:**
- Paper: *RoFormer* (Su et al. 2021)
- Course: Stanford CS25 RoPE lecture
- Repo: `lucidrains/rotary-embedding-torch`
- Blog: EleutherAI *Rotary Embeddings — A Relative Revolution*

### Part 3 — KV Cache & Long-Context Systems

| Topic | Key Concepts | What to Implement |
|---|---|---|
| KV Cache & Memory Layout | Memory formula, H2O/StreamingLLM eviction, Sliding Window Attention | KV cache generation loop; H2O eviction sim; prefill vs decode profiling |

**Resources:**
- Paper: *StreamingLLM* (Xiao et al. 2023)
- Course: MIT 6.S191 Sequence Modelling
- Repo: `mit-han-lab/streaming-llm`
- Blog: Lilian Weng *Large Transformer Model Inference Optimisation*

## Core Engineering Project
**Custom Transformer Block from Scratch** — see [`project/README.md`](project/README.md)
