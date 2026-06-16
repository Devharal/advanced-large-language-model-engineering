# Module 2 — GPT Internals and Generation Mechanics

> Deconstruct autoregressive decoder-only architectures, decoding algorithms, and
> sequence generation bottlenecks.

## Status
- [ ] Readiness map reviewed (`notes/00-readiness-map.md`)
- [ ] Part 1 notes + implementations
- [ ] Part 2 notes + implementations
- [ ] Part 3 notes + implementations
- [ ] Core Engineering Project complete

## Folder Structure
```
02-gpt-internals-generation/
├── README.md
├── notes/
│   ├── 00-readiness-map.md
│   ├── 01-decoder-only-architecture.md
│   ├── 02-tokenization-bpe-sentencepiece.md
│   ├── 03-sampling-strategies.md
│   ├── 04-speculative-decoding.md
│   └── 05-generation-failure-modes.md
├── src/
│   ├── minimal_gpt.py
│   ├── bpe_tokenizer.py
│   ├── sampling.py            # temperature, top-k, top-p, min-p, repetition penalty
│   ├── speculative_decoding.py
│   └── repetition_monitor.py
├── notebooks/
│   ├── training_loss_curves.ipynb
│   ├── fertility_rate_by_language.ipynb
│   └── prefill_decode_profile.ipynb
├── benchmarks/
│   └── results.md
└── project/
    ├── README.md
    └── results/
```

## Topics & Resource Directory

### Part 1 — Autoregressive Architecture & Tokenization

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Decoder-Only Architecture | Autoregressive factorisation, causal masking, logits, cross-entropy | Minimal GPT-2 style model; causal mask visualisation; loss curves |
| Tokenization | BPE, SentencePiece, fertility rate, token healing | Train BPE tokenizer; fertility rate across languages; token healing |

**Resources:**
- Paper: *Language Models are Few-Shot Learners* (Brown et al. 2020); *Neural MT of Rare Words with Subword Units* (Sennrich et al. 2016)
- Course: Karpathy *Let's build GPT from scratch*; *Let's build the GPT Tokenizer*
- Repo: `karpathy/nanoGPT`; `openai/tiktoken`
- Blog: Jay Alammar *The Illustrated GPT-2*; HF *Summary of the Tokenizers*

### Part 2 — Sampling & Decoding Algorithms

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Sampling Strategies | Temperature, top-k, top-p, min-p, repetition penalty | Configurable generation loop with logit processor pipeline; ablations |
| Speculative Decoding | Draft-verifier, acceptance criteria, Medusa, EAGLE | Speculative decoding loop; measure speedup vs draft length |

**Resources:**
- Paper: *The Curious Case of Neural Text Degeneration* (Holtzman et al. 2019); *Fast Inference via Speculative Decoding* (Leviathan et al. 2023)
- Course: Stanford CS224N L9 (Decoding Algorithms)
- Repo: `huggingface/transformers` GenerationConfig; `EAGLE-LLM/EAGLE`
- Blog: Maxime Labonne *Decoding Strategies in LLMs*; Introl.com Speculative Decoding 2025

### Part 3 — Generation Failure Modes

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Prefill/Decode Phases & Failure Modes | Compute-bound vs memory-bound, repetition loops, semantic collapse, KV OOM | Profile long-generation run; n-gram repetition monitor; coherence vs length |

## Core Engineering Project
**Speculative Decoding Engine** — see [`project/README.md`](project/README.md)
