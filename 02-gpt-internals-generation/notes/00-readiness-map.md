# Section A — Module Readiness Map

## Part A1 — Prerequisite Dependency Tree

- **FROM MODULE 1 (required)**
  - Causal masking
  - Multi-head attention and residual streams
  - Layer normalisation and feedforward blocks
- **LANGUAGE MODELLING FUNDAMENTALS**
  - Autoregressive factorisation: p(x₁...xₙ) = ∏p(xₜ|x<ₜ)
  - Cross-entropy loss
  - Perplexity = exp(cross-entropy)
- **TOKENIZATION CONCEPTS**
  - Why char/word tokenisation fails at scale
  - Subword intuition; vocabulary size tradeoffs
- **PROBABILITY & SAMPLING**
  - Distributions over discrete vocabularies
  - Temperature scaling
  - Greedy decoding and its limits

## YOU ARE NOW READY FOR
Decoder-only GPT → BPE tokenization → Sampling strategies → Speculative decoding

## Part A2 — Prerequisite Detail Table

| Concept | Why You Need It | Where to Learn It |
|---|---|---|
| Autoregressive LM | Core of decoder-only GPT | Karpathy: Let's build GPT (YouTube, 2hr) |
| Cross-entropy loss | GPT training objective | d2l.ai Ch.3.4; CS224N L4 |
| Softmax temperature | Base operation for all samplers | HF blog: How to generate text |
| BPE intuition | Debug generation failures | Karpathy: Let's build the GPT tokenizer |
| KV cache concept | Needed for speculative decoding / prefill-decode | Lilian Weng inference optimisation blog |

## Checklist
- [ ] I can write the autoregressive factorisation and explain causal masking's role
- [ ] I understand cross-entropy as -log p(correct token)
- [ ] I understand temperature's effect on logit distribution shape
- [ ] I can explain BPE merge process at a high level
- [ ] I understand why KV cache exists before studying speculative decoding
