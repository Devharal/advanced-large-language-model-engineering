# Section A — Module Readiness Map

Read this before starting Module 1. Work top-to-bottom; gaps surface as confusion later.

## Part A1 — Prerequisite Dependency Tree

- **MATHEMATICS**
  - Linear algebra: matrix multiply, dot product, vector spaces, rank
  - Calculus: partial derivatives, chain rule, Jacobians
  - Probability: distributions, softmax as normalised exp, entropy
- **DEEP LEARNING FOUNDATIONS**
  - Feedforward NNs: layers, activations (ReLU, GeLU), loss functions
  - Backpropagation: gradient flow, vanishing gradients, learning rate
  - Normalisation: BatchNorm intuition → LayerNorm
  - GPU memory model: HBM vs SRAM, bandwidth vs compute bound
- **SEQUENCE MODELLING HISTORY** (context, not mastery)
  - RNN/LSTM: why sequential processing is a bottleneck
  - Seq2Seq with Bahdanau attention
- **DIRECT PREREQUISITES**
  - Vanilla scaled dot-product attention (single head)
  - Positional encoding: sine/cosine
  - Vaswani et al. 2017 (original Transformer)
  - Encoder vs decoder vs encoder-decoder

## YOU ARE NOW READY FOR
Multi-Head Attention → GQA/MQA → FlashAttention 1/2/3 → RoPE → KV Cache

## Part A2 — Prerequisite Detail Table

| Concept | Why You Need It | Where to Learn It |
|---|---|---|
| Matrix multiply & dot product | Attention is QKᵀ | 3Blue1Brown: Essence of Linear Algebra |
| Softmax function | Attention weights via softmax, numerical stability | CS224N L1; d2l.ai Ch.3 |
| Backpropagation | Pre-LN vs Post-LN gradient stability | Karpathy micrograd; CS231n |
| LayerNorm | RMSNorm is a simplification | Ba et al. Layer Normalization (2016) |
| GPU memory hierarchy | FlashAttention's IO-awareness motivation | NVIDIA Hopper whitepaper; Tri Dao blog |
| Bahdanau attention (2014) | MHA's historical origin | Bahdanau et al. arXiv:1409.0473 |

## Checklist
- [ ] I can explain why attention is QKᵀ/√dₖ
- [ ] I can derive softmax numerical stability (log-sum-exp)
- [ ] I understand Pre-LN vs Post-LN gradient flow conceptually
- [ ] I understand HBM vs SRAM and why it matters for kernels
- [ ] I can explain Bahdanau attention's role in MHA's history
