# Section A — Module Readiness Map

## Part A1 — Prerequisite Dependency Tree

- **FROM MODULE 2 (required)**
  - Decoder-only architecture and forward pass mechanics
  - Cross-entropy training loss and gradient descent
- **OPTIMISATION FOUNDATIONS**
  - Adam/AdamW: adaptive LR, weight decay
  - LR schedulers: warmup, cosine decay
  - Overfitting: train/val divergence, regularisation
- **MATRIX THEORY FOR LORA**
  - Matrix rank; SVD intuition
  - Why fine-tuning updates are intrinsically low-rank (Aghajanyan 2021)
- **REINFORCEMENT LEARNING BASICS (for RLHF)**
  - Reward, policy, value function — conceptual
  - Policy gradient
  - KL divergence

## YOU ARE NOW READY FOR
LoRA → QLoRA → DoRA → RLHF+PPO → DPO → KTO → ORPO → Training Engineering

## Part A2 — Prerequisite Detail Table

| Concept | Why You Need It | Where to Learn It |
|---|---|---|
| Full fine-tuning mechanics | Baseline for understanding LoRA's efficiency | HF: Fine-tuning a pre-trained model |
| Matrix rank & SVD intuition | LoRA's rank hyperparameter | 3Blue1Brown Linear Algebra Ch.7-9 |
| KL divergence | PPO KL penalty, DPO reference comparison | Lilian Weng Diffusion blog (KL section); d2l.ai stats |
| Policy gradient / REINFORCE | PPO is an advanced policy gradient method | Spinning Up in Deep RL; Sutton & Barto Ch.13 |
| Preference data / Bradley-Terry | RLHF/DPO data pipelines | InstructGPT paper Sections 1-3 |
| Quantization basics (INT8/FP16) | QLoRA's NF4 rationale | HF Quantization docs; Tim Dettmers bitsandbytes blog |

## Checklist
- [ ] I understand what full fine-tuning updates and why it's expensive
- [ ] I can explain matrix rank and what "low-rank update" means
- [ ] I can explain KL divergence and its role in PPO/DPO
- [ ] I understand policy gradient at a conceptual level
- [ ] I understand Bradley-Terry pairwise preference modeling
- [ ] I understand why NF4 suits normally-distributed weights
