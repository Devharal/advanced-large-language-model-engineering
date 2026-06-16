# Section A — Module Readiness Map

## Part A1 — Prerequisite Dependency Tree

- **FROM MODULES 2 & 3 (required)**
  - Autoregressive generation and sampling strategies
  - RLHF and reward modelling fundamentals
- **SCALING LAW INTUITION**
  - Power laws (N^α)
  - Chinchilla compute-optimal training
  - Train-time vs inference-time compute substitution
- **SEARCH ALGORITHMS**
  - Greedy search; beam search; Monte Carlo Tree Search
- **PROCESS REWARD MODELS**
  - Outcome reward (sparse) vs process reward (dense)

## YOU ARE NOW READY FOR
Scaling Laws → PRM/ORM → Best-of-N → Self-Consistency → MCTS → Self-Refinement

## Part A2 — Prerequisite Detail Table

| Concept | Why You Need It | Where to Learn It |
|---|---|---|
| Chinchilla scaling law | Test-time compute is the "inference-side" of scaling | Hoffmann et al. 2022; Karpathy scaling laws lecture |
| Beam search | Best-of-N / PRM-guided search extend this | HF "How to generate text"; CS224N decoding lecture |
| Monte Carlo Tree Search | Used by AlphaCode 2 and reasoning systems | Wikipedia MCTS; David Silver AlphaGo lecture |
| Process reward models | o1's breakthrough is PRM-guided search | Lightman et al. 2023 "Let's Verify Step by Step" |
| GSM8K/MATH/HumanEval familiarity | Needed for test-time compute experiments | GSM8K paper; HumanEval paper |

## Checklist
- [ ] I can explain the Chinchilla compute-optimal tradeoff (N vs D)
- [ ] I understand beam search vs greedy decoding
- [ ] I understand MCTS at a conceptual level (rollouts, expansion, backprop)
- [ ] I can distinguish ORM (final answer) vs PRM (step-level)
- [ ] I'm familiar with GSM8K/MATH/HumanEval benchmark formats
