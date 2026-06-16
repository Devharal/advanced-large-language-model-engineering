# Module 11 — Reasoning Models and Test-Time Compute Scaling

> Analyse tradeoffs between training-time and inference-time compute scaling;
> implement core test-time compute strategies.

## Status
- [ ] Readiness map reviewed (`notes/00-readiness-map.md`)
- [ ] Part 1 notes + implementations
- [ ] Part 2 notes + implementations
- [ ] Core Engineering Project complete

## Folder Structure
```
11-reasoning-test-time-compute/
├── README.md
├── notes/
│   ├── 00-readiness-map.md
│   ├── 01-scaling-laws-system1-2.md
│   ├── 02-prm-orm-reward-models.md
│   └── 03-best-of-n-self-consistency-mcts.md
├── src/
│   ├── chinchilla_scaling_repro.py
│   ├── prm_training.py
│   ├── orm_majority_voting.py
│   ├── best_of_n.py
│   ├── self_consistency.py
│   ├── beam_search_reasoning.py
│   ├── mcts_reasoning.py
│   ├── self_refinement_loop.py
│   └── dynamic_compute_router.py
├── notebooks/
│   ├── test_time_accuracy_vs_compute.ipynb
│   └── pareto_frontier_bootstrap_ci.ipynb
└── project/
    ├── README.md
    └── results/
```

## Topics & Resource Directory

### Part 1 — Scaling Laws & Reward Models

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Scaling Laws & System 1/2 | Chinchilla C≈6ND, System-1 vs System-2, test-time scaling power law | Reproduce Chinchilla curve on 10M-300M models; plot test-time accuracy vs compute; find crossover point |
| Process & Outcome Reward Models | ORM vs PRM, formal verification, DeepSeek-R1/GRPO | Train PRM on math reasoning; ORM+majority vs PRM+beam search; verifiable reward via code test pass rate |

**Resources:**
- Paper: *Chinchilla* (Hoffmann et al. 2022); *Let's Verify Step by Step* (Lightman et al. 2023); *DeepSeek-R1* (2025)
- Course: Stanford CS25 V4 Test-Time Compute / Open-Source Reasoning Models
- Repo: `openai/evals`; `openai/prm800k`; `deepseek-ai/DeepSeek-R1`
- Blog: Karpathy Intro to LLMs; Nathan Lambert Understanding o1; Latent Space R1 deep dive

### Part 2 — Inference Strategies

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Best-of-N, Self-Consistency & Search | Best-of-N w/ ORM, self-consistency majority vote, beam search, MCTS | Best-of-N w/ ORM on GSM8K; self-consistency accuracy delta; beam search vs greedy on MATH |
| Self-Refinement & Agentic Reasoning | Iterative critique loops, when refinement helps/hurts, error accumulation, dynamic routing | Iterative critique + test execution loop; find degradation point; dynamic 3B/70B router |

**Resources:**
- Paper: *Self-Consistency Improves CoT Reasoning* (Wang et al. 2022)
- Course: DeepLearning.AI Reasoning with o1
- Repo: `hkust-nlp/dart-math`
- Blog: Sebastian Raschka Reasoning Models Deep Dive

## Core Engineering Project
**Test-Time Compute Scaling Harness** — see [`project/README.md`](project/README.md)
