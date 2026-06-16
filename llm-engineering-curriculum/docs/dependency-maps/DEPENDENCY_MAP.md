# Cross-Module Dependency Map

This document is the textual companion to the dependency diagram. Use it to decide
module order and to sanity-check Capstone module selection.

## Dependency Table

| Module | Hard Prerequisites | Soft / Methodological Prerequisites |
|---|---|---|
| 01 — Deep Transformer Architectures | None (entry point) | — |
| 02 — GPT Internals & Generation Mechanics | Module 1 (causal masking, MHA, residual streams, LN/FFN) | — |
| 03 — Advanced Fine-Tuning & Alignment | Module 2 (forward pass, cross-entropy loss) | — |
| 04 — Production-Grade RAG | None hard | Module 2's context-window framing helps |
| 05 — Agentic AI & Cognitive Architectures | Modules 1–4 (tool use via structured prompting, RAG for memory) | — |
| 06 — Enterprise Evaluation & Benchmarking | None hard | Provides methodology used by 3, 11, 12, Capstone |
| 07 — LLMOps, Serving & Quantization | Module 1 (attention kernels), Module 3 (mixed precision) | — |
| 08 — Multimodal Foundations | Module 1 (attention as basis for ViT) | — |
| 09 — Model Context Protocol | Module 5 (tool/function calling, agent loops) | — |
| 10 — Multi-Agent Systems & Durable Swarms | Modules 5 & 9 (single-agent loops, MCP tool integration) | — |
| 11 — Reasoning Models & Test-Time Compute | Modules 2 & 3 (sampling, RLHF/reward modeling) | Module 6 methodology (bootstrap CI) |
| 12 — AI Safety, Alignment & Governance | Modules 3, 6, 7 (integration module) | Uses Module 3's fine-tuned model as target |
| Capstone | ≥6 modules of the student's choosing | Module 6 methodology mandatory for Milestone 3 |

## Recommended Build Order

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → Capstone
```

This is the default linear order and satisfies every dependency. Valid
alternative orderings exist (e.g., Module 6 can be pulled earlier since it has no
hard prerequisites — some students complete it right after Module 2 so its
evaluation methodology is available sooner for Module 3's alignment comparisons).

## Notes on Module 6's Special Role

Module 6 (Evaluation & Benchmarking) is unusual: it has no hard prerequisites, but
its **methodology** (bootstrap CIs, Fleiss' Kappa, LLM-as-Judge debiasing,
contamination scans) is required by:
- Module 3's LoRA vs DoRA / DPO vs KTO vs ORPO comparison tables
- Module 11's Pareto frontier with bootstrap CI
- Module 12's red-team attack success rate reporting
- Capstone Milestone 3 in full

Consider reading Module 6's notes (not necessarily completing its full project)
before Module 3, if your pacing allows.

## Notes on Module 12's Integration Role

Module 12 explicitly reuses the **Module 3 fine-tuned model** as its red-teaming
and model-card target. Do not skip or significantly alter Module 3's final
checkpoint before reaching Module 12 — document its location/config clearly in
Module 3's `project/results/`.
