# Master Timeline

A suggested 44-week pacing plan. Adjust based on prior experience — Modules 1–3 and
6 are foundational and shouldn't be rushed; Modules 8–11 can be reordered relative
to each other if a Capstone archetype demands it (just respect the dependency map).

| Weeks | Module | Focus |
|---|---|---|
| 1–4 | 01 — Deep Transformer Architectures | Attention, GQA/MQA, FlashAttention, RoPE, RMSNorm/SwiGLU, KV cache |
| 5–8 | 02 — GPT Internals & Generation Mechanics | Decoder-only GPT, tokenization, sampling, speculative decoding |
| 9–13 | 03 — Advanced Fine-Tuning & Alignment | LoRA/QLoRA/DoRA, RLHF/PPO, DPO/KTO/ORPO, training engineering |
| 14–18 | 04 — Production-Grade RAG | Hybrid retrieval, vector indexes, chunking, HyDE, GraphRAG, RAGAS |
| 19–23 | 05 — Agentic AI & Cognitive Architectures | ReAct/Reflexion/ToT, state machines, memory, HITL, safety |
| 24–27 | 06 — Enterprise Evaluation & Benchmarking | Metrics, bootstrap CI, LLM-as-Judge bias, CI/CD eval |
| 28–32 | 07 — LLMOps, Serving & Quantization | FSDP/ZeRO, GPTQ/AWQ/GGUF, vLLM/PagedAttention, observability |
| 33–35 | 08 — Multimodal Foundations | ViT, CLIP, early/late fusion, video/audio encoding |
| 36 | 09 — Model Context Protocol | MCP topology, primitives, transports, security, N×M |
| — | 10 — Multi-Agent Systems & Durable Swarms | Topologies, handoffs, failure recovery, swarms |
| — | 11 — Reasoning Models & Test-Time Compute | Scaling laws, PRM/ORM, Best-of-N, MCTS, self-refinement |
| — | 12 — AI Safety, Alignment & Governance | Alignment theory, SAEs, red-teaming, governance |
| 37–38 | Capstone Milestone 1 — Proposal | Architecture doc, success criteria, risk analysis |
| 39–40 | Capstone Milestone 2 — Prototype | End-to-end pipeline, baseline eval, failure modes |
| 41–42 | Capstone Milestone 3 — Evaluation Report | Bootstrap CI, ablations, human eval, contamination audit |
| 43–44 | Capstone Milestone 4 — Oral Defense | Presentation, panel Q&A, live failure demo |

> Modules 9–12 (one week each, slots flexible) should be scheduled before Capstone
> Milestone 1 if your chosen archetype depends on them (see Dependency Map). A common
> approach: compress Modules 9–12 into weeks 33–36 in parallel with Module 8 reading,
> since their core projects are lighter-weight than Modules 1–7.

## Per-Module Pacing Template
Each module week should roughly follow:
1. **Day 1–2:** Readiness map — fill gaps, complete checklist
2. **Day 3–5:** Topic notes — condense "Concepts to Learn" into `notes/`
3. **Day 6–10:** "What to Implement" — build each item in `src/`/`notebooks/`
4. **Remaining days:** Core Engineering Project + results report
