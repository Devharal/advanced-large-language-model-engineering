# Advanced Large Language Model Engineering
### From Foundations to Advanced Agentic Architectures

[![Status](https://img.shields.io/badge/status-in%20progress-yellow)]()
[![Modules](https://img.shields.io/badge/modules-12%20%2B%20capstone-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

A personally designed, self-study curriculum for building deep, implementation-level
mastery of modern LLM engineering — from transformer internals to multi-agent,
multimodal, and safety-critical production systems.

This repository is the **single source of truth** for the course: structured notes,
from-scratch implementations, benchmarks, resource directories, and the hands-on
**Core Engineering Project** for every module, plus a full **Capstone**.

---

## How to Use This Repo

1. **Work module-by-module, in order.** Each module's `Section A — Module Readiness Map`
   (see `notes/00-readiness-map.md` in each folder) lists prerequisites — do not skip
   these even if topics feel familiar.
2. **Notes first, code second.** Read/condense the topic notes into `notes/`, then
   implement everything listed under *"What to Implement"* in `src/` or `notebooks/`.
3. **Every module ends with a Core Engineering Project.** This is the non-negotiable
   deliverable — it lives in each module's `project/` folder with its own README,
   acceptance criteria, and results report.
4. **Track progress** using the [Master Timeline](docs/timeline/MASTER_TIMELINE.md) and
   each module's dashboard (`docs/dashboards/`).
5. **Respect dependencies.** See [Cross-Module Dependency Map](docs/dependency-maps/DEPENDENCY_MAP.md)
   before starting Modules 8–12 and the Capstone — they assume completed prior modules.
6. **Log everything.** Use `docs/dashboards/progress.md` to check off readiness items,
   "what to implement" items, and project milestones as you complete them.

---

## Repository Structure

```
llm-engineering-curriculum/
├── README.md                          # You are here
├── 01-deep-transformer-architectures/
├── 02-gpt-internals-generation/
├── 03-finetuning-alignment/
├── 04-production-rag/
├── 05-agentic-ai-cognitive-architectures/
├── 06-evaluation-benchmarking/
├── 07-llmops-serving-quantization/
├── 08-multimodal-foundations/
├── 09-model-context-protocol/
├── 10-multi-agent-systems/
├── 11-reasoning-test-time-compute/
├── 12-ai-safety-alignment-governance/
├── capstone/
├── docs/
│   ├── timeline/MASTER_TIMELINE.md
│   ├── dependency-maps/DEPENDENCY_MAP.md
│   └── dashboards/progress.md
├── resources/
│   ├── papers/
│   └── datasets/
└── scripts/
```

### Per-Module Structure (applies to Modules 1–12)

```
NN-module-name/
├── README.md              # Module overview, topics, resource directory, links to project
├── notes/
│   ├── 00-readiness-map.md      # Section A: prerequisite dependency tree + detail table
│   ├── 01-<topic>.md            # One file per topic (concepts to learn, condensed)
│   └── ...
├── src/                    # From-scratch implementations (pure PyTorch / pure Python)
├── notebooks/              # Exploration, ablations, plots, benchmark notebooks
├── configs/ (where relevant)
├── data/ / servers/ / dashboards/ (module-specific extras)
└── project/
    ├── README.md           # Core Engineering Project spec + acceptance checklist
    └── results/             # Final report, plots, tables, benchmark outputs
```

---

## Module Index

| # | Module | Core Theme | Core Engineering Project |
|---|--------|-----------|---------------------------|
| [01](01-deep-transformer-architectures/) | Deep Transformer Architectures | MHA → GQA/MQA → FlashAttention → RoPE → KV Cache | [Custom Transformer Block from Scratch](01-deep-transformer-architectures/project/) |
| [02](02-gpt-internals-generation/) | GPT Internals & Generation Mechanics | Decoder-only GPT, tokenization, sampling, speculative decoding | [Speculative Decoding Engine](02-gpt-internals-generation/project/) |
| [03](03-finetuning-alignment/) | Advanced Fine-Tuning & Alignment | LoRA/QLoRA/DoRA, RLHF/PPO, DPO, KTO, ORPO | [End-to-End Alignment Pipeline](03-finetuning-alignment/project/) |
| [04](04-production-rag/) | Production-Grade RAG | Hybrid search, GraphRAG, chunking, HyDE, RAGAS | [Hybrid GraphRAG Pipeline](04-production-rag/project/) |
| [05](05-agentic-ai-cognitive-architectures/) | Agentic AI & Cognitive Architectures | ReAct/Reflexion, state machines, memory tiers, HITL, safety | [Framework-Free Cyclic State-Machine Agent](05-agentic-ai-cognitive-architectures/project/) |
| [06](06-evaluation-benchmarking/) | Enterprise Evaluation & Benchmarking | LLM-as-Judge, bootstrap CI, bias audits, CI/CD evals | [Automated Evaluation Pipeline](06-evaluation-benchmarking/project/) |
| [07](07-llmops-serving-quantization/) | LLMOps, Serving & Quantization | FSDP/ZeRO, GPTQ/AWQ/GGUF, vLLM/PagedAttention, observability | [Production Inference Cluster](07-llmops-serving-quantization/project/) |
| [08](08-multimodal-foundations/) | Multimodal Foundations | ViT, CLIP, early/late fusion, video/audio encoding | [Late-Fusion Vision-Language Adapter](08-multimodal-foundations/project/) |
| [09](09-model-context-protocol/) | Model Context Protocol | MCP topology, primitives, transports, security, N×M | [Secure MCP Server + Agentic Loop](09-model-context-protocol/project/) |
| [10](10-multi-agent-systems/) | Multi-Agent Systems & Durable Swarms | Topologies, handoffs, cascading failure recovery, swarms | [Durable Multi-Agent Orchestration Grid](10-multi-agent-systems/project/) |
| [11](11-reasoning-test-time-compute/) | Reasoning Models & Test-Time Compute | Scaling laws, PRM/ORM, Best-of-N, self-consistency, MCTS | [Test-Time Compute Scaling Harness](11-reasoning-test-time-compute/project/) |
| [12](12-ai-safety-alignment-governance/) | AI Safety, Alignment & Governance | Alignment theory, SAE interpretability, red-teaming, governance | [Layered Safety Evaluation Suite](12-ai-safety-alignment-governance/project/) |
| 🎓 | [Capstone](capstone/) | Full-Stack Original LLM System Design & Defense | 4-Milestone thesis-style project |

---

## Module Details

### Module 1 — Deep Transformer Architectures
**Goal:** Master the mathematical foundations, memory layouts, and hardware execution
patterns of the modern Transformer backbone.

| Part | Topics |
|------|--------|
| 1 — Attention Mechanisms | Scaled Dot-Product & Multi-Head Attention, MHA vs GQA vs MQA, FlashAttention 1/2/3 |
| 2 — Positional Encoding & Normalization | RoPE (+ YaRN), Pre-LN vs Post-LN, RMSNorm, SwiGLU |
| 3 — KV Cache & Long-Context Systems | KV cache memory layout, H2O / StreamingLLM eviction, Sliding Window Attention |

**Core Engineering Project:** Custom Transformer Block from Scratch — GQA + RoPE +
RMSNorm + SwiGLU + causal mask; FlashAttention-2 vs naive attention profiling; KV cache
generation loop; Pre-LN vs Post-LN gradient stability study.

---

### Module 2 — GPT Internals and Generation Mechanics
**Goal:** Deconstruct autoregressive decoder-only architectures, decoding algorithms,
and sequence generation bottlenecks.

| Part | Topics |
|------|--------|
| 1 — Autoregressive Architecture & Tokenization | Decoder-only GPT, BPE, SentencePiece, token healing |
| 2 — Sampling & Decoding Algorithms | Temperature, top-k, top-p, min-p, repetition penalty, speculative decoding (Medusa, EAGLE) |
| 3 — Generation Failure Modes | Prefill/decode phases, repetition loops, semantic collapse, KV cache OOM |

**Core Engineering Project:** Speculative Decoding Engine — full configurable sampler
pipeline; speculative decoding with 7B verifier + 1B draft; acceptance rate vs draft
length; n-gram repetition loop detector.

---

### Module 3 — Advanced Fine-Tuning and Alignment Paradigms
**Goal:** Design and execute data-packed, parameter-efficient fine-tuning pipelines
and mathematically grounded alignment routines.

| Part | Topics |
|------|--------|
| 1 — PEFT | LoRA, QLoRA (NF4, double quant, paged optimizers), DoRA |
| 2 — Alignment Paradigms | RLHF + PPO, DPO, KTO, ORPO |
| 3 — Training Engineering | Mixed precision (FP16/BF16/FP8), gradient checkpointing/accumulation, dataset packing |

**Core Engineering Project:** End-to-End Alignment Pipeline — QLoRA + custom LoRA on a
7B model, DPO on packed preference data; LoRA vs DoRA and DPO vs KTO vs ORPO
comparison tables; full training instrumentation; merged adapter deployment.

---

### Module 4 — Production-Grade Retrieval-Augmented Generation
**Goal:** Engineer resilient, deterministic knowledge-retrieval systems combining
hybrid search, semantic indexing, and graph topologies.

| Part | Topics |
|------|--------|
| 1 — Embedding & Search | Dense/sparse hybrid (RRF), cross-encoder reranking, HNSW/IVF vector indexes |
| 2 — Chunking & Query Enhancement | Fixed/semantic/parent-child/agentic chunking, HyDE, self-querying, multi-query |
| 3 — GraphRAG & Production Hardening | Entity extraction, community detection, "lost in the middle", RAGAS triad |

**Core Engineering Project:** Hybrid GraphRAG Pipeline — Neo4j entity graph + community
summaries, fused with HNSW via RRF, parent-child chunking + HyDE; RAGAS-gated CI/CD;
single-hop vs multi-hop benchmark (GraphRAG vs naive RAG vs hybrid).

---

### Module 5 — Agentic AI and Cognitive Architectures
**Goal:** Design robust multi-turn reasoning loops using state machines, custom
parsers, and safe execution boundaries.

| Part | Topics |
|------|--------|
| 1 — Reasoning Frameworks | ReAct, Plan-and-Solve, Reflexion, Tree-of-Thoughts, Self-Discover |
| 2 — State & Memory Management | State machines/graphs (cyclic), in-context/summary/sliding-window/external memory, checkpointing |
| 3 — Tool Use, HITL & Safety | JSON schema tool parsing, HITL interruption/resume, prompt injection, loop detection |

**Core Engineering Project:** Framework-Free Cyclic State-Machine Agent — Plan → Act →
Observe → Critique → Plan loop in pure Python; 3 real tools; HITL approval gate;
semantic loop detector + step-budget enforcer; prompt-injection recovery demo.

---

### Module 6 — Enterprise Evaluation and Scientific Benchmarking
**Goal:** Establish rigorous, statistically sound, and automated verification
frameworks for non-deterministic AI systems.

| Part | Topics |
|------|--------|
| 1 — Evaluation Metrics | EM, BLEU/ROUGE, BERTScore, G-Eval/LLM-as-Judge, RAGAS triad |
| 2 — Statistical Rigour | Bootstrap CIs, Cohen's/Fleiss' Kappa, multiple comparison correction |
| 3 — CI/CD Evaluation Pipelines | Two-tier eval (fast deterministic + slow stochastic), regression alerts, Evol-Instruct |

**Core Engineering Project:** Automated Evaluation Pipeline — two-tier CI/CD with
position-swap debiased LLM-as-Judge; bootstrap 95% CI + Fleiss' Kappa everywhere;
MMLU/GSM8K contamination scan; 500 synthetic test cases via Evol-Instruct.

---

### Module 7 — LLMOps, Serving Infrastructure, and Quantization Physics
**Goal:** Deploy, optimize, and scale large open-source models under strict hardware
memory constraints and high-concurrency requirements.

| Part | Topics |
|------|--------|
| 1 — Distributed Training | Data/Tensor/Pipeline Parallelism, ZeRO, FSDP |
| 2 — Quantization | INT8/INT4 PTQ, GPTQ, AWQ, GGUF/llama.cpp K-quants |
| 3 — Serving Infrastructure | vLLM/PagedAttention, continuous batching, prefix caching, observability/cost |

**Core Engineering Project:** Production Inference Cluster — vLLM with PagedAttention +
continuous batching + prefix caching serving 7B+70B; dynamic 3-tier router;
OpenTelemetry + Grafana dashboards; load test to saturation with KV-cache eviction
storm diagnosis.

---

### Module 8 — Multimodal Foundations and Cross-Modal Alignment
**Goal:** Unify textual semantic spaces with visual and audio temporal arrays using
shared representational topologies.

| Part | Topics |
|------|--------|
| 1 — Visual Foundations | ViT, CLIP contrastive pre-training, Perceiver Resampler, early vs late fusion, AnyRes |
| 2 — Video & Audio | Spatio-temporal attention, frame sampling, mel-spectrogram tokenization, cross-modal contrastive loss |
| 3 — Production Challenges | Inter-modal interference, context-scaling bottleneck, mixed-modality batching |

**Core Engineering Project:** Late-Fusion Vision-Language Adapter — frozen ViT-L + 7B
LLM with custom 2-layer MLP projector on LLaVA-style data; AnyRes tiling on
VQAv2/COCO; MMLU/HumanEval regression check; mixed-modality dynamic batching profile.

---

### Module 9 — Model Context Protocol and Standardized Tool Integration
**Goal:** Standardize connection architectures between frontier models and external
environments using uniform real-time protocols.

| Part | Topics |
|------|--------|
| 1 — MCP Architecture | Host/Client/Server topology, JSON-RPC 2.0, Resources/Tools/Prompts/Sampling, stdio/SSE |
| 2 — Security & Sandboxing | Credential containment, Docker/Wasm sandboxing, access control, injection via tool results |
| 3 — Enterprise Integration | N×M problem, tool discovery, multi-server orchestration, latency overhead |

**Core Engineering Project:** Secure MCP Server + Agentic Loop — sandboxed code
executor (Docker) + read-only filesystem + web search over stdio; credential isolation
+ output sanitisation + injection guard; session-drop recovery with checkpoint restore.

---

### Module 10 — Advanced Multi-Agent Systems and Durable Swarms
**Goal:** Architect decentralized, asynchronous, infinite-horizon multi-agent networks
with durable state and autonomous coordination.

| Part | Topics |
|------|--------|
| 1 — Multi-Agent Topologies | Orchestrator-worker, hierarchical supervision, peer-to-peer, durable execution, handoffs/HMAC, context isolation |
| 2 — Failure Recovery & Swarm Dynamics | Cascading failure rates, checkpoint rollback, deadlock detection, dynamic spawning, audit logs |

**Core Engineering Project:** Durable Multi-Agent Orchestration Grid — durable
orchestrator that spawns specialist workers and merges results; per-step checkpointing
with mid-task kill/restart; deadlock detection + cascading failure containment;
structured JSON audit log.

---

### Module 11 — Reasoning Models and Test-Time Compute Scaling
**Goal:** Analyse tradeoffs between training-time and inference-time compute scaling;
implement core test-time compute strategies.

| Part | Topics |
|------|--------|
| 1 — Scaling Laws & Reward Models | Chinchilla scaling, System 1/2, ORM vs PRM, DeepSeek-R1/GRPO |
| 2 — Inference Strategies | Best-of-N, self-consistency, beam search over reasoning, MCTS, self-refinement, dynamic routing |

**Core Engineering Project:** Test-Time Compute Scaling Harness — 7B model under
single-pass / Best-of-N (N=1,4,16,64) / iterative self-refinement; PRM-guided beam
search vs ORM-guided Best-of-N; compute-accuracy Pareto frontier with 95% bootstrap
CI; error-accumulation demo in a 5-step agentic task.

---

### Module 12 — AI Safety, Alignment, and Governance Engineering
**Goal:** Design, implement, and document technical safety controls and governance
processes spanning the full model lifecycle.

| Part | Topics |
|------|--------|
| 1 — Alignment & Interpretability | Reward hacking, specification gaming, Constitutional AI, SAEs, activation patching, logit lens |
| 2 — Red Teaming & Defense | Jailbreak taxonomy, automated red-team pipelines, OWASP LLM Top 10, defense-in-depth, watermarking |
| 3 — Governance & Reporting | Model cards, system cards, EU AI Act risk tiers, NIST AI RMF |

**Core Engineering Project:** Layered Safety Evaluation Suite — automated red-team
(200 prompts × 5 OWASP categories); 3-layer defense with a bypass-all-3 demo;
green-list watermarking at 100/200/500 tokens; full model card + system card mapped to
EU AI Act risk tier (using the Module 3 fine-tuned model).

---

## Capstone — Full-Stack Original LLM System Design and Defense

Integrate components from **at least 6 of the 12 modules** into an original,
non-trivial LLM system, evaluated against a rigorous self-designed benchmark and
defended in a thesis-committee-style review.

| Milestone | Weight | Folder |
|-----------|--------|--------|
| 1 — Proposal (architecture doc, success criteria, risk analysis, data/compute plan) | 10% | [`capstone/proposal/`](capstone/proposal/) |
| 2 — Prototype (end-to-end pipeline, baseline eval, failure mode report) | 20% | [`capstone/prototype/`](capstone/prototype/) |
| 3 — Evaluation Report (bootstrap CIs, ablations, human eval ≥0.60 Kappa, contamination audit, safety section) | 30% | [`capstone/evaluation/`](capstone/evaluation/) |
| 4 — Oral Defense (presentation, panel Q&A, live failure demo, governance docs) | 40% | [`capstone/defense/`](capstone/defense/) |

**Suggested archetypes** (see [`capstone/README.md`](capstone/README.md) for full
descriptions and module mappings):
- Agentic RAG Research Assistant (Modules 3, 4, 5, 6, 10, 12)
- Domain-Specific Fine-Tuned Production API (Modules 3, 6, 7, 12)
- Multimodal Document Intelligence System (Modules 3, 4, 5, 8, 6)
- Reasoning-Enhanced Code Agent (Modules 2, 5, 9, 11, 12)
- Enterprise Multi-Agent Swarm (Modules 4, 5, 9, 10, 12)
- Safety Research System (Modules 6, 11, 12)

Governance artifacts (model card, system card, EU AI Act mapping) live in
[`capstone/governance/`](capstone/governance/).

---

## Cross-Module Dependencies

See [`docs/dependency-maps/DEPENDENCY_MAP.md`](docs/dependency-maps/DEPENDENCY_MAP.md)
for the full graph. Summary:

- **Module 1** is the foundation for everything (attention, normalization, KV cache).
- **Module 2** builds directly on Module 1 (causal masking, residual streams).
- **Module 3** requires Modules 1–2 (forward pass, cross-entropy loss).
- **Module 4** is largely independent but assumes Module 2's context-window framing.
- **Module 5** requires Modules 1–4 (tool use via structured prompting, RAG for memory).
- **Module 6** is independent but is a prerequisite *methodology* for Modules 3, 11, 12,
  and the Capstone.
- **Module 7** requires Modules 1 & 3 (attention kernels, mixed precision).
- **Module 8** requires Module 1 (attention as the basis for ViT).
- **Module 9** requires Module 5 (tool/function calling, agent loops).
- **Module 10** requires Modules 5 & 9 (single-agent loops, MCP tool integration).
- **Module 11** requires Modules 2 & 3 (sampling, RLHF/reward modeling).
- **Module 12** is an integration module — requires Modules 3, 6, 7 at minimum.
- **Capstone** requires ≥6 modules of the student's choosing.

---

## Progress Tracking

- [Master Timeline](docs/timeline/MASTER_TIMELINE.md) — week-by-week plan across all
  12 modules + capstone (44+ weeks).
- [Progress Dashboard](docs/dashboards/progress.md) — checklist per module: readiness
  map ✅, topic notes ✅, "what to implement" items ✅, Core Engineering Project ✅,
  resource directory reviewed ✅.

---

## Resources

- [`resources/papers/`](resources/papers/) — local notes/summaries on must-read papers
  per module (organized by module number prefix, e.g. `01-attention-is-all-you-need.md`).
- [`resources/datasets/`](resources/datasets/) — dataset notes, licenses, and
  download/prep scripts.
- Each module's `README.md` contains the full **Resource Directory** table (paper,
  university course, GitHub repo, blog/article) as originally curated.

---

## Setup

```bash
git clone <your-repo-url>
cd llm-engineering-curriculum
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # create per-module requirements as needed
```

Each module/project folder may have its own `requirements.txt` or environment file
where dependencies diverge significantly (e.g., quantization libraries in Module 7,
Neo4j + GraphRAG in Module 4).

---

## License

MIT — for personal study material. Cited papers, courses, and external repositories
retain their own licenses; this repo only contains original notes, code, and
configuration.
