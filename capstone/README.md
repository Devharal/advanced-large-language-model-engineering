# Capstone — Full-Stack Original LLM System Design and Defense

Design, build, evaluate, and defend an original end-to-end LLM system integrating
components from **at least 6 of the 12 modules**, evaluated against a rigorous
benchmark and defended in a doctoral-thesis-committee format.

## Folder Structure
```
capstone/
├── README.md
├── proposal/
│   ├── architecture_design_doc.md     # 15-25 pages
│   ├── problem_statement.md           # quantitative success criteria
│   ├── risk_analysis.md               # top 3 risks + mitigations
│   ├── data_sourcing_plan.md          # provenance, license, contamination plan
│   └── compute_budget.md
├── prototype/
│   ├── src/                           # working end-to-end pipeline
│   ├── README.md                      # build/run instructions, dependency lock
│   ├── baseline_evaluation.md         # vs random/naive baseline, with CI
│   └── failure_modes.md               # 3 concrete cases + root cause
├── evaluation/
│   ├── bootstrap_ci_report.md         # all primary metrics, 95% CI
│   ├── ablation_study.md
│   ├── human_evaluation.md            # 50+ outputs, Fleiss' Kappa >= 0.60
│   ├── contamination_audit.md
│   ├── safety_alignment_section.md
│   └── baseline_comparisons.md        # vs >=2 published/public baselines
├── defense/
│   ├── presentation.md / .pdf         # 25-min presentation outline
│   ├── panel_qa_prep.md               # anticipated adversarial questions
│   └── live_failure_demo.md
└── governance/
    ├── model_card.md
    ├── system_card.md
    └── eu_ai_act_risk_tier_mapping.md
```

## Milestones

### Milestone 1 — Proposal (Weeks 37–38, 10%)
- Written architectural design document (15–25 pages): system topology, data
  flows, component justification, module integration map
- Problem statement with quantitative success criteria (e.g. "F1 > 0.85 on
  held-out eval set")
- Risk analysis: top 3 architectural risks + mitigations
- Data sourcing plan: provenance, license, contamination scan methodology
- Compute budget estimate: GPU-hours, storage, reproduction cost

**Evaluation criteria:**
- [ ] Technically sound — no unresolvable architectural conflicts
- [ ] Scope is executable within semester compute allocation
- [ ] ≥6 modules clearly integrated with justified rationale
- [ ] Success criteria measurable, not subjective

### Milestone 2 — Prototype (Weeks 39–40, 20%)
- Working end-to-end pipeline (not production quality)
- Source repo with reproducible build instructions + dependency lock file
- Baseline evaluation vs random/naive baseline
- Failure mode report: 3 concrete cases with root cause analysis
- Revised risk assessment based on prototype findings

**Evaluation criteria:**
- [ ] Pipeline runs end-to-end on a fresh clone without manual intervention
- [ ] Baseline evaluation is statistically valid (includes CI, not single point)
- [ ] Failure modes are specific and mechanistic, not vague

### Milestone 3 — Evaluation Report (Weeks 41–42, 30%)
- Full Module 6 methodology: bootstrap 95% CI on all primary metrics
- Ablation study: remove each major component, quantify contribution
- Human evaluation of 50+ outputs: Fleiss' Kappa ≥ 0.60
- Contamination audit: confirm benchmark not in training data
- Safety & alignment section: red-team results, failure rate, residual risks
- Comparison to ≥2 published/public baselines on the same task

**Evaluation criteria:**
- [ ] All metrics reported with bootstrap 95% CI — no bare point estimates
- [ ] Ablation study reveals each component's relative contribution
- [ ] Kappa ≥ 0.60 on human eval; disagreements discussed
- [ ] Safety section identifies ≥1 real vulnerability + mitigation

### Milestone 4 — Oral Defense (Weeks 43–44, 40%)
- 25-minute presentation: problem, architecture, key decisions, results
- 20-minute panel examination: adversarial questions on design, failures, safety
- Defend each architectural choice vs alternatives
- Live demonstration: "show me where your system fails"
- Articulate safety/governance posture per Module 12 framework

**Evaluation criteria:**
- [ ] Architecture decisions defended with ablation evidence, not intuition
- [ ] ≥3 real failure modes identified live during examination
- [ ] Governance docs meet model card / system card standard (Module 12)
- [ ] Ownership of all engineering decisions demonstrated — no unexplained components

## Suggested System Archetypes

| Archetype | Module Integration |
|---|---|
| Agentic RAG Research Assistant | Hybrid GraphRAG + multi-agent orchestration + automated eval harness | 3, 4, 5, 6, 10, 12 |
| Domain-Specific Fine-Tuned Production API | QLoRA fine-tune + DPO + vLLM serving + observability + guardrails | 3, 6, 7, 12 |
| Multimodal Document Intelligence System | VLM adapter + RAG on PDFs/images + agent reasoning + eval pipeline | 3, 4, 5, 8, 6 |
| Reasoning-Enhanced Code Agent | Speculative decoding + test-time compute + MCP tools + safety red-team | 2, 5, 9, 11, 12 |
| Enterprise Multi-Agent Swarm | Durable orchestration + MCP tool ecosystem + GraphRAG KB + governance | 4, 5, 9, 10, 12 |
| Safety Research System | Mechanistic interpretability + automated red-teaming + watermarking + governance | 6, 11, 12 |

Students may propose alternatives that integrate ≥6 modules with a clear,
non-trivial real-world problem.

## Grading Weights
- Milestone 1 (Proposal): **10%**
- Milestone 2 (Prototype + Failure Mode Report): **20%**
- Milestone 3 (Evaluation Report): **30%**
- Milestone 4 (Oral Defense): **40%**
