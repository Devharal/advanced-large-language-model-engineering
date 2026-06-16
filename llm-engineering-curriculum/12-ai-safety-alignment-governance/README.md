# Module 12 — AI Safety, Alignment, and Governance Engineering

> Design, implement, and document technical safety controls and governance processes
> spanning the full model lifecycle.

## Status
- [ ] Readiness map reviewed (`notes/00-readiness-map.md`)
- [ ] Part 1 notes + implementations
- [ ] Part 2 notes + implementations
- [ ] Part 3 notes + implementations
- [ ] Core Engineering Project complete

## Folder Structure
```
12-ai-safety-alignment-governance/
├── README.md
├── notes/
│   ├── 00-readiness-map.md
│   ├── 01-alignment-theory.md          # reward hacking, Goodhart's, corrigibility
│   ├── 02-mechanistic-interpretability.md  # SAEs, activation patching, logit lens
│   ├── 03-automated-red-teaming.md
│   ├── 04-defense-in-depth.md
│   └── 05-governance-frameworks.md     # model/system cards, EU AI Act, NIST RMF
├── src/
│   ├── reward_hacking_demo.py
│   ├── constitutional_ai_loop.py       # critique-revision
│   ├── corrigibility_test_suite.py
│   ├── activation_patching.py          # GPT-2 small "Eiffel Tower -> Paris"
│   ├── sparse_autoencoder.py
│   ├── logit_lens.py
│   ├── red_team_pipeline.py            # 200 prompts x 5 OWASP categories
│   ├── refusal_consistency.py
│   ├── jailbreak_classifier.py
│   ├── defense_in_depth.py             # input filter + LLM + output classifier
│   └── watermarking.py                 # green-list scheme
├── notebooks/
│   └── sae_feature_exploration.ipynb
└── project/
    ├── README.md
    └── results/
```

## Topics & Resource Directory

### Part 1 — Alignment Foundations & Interpretability

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Alignment Theory | Reward hacking, specification gaming (Goodhart's Law), corrigibility, mesa-optimisation | Demonstrate reward hacking in RL env; Constitutional AI critique-revision loop; corrigibility test suite |
| Mechanistic Interpretability | Activation patching, SAEs, circuit analysis, logit lens | Activation patching on GPT-2 small (factual recall); train small SAE, find 5 interpretable features; logit lens on factual recall |

**Resources:**
- Paper: *Constitutional AI* (Bai et al., Anthropic 2022); *Scaling Monosemanticity* (Anthropic 2024)
- Course: AISF Alignment course; Neel Nanda Mech Interp Tutorials
- Repo: `anthropics/anthropic-cookbook`; `TransformerLensOrg/TransformerLens`
- Blog: Anthropic Core Views on AI Safety; Transformer Circuits Thread

### Part 2 — Red Teaming & Defense

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Automated Red Teaming | Jailbreak taxonomy, refusal consistency, automated pipelines, OWASP LLM Top 10 | 200-prompt red-team pipeline across 5 categories; refusal consistency via 10 rephrasings; jailbreak output classifier |
| Defense-in-Depth | Input filtering, output classifier, runtime monitoring, watermarking | 3-layer defense (input filter + alignment + output classifier); bypass-all-3 demo; green-list watermarking |

**Resources:**
- Paper: *Red Teaming Language Models with Language Models* (Perez et al. 2022)
- Course: AISF Technical Safety module
- Repo: `NVIDIA/garak`
- Blog: OWASP LLM Top 10

### Part 3 — Governance & Reporting

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Governance Frameworks | Model cards, system cards, EU AI Act risk tiers, NIST AI RMF | Model card for fine-tuned model; system card for RAG assistant; EU AI Act risk-tier mapping |

**Resources:**
- Paper: *Model Cards for Model Reporting* (Mitchell et al. 2019)
- Course: Montreal AI Ethics Institute AI Governance
- Repo: `facebookresearch/ResponsibleAI`
- Blog: NIST AI Risk Management Framework

## Core Engineering Project
**Layered Safety Evaluation Suite** — see [`project/README.md`](project/README.md)

> Note: uses the **Module 3 fine-tuned model** as the target for red-teaming and the
> model card/system card.
