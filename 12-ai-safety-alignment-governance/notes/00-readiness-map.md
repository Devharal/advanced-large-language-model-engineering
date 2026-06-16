# Section A — Module Readiness Map

## Part A1 — Prerequisite Dependency Tree

- **FROM ALL MODULES (integration module)**
  - Fine-tuning, alignment (RLHF/DPO), reward modelling (Module 3)
  - Evaluation pipelines and red-teaming methodology (Module 6)
  - Production deployment and observability (Module 7)
- **ALIGNMENT THEORY FOUNDATIONS**
  - Goodhart's Law
  - Reward hacking taxonomy: specification gaming, wireheading, goal misgeneralisation
  - OWASP LLM Top 10
- **INTERPRETABILITY BASICS**
  - Activation analysis
  - Attention visualisation
  - Probing classifiers
- **GOVERNANCE & REGULATION**
  - EU AI Act risk tiers (unacceptable/high/limited/minimal)
  - NIST AI RMF: Govern, Map, Measure, Manage

## YOU ARE NOW READY FOR
Alignment Theory → SAE Interpretability → Automated Red-Teaming → Defence-in-Depth → Governance

## Part A2 — Prerequisite Detail Table

| Concept | Why You Need It | Where to Learn It |
|---|---|---|
| Goodhart's Law / reward hacking | Core framing of the alignment problem | Russell "Human Compatible" Ch.5; DeepMind "Specification Gaming" |
| OWASP LLM Top 10 | Red-team taxonomy follows this | owasp.org LLM Top 10 |
| Activation analysis / probing | Simpler precursor to SAEs/activation patching | Neel Nanda Mech Interp Explainer |
| EU AI Act risk tiers | Required for compliant system cards | EU AI Act full text; NIST AI RMF quickstart |
| Model cards (Mitchell et al. 2019) | Standard format for Module 12 deliverable | Mitchell et al. 2019 arXiv:1810.03993 |

## Checklist
- [ ] I can explain Goodhart's Law in the context of reward models
- [ ] I'm familiar with the OWASP LLM Top 10 categories
- [ ] I understand activation analysis and probing classifiers conceptually
- [ ] I can map a system to an EU AI Act risk tier
- [ ] I've read the Model Cards paper (Mitchell et al. 2019)
- [ ] Modules 3, 6, and 7 are complete (integration prerequisite)
