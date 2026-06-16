# Section A — Module Readiness Map

## Part A1 — Prerequisite Dependency Tree

- **STATISTICS FOUNDATIONS**
  - Mean, variance, std dev — population vs sample
  - Confidence intervals (what 95% CI means)
  - Hypothesis testing: null hypothesis, p-value, Type I/II errors
  - Bootstrap resampling
- **NLP METRICS HISTORY**
  - BLEU (n-gram precision); ROUGE (recall n-gram overlap)
  - Why these fail for open-ended generation (semantic equivalence problem)
- **ANNOTATION METHODOLOGY**
  - Likert scales, pairwise comparison
  - Inter-annotator disagreement

## YOU ARE NOW READY FOR
LLM-as-Judge → Bootstrap CI → Bias Audit → Contamination Scan → CI/CD Eval Pipeline

## Part A2 — Prerequisite Detail Table

| Concept | Why You Need It | Where to Learn It |
|---|---|---|
| Bootstrap resampling | All LLM metric CIs use this (non-parametric) | Efron & Hastie Ch.2; Khan Academy bootstrap |
| Cohen's Kappa | Validating human eval data | Wikipedia Cohen's Kappa; sklearn docs |
| BLEU/ROUGE limitations | Justifies LLM-as-Judge | Reiter 2018 'Validity of BLEU' |
| LLM API basics | LLM-as-Judge needs structured rubric calls | OpenAI/Anthropic API quickstarts |
| N-gram overlap computation | Contamination detection | Python NLTK n-gram tutorial |

## Checklist
- [ ] I can explain what a 95% CI actually means
- [ ] I understand bootstrap resampling intuitively
- [ ] I can compute Cohen's Kappa by hand on a small example
- [ ] I understand why BLEU/ROUGE fail for open-ended generation
- [ ] I've made a structured-rubric LLM API call before
