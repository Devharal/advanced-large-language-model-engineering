# Core Engineering Project — Automated Evaluation Pipeline

## Objective
Build a statistically rigorous, two-tier CI/CD evaluation system for non-deterministic
LLM outputs.

## Deliverables
1. **Two-tier CI/CD eval system**: fast deterministic tests (<30s) + batch
   LLM-as-a-Judge with position-swap debiasing.
2. **Bootstrap 95% CI + Fleiss' Kappa**: report all metrics with intervals — never
   single-point scores.
3. **Contamination scan**: a fine-tuned model's training data vs MMLU and GSM8K;
   report overlap percentage.
4. **Evol-Instruct**: generate 500 synthetic test cases from 50 seed examples;
   validate quality with human spot-check on a random 10% sample.

## Acceptance Checklist
- [ ] Fast tier (regex/JSON schema/exact match) runs in <30s on every commit
- [ ] Slow tier (LLM-as-Judge/human eval) runs on release branches
- [ ] Position-swap debiasing implemented and flip-rate measured
- [ ] All reported metrics include bootstrap 95% CI — no bare point estimates
- [ ] Fleiss' Kappa computed for 3-annotator labels; low-agreement examples flagged
- [ ] Contamination scan reports concrete overlap % against MMLU and GSM8K
- [ ] 500 Evol-Instruct cases generated; 10% human spot-check quality report included
- [ ] Regression alert triggers correctly on a >2σ metric drop (demonstrated)

## Results
Place final report, plots, and tables in `results/`.
