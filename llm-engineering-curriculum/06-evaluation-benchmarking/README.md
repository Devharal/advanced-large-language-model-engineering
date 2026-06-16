# Module 6 — Enterprise Evaluation and Scientific Benchmarking

> Establish rigorous, statistically sound, and automated verification frameworks for
> non-deterministic AI systems.

## Status
- [ ] Readiness map reviewed (`notes/00-readiness-map.md`)
- [ ] Part 1 notes + implementations
- [ ] Part 2 notes + implementations
- [ ] Part 3 notes + implementations
- [ ] Core Engineering Project complete

## Folder Structure
```
06-evaluation-benchmarking/
├── README.md
├── notes/
│   ├── 00-readiness-map.md
│   ├── 01-automated-metrics.md          # EM, BLEU/ROUGE, BERTScore, G-Eval, RAGAS
│   ├── 02-statistical-significance.md   # bootstrap CI, Kappa, multiple comparison
│   ├── 03-llm-as-judge-biases.md
│   └── 04-cicd-eval-pipelines.md
├── src/
│   ├── metrics.py             # EM, ROUGE-L, BERTScore, G-Eval
│   ├── llm_as_judge.py        # position-swap debiasing, verbosity normalisation
│   ├── bootstrap_ci.py
│   ├── kappa.py                # Cohen's / Fleiss' Kappa
│   ├── contamination_scan.py   # n-gram overlap vs MMLU/GSM8K
│   ├── evol_instruct.py        # synthetic test case generation
│   └── eval_pipeline.py        # two-tier CI/CD
├── notebooks/
│   └── metric_ranking_comparison.ipynb
└── project/
    ├── README.md
    └── results/
```

## Topics & Resource Directory

### Part 1 — Evaluation Metrics

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Automated Metrics | EM, BLEU/ROUGE, BERTScore, G-Eval/LLM-as-Judge, RAGAS triad | Compute all on same QA set; LLM-as-Judge w/ rubric; RAGAS as CI/CD gate |

**Resources:**
- Paper: *Judging LLM-as-a-Judge with MT-Bench* (Zheng et al. 2023); *RAGAS* (Es et al. 2023)
- Course: Hamel Husain Mastering LLMs — Evaluation
- Repo: `BerriAI/litellm`; `confident-ai/deepeval`
- Blog: Eugene Yan; Jason Liu Substack

### Part 2 — Statistical Rigour

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Statistical Significance | Bootstrap CIs, Cohen's/Fleiss' Kappa, Bonferroni/BH correction | Bootstrap 95% CI on ROUGE/BERTScore; Fleiss' Kappa calculator; Bonferroni significance tests |
| LLM-as-Judge Biases | Position bias, verbosity bias, self-enhancement bias, contamination | Reproduce position bias; contamination scan; debiased judge (position swap + length norm) |

**Resources:**
- Paper: *Beyond Accuracy: Behavioral Testing of NLP with CheckList* (Ribeiro et al. 2020)
- Course: CMU 11-711 Evaluation Methodology
- Repo: `Arize-AI/phoenix`
- Blog: Chip Huyen Substack

### Part 3 — CI/CD Evaluation Pipelines

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Automated Evaluation Pipelines | Fast deterministic vs slow stochastic tiers, regression alerts, Evol-Instruct | Two-tier CI/CD; 500 synthetic cases from 50 seeds; regression alerts at >2σ |

## Core Engineering Project
**Automated Evaluation Pipeline** — see [`project/README.md`](project/README.md)

> Note: this module's methodology (bootstrap CI, Fleiss' Kappa, contamination scans,
> LLM-as-Judge debiasing) is a **prerequisite methodology** for Modules 3, 11, 12,
> and the Capstone Milestone 3.
