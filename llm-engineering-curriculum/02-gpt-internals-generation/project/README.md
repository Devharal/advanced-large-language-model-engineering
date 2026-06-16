# Core Engineering Project — Speculative Decoding Engine

## Objective
Build a full configurable autoregressive generation pipeline and a working
speculative decoding system with measured speedups.

## Deliverables
1. **Configurable sampler pipeline** (`src/sampling.py`): temperature, top-k, top-p,
   min-p, repetition penalty — composable logit processors.
2. **Speculative decoding implementation** (`src/speculative_decoding.py`): 7B
   verifier + 1B draft model; measure speedup across code, creative writing, and
   factual prompts.
3. **Acceptance rate vs draft length** visualisation (k=2,4,8).
4. **Compute/memory phase profile**: prefill vs decode regimes.
5. **Repetition loop detector** (`src/repetition_monitor.py`) with n-gram entropy
   monitoring and fallback strategy.

## Acceptance Checklist
- [ ] All 5 sampling strategies implemented and independently testable
- [ ] Logit processor pipeline applies strategies in documented, correct order
- [ ] Speculative decoding produces output distribution equivalent to target model
- [ ] Wall-clock speedup measured at k=2,4,8 across ≥3 prompt categories
- [ ] Acceptance rate vs theoretical speedup formula validated empirically
- [ ] Prefill (compute-bound) vs decode (memory-bound) phases clearly distinguished in profile
- [ ] Repetition loop induced, detected, and mitigated with fallback
- [ ] Coherence/perplexity degradation measured at 4K/16K/32K tokens

## Results
Place final report, plots, and tables in `results/`.
