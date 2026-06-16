# Core Engineering Project — Test-Time Compute Scaling Harness

## Objective
Build a harness that benchmarks a 7B model under multiple test-time compute
strategies and produces a compute-accuracy Pareto frontier.

## Deliverables
1. **Benchmark 3 strategies**: single-pass, Best-of-N (N=1,4,16,64) with ORM, and
   iterative self-refinement.
2. **PRM-guided beam search**: train a PRM on a math reasoning dataset; compare vs
   ORM-guided Best-of-N.
3. **Compute-accuracy Pareto frontier** with 95% bootstrap CI; identify the
   inflection point of diminishing returns.
4. **Error accumulation demo**: a 5-step agentic task where test-time scaling
   improves step-1 accuracy but increases overall chain failure rate.

## Acceptance Checklist
- [ ] Single-pass, Best-of-N (4 values of N), and self-refinement all benchmarked
- [ ] PRM trained with step-level correctness labels
- [ ] PRM-guided beam search vs ORM-guided Best-of-N compared on same benchmark
- [ ] Pareto frontier plotted with bootstrap 95% CI bands
- [ ] Inflection point of diminishing returns explicitly identified
- [ ] 5-step agentic task demonstrates error accumulation despite step-1 improvement
- [ ] Dynamic router (3B vs 70B) implemented and evaluated on cost/accuracy tradeoff

## Results
Place final report, plots, and tables in `results/`.
