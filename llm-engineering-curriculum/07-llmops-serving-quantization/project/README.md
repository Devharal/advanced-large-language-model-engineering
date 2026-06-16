# Core Engineering Project — Production Inference Cluster

## Objective
Deploy a vLLM-based serving cluster with quantization and dynamic routing, fully
observable, load-tested to saturation.

## Deliverables
1. **vLLM deployment** with PagedAttention, continuous batching, prefix caching —
   serving a 7B + 70B model pair.
2. **Dynamic 3-tier router** (SLM/open/frontier) with query-complexity
   classification; routing decision logs.
3. **OpenTelemetry + Grafana dashboards**: TTFT, throughput, GPU utilization, queue
   depth.
4. **Load test to saturation**: reproduce and diagnose a KV-cache eviction storm;
   implement a prefix-cache routing fix.

## Acceptance Checklist
- [ ] vLLM serves both 7B and 70B models with PagedAttention enabled
- [ ] Prefix caching measurably reduces TTFT for shared system prompts
- [ ] Continuous batching verified to eliminate padding waste under mixed-length load
- [ ] Router correctly classifies and dispatches across 3 tiers; decisions logged
- [ ] Grafana dashboards show TTFT, throughput, GPU util, queue depth live
- [ ] Load test identifies saturation point at one of 10/50/100/200 req/s
- [ ] KV-cache eviction storm reproduced, diagnosed, and fixed via prefix-cache routing
- [ ] Autoscaling rule triggers correctly when p95 TTFT exceeds SLA

## Results
Place final report, dashboards exports, and load-test data in `results/`.
