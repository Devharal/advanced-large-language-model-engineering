# Module 7 — LLMOps, Serving Infrastructure, and Quantization Physics

> Deploy, optimize, and scale large open-source models under strict hardware memory
> constraints and high-concurrency requirements.

## Status
- [ ] Readiness map reviewed (`notes/00-readiness-map.md`)
- [ ] Part 1 notes + implementations
- [ ] Part 2 notes + implementations
- [ ] Part 3 notes + implementations
- [ ] Core Engineering Project complete

## Folder Structure
```
07-llmops-serving-quantization/
├── README.md
├── notes/
│   ├── 00-readiness-map.md
│   ├── 01-parallelism-strategies.md     # DP/TP/PP/ZeRO/FSDP
│   ├── 02-post-training-quantization.md # GPTQ/AWQ/GGUF
│   ├── 03-vllm-pagedattention.md
│   └── 04-observability-cost.md
├── src/
│   ├── fsdp_training.py
│   ├── deepspeed_zero_configs.py
│   ├── tp_dp_2d_parallel.py
│   ├── gptq_awq_quantize.py
│   ├── gguf_convert_benchmark.py
│   ├── vllm_deploy.py
│   ├── dynamic_router.py        # 3-tier SLM/open/frontier router
│   └── otel_instrumentation.py
├── configs/
│   ├── deepspeed_zero3.json
│   ├── fsdp_config.yaml
│   └── vllm_server_config.yaml
├── dashboards/
│   └── grafana_ttft_throughput.json
└── project/
    ├── README.md
    └── results/
```

## Topics & Resource Directory

### Part 1 — Distributed Training

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Parallelism Strategies | DP, TP, PP, ZeRO, ZeRO-3/FSDP | FSDP on 2+ GPUs vs DDP; ZeRO-2/3 memory profiling; 2D (TP+DP) on 4-GPU node |

**Resources:**
- Paper: *ZeRO* (Rajbhandari et al. 2019)
- Course: CMU 11-868 Distributed Training
- Repo: `microsoft/DeepSpeed`
- Blog: Lilian Weng Large-Scale Distributed Training

### Part 2 — Quantization

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Post-Training Quantization | INT8/INT4 PTQ, GPTQ, AWQ, GGUF K-quants | Quantize 7B w/ GPTQ + AWQ, perplexity on WikiText-103; GGUF Q4_K_M CPU benchmark; memory profile FP16/INT8/INT4 |

**Resources:**
- Paper: *GPTQ* (Frantar et al. 2022)
- Course: MIT EfficientML.ai Quantization
- Repo: `ggerganov/llama.cpp`
- Blog: Tim Dettmers Overview of Quantization Methods

### Part 3 — Serving Infrastructure

| Topic | Key Concepts | What to Implement |
|---|---|---|
| vLLM & PagedAttention | Paged KV cache, continuous batching, prefix caching, dynamic routing | Deploy vLLM w/ paged attention + prefix caching; dynamic 3-tier router; OpenTelemetry tracing |
| Observability & Cost | TTFT, throughput, distributed tracing, cold-start | Grafana TTFT/throughput dashboards; load test 10/50/100/200 req/s; autoscaling rule on p95 TTFT |

**Resources:**
- Paper: *Efficient Memory Management with PagedAttention* (Kwon et al. 2023); *Orca* (Yu et al. 2022)
- Course: CMU 11-868 LLM Serving Systems; Stanford CS329S
- Repo: `vllm-project/vllm`; `open-telemetry/opentelemetry-python`
- Blog: vLLM Blog 2024 Retrospective; Chip Huyen Designing ML Systems

## Core Engineering Project
**Production Inference Cluster** — see [`project/README.md`](project/README.md)
