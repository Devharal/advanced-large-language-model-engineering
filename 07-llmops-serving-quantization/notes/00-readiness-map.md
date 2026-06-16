# Section A — Module Readiness Map

## Part A1 — Prerequisite Dependency Tree

- **GPU & SYSTEMS FUNDAMENTALS**
  - CUDA: threads, warps, blocks, grids
  - GPU memory: HBM vs SRAM latency hierarchy
  - FLOP counting for matmul at given precision
- **DISTRIBUTED SYSTEMS CONCEPTS**
  - Process groups, all-reduce
  - NCCL transport layer
  - Ring all-reduce
- **NUMERICAL PRECISION**
  - IEEE 754 FP32 structure
  - FP16 overflow / NaN causes
- **QUANTIZATION BASICS**
  - INT8 absmax scaling, zero-point calibration
  - Weight-only vs activation quantization

## YOU ARE NOW READY FOR
FSDP/ZeRO → GPTQ/AWQ/GGUF → vLLM/PagedAttention → Observability Stack

## Part A2 — Prerequisite Detail Table

| Concept | Why You Need It | Where to Learn It |
|---|---|---|
| CUDA programming model | FlashAttention/GPTQ/vLLM kernels are CUDA | NVIDIA CUDA by Example; Fast.ai GPU module |
| All-reduce / ring topology | DDP/FSDP/ZeRO communication cost | CMU 11-868 slides; NCCL docs |
| FP16/BF16 formats | Mixed precision, loss scaling, FP8 | Module 3 prereqs; NVIDIA mixed precision guide |
| Naive LLM serving | Baseline for PagedAttention/continuous batching | Lilian Weng inference optimisation |
| Linux system monitoring | GPU profiling, NCCL debugging, OOM analysis | nvidia-smi/nvtop/py-spy docs |

## Checklist
- [ ] I understand the CUDA threads/warps/blocks/grids model at a high level
- [ ] I can explain all-reduce and why it's needed for gradient sync
- [ ] I understand FP16's 5-exponent-bit overflow problem
- [ ] I know what a naive (unbatched) LLM serving loop looks like
- [ ] I'm comfortable with nvidia-smi / nvtop for GPU monitoring
