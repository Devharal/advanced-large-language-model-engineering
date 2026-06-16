# Core Engineering Project — Late-Fusion Vision-Language Adapter

## Objective
Freeze a pre-trained ViT-L and a 7B LLM, train a custom 2-layer MLP projector on
LLaVA-style data, and evaluate end-to-end.

## Deliverables
1. **2-layer MLP projector** connecting frozen ViT-L to frozen 7B LLM, trained on
   LLaVA-style captioning data.
2. **AnyRes tiling** for high-resolution input; evaluation on VQAv2 and COCO
   captioning (CIDEr score).
3. **Inter-modal interference study**: MMLU and HumanEval before/after projector
   training.
4. **Mixed-modality dynamic batching pipeline**: throughput profile for 1/2/4-image
   batch sizes.

## Acceptance Checklist
- [ ] ViT-L and LLM remain frozen; only projector parameters trained
- [ ] Projector training converges; CIDEr score reported on COCO captioning
- [ ] VQAv2 evaluation completed with documented accuracy
- [ ] AnyRes tiling implemented and shown to improve quality vs fixed-resolution
- [ ] MMLU/HumanEval measured before and after — regression quantified
- [ ] Mixed-modality batching pipeline handles heterogeneous tensor shapes
- [ ] Throughput profiled at 1/2/4-image batch sizes with memory bottleneck identified

## Results
Place final report, plots, and tables in `results/`.
