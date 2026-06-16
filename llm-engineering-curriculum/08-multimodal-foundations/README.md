# Module 8 — Multimodal Foundations and Cross-Modal Alignment

> Unify textual semantic spaces with visual and audio temporal arrays using shared
> representational topologies.

## Status
- [ ] Readiness map reviewed (`notes/00-readiness-map.md`)
- [ ] Part 1 notes + implementations
- [ ] Part 2 notes + implementations
- [ ] Part 3 notes + implementations
- [ ] Core Engineering Project complete

## Folder Structure
```
08-multimodal-foundations/
├── README.md
├── notes/
│   ├── 00-readiness-map.md
│   ├── 01-vit-clip.md
│   ├── 02-early-late-fusion.md
│   ├── 03-video-audio-encoding.md
│   └── 04-inter-modal-interference.md
├── src/
│   ├── vit_patch_embedding.py
│   ├── clip_contrastive.py
│   ├── perceiver_resampler.py
│   ├── late_fusion_projector.py   # LLaVA-style MLP projector
│   ├── anyres_tiling.py
│   ├── video_audio_encoder.py     # mel-spectrogram tokenization
│   └── mixed_modality_batching.py
├── notebooks/
│   ├── cifar10_vit_classification.ipynb
│   ├── clip_small_dataset_training.ipynb
│   └── mmlu_humaneval_regression.ipynb
└── project/
    ├── README.md
    └── results/
```

## Topics & Resource Directory

### Part 1 — Visual Foundations

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Vision Transformers & CLIP | ViT patches, CLIP contrastive loss, Perceiver Resampler | ViT patch embed + class token on CIFAR-10; small CLIP contrastive training; Perceiver 256→64 compression |
| Early vs Late Fusion | Early (any-to-any) vs late fusion, MLP projector, AnyRes tiling | Late-fusion MLP projector (frozen ViT+LLM); fine-tune projector, CIDEr eval; AnyRes on 4K images |

**Resources:**
- Paper: *CLIP* (Radford et al. 2021); *LLaVA — Visual Instruction Tuning* (Liu et al. 2023)
- Course: CMU 11-777 Multimodal ML
- Repo: `openai/CLIP`; `haotian-liu/LLaVA`
- Blog: Lilian Weng Contrastive Representation Learning; Sebastian Raschka Multimodal LLMs

### Part 2 — Video & Audio

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Video & Audio Encoding | Spatio-temporal attention, frame sampling, mel-spectrogram, cross-modal contrastive | Frame-sampled video encoder; mel-spectrogram tokenizer; audio-visual contrastive alignment |

**Resources:**
- Paper: *ImageBind* (Girdhar et al. 2023)
- Course: Stanford CS25 V3 Multimodal Foundation Models
- Repo: `facebookresearch/ImageBind`
- Blog: The Gradient — VLMs in 2024

### Part 3 — Production Challenges

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Inter-Modal Interference & Scaling | Modal interference regression, context-scaling bottleneck, mixed-modality batching | Fine-tune on images only, measure MMLU regression; mixed-modality dynamic batching; memory profile 1 vs 4 images |

## Core Engineering Project
**Late-Fusion Vision-Language Adapter** — see [`project/README.md`](project/README.md)
