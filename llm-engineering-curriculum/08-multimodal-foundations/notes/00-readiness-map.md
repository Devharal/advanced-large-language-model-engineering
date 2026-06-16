# Section A — Module Readiness Map

## Part A1 — Prerequisite Dependency Tree

- **FROM MODULE 1 (required)**
  - Transformer architecture: patches as tokens (visual tokenization)
  - Attention: image patches attending to each other
- **COMPUTER VISION BASICS**
  - Image tensors [B, C, H, W]; convolutions (conceptual)
  - Image classification: softmax over class logits
- **CONTRASTIVE LEARNING**
  - Contrastive loss: pull similar, push dissimilar
  - SimCLR/MoCo self-supervised representation learning
  - Shared embedding space
- **AUDIO SIGNAL PROCESSING**
  - Waveform (amplitude vs time)
  - Mel-spectrogram via STFT

## YOU ARE NOW READY FOR
ViT → CLIP → Early/Late Fusion → Video/Audio Encoding → Cross-Modal Alignment

## Part A2 — Prerequisite Detail Table

| Concept | Why You Need It | Where to Learn It |
|---|---|---|
| Image tensor [B,C,H,W] | ViT patch embedding requires this reshape | PyTorch tensors tutorial; Fast.ai Lesson 1 |
| Contrastive learning (SimCLR) | CLIP = contrastive learning on image-text pairs | SimCLR paper; Lilian Weng Contrastive Rep Learning |
| Mel-spectrogram computation | Audio tokenization pipeline | librosa tutorial; 3Blue1Brown Fourier Transform |
| Cross-attention | Perceiver Resampler / late-fusion projectors | Module 1 prereq; Illustrated Transformer |
| Transfer learning / frozen encoders | Late-fusion VLMs freeze both encoders | HF Transfer learning tutorial; Fast.ai Lesson 2 |

## Checklist
- [ ] I understand how an image becomes a tensor [B,C,H,W]
- [ ] I can explain contrastive loss (pull/push) conceptually
- [ ] I understand mel-spectrograms at a high level
- [ ] I understand cross-attention (Q from one source, K/V from another)
- [ ] I understand what "freezing" parameters means and why late-fusion uses it
