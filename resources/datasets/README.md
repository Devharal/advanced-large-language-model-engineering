# Dataset Notes

Track datasets used across modules: source, license, size, and prep scripts.
Do not commit raw data — it's gitignored. Document download/prep steps instead.

## Format
```markdown
## <Dataset Name>
- **Used in:** Module 0X
- **Source:** <url>
- **License:** <license>
- **Size:** <approx size>
- **Prep:** `scripts/prep_<dataset>.py` — what it does
- **Notes:** anything unusual (contamination risk, license restrictions, etc.)
```

## Datasets Likely Needed (by module)

| Module | Datasets |
|---|---|
| 02 | Small text corpus for BPE training; WikiText for fertility-rate measurement |
| 03 | Preference dataset (chosen/rejected pairs) for DPO/KTO/ORPO; instruction-tuning corpus |
| 04 | Document corpus for hybrid/GraphRAG indexing; MMLU/GSM8K (contamination scan target) |
| 06 | QA test set for metric comparison; MMLU/GSM8K for contamination scans |
| 07 | WikiText-103 for quantization perplexity evaluation |
| 08 | CIFAR-10 (ViT classification); small image-caption pairs (CLIP); LLaVA-style captioning data; VQAv2/COCO |
| 11 | GSM8K, MATH, HumanEval |
| 12 | PRM800K-style step-labeled data; red-team prompt seed sets |
