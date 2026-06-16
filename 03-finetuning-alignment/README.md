# Module 3 — Advanced Fine-Tuning and Alignment Paradigms

> Design and execute data-packed, parameter-efficient fine-tuning pipelines and
> mathematically grounded alignment routines.

## Status
- [ ] Readiness map reviewed (`notes/00-readiness-map.md`)
- [ ] Part 1 notes + implementations
- [ ] Part 2 notes + implementations
- [ ] Part 3 notes + implementations
- [ ] Core Engineering Project complete

## Folder Structure
```
03-finetuning-alignment/
├── README.md
├── notes/
│   ├── 00-readiness-map.md
│   ├── 01-lora.md
│   ├── 02-qlora.md
│   ├── 03-dora.md
│   ├── 04-rlhf-ppo.md
│   ├── 05-dpo.md
│   ├── 06-kto-orpo.md
│   ├── 07-mixed-precision-gradients.md
│   └── 08-dataset-packing-quality.md
├── src/
│   ├── lora_manual.py          # manual LoRA injection (no PEFT)
│   ├── qlora_setup.py
│   ├── dora_config.py
│   ├── reward_model.py
│   ├── ppo_loop.py
│   ├── dpo_manual.py
│   ├── kto_orpo_trainers.py
│   ├── amp_training_loop.py
│   └── packed_dataset.py
├── notebooks/
│   ├── lora_vs_dora_ablation.ipynb
│   ├── dpo_kto_orpo_noisy_labels.ipynb
│   ├── gradient_norm_pre_post_ln.ipynb
│   └── nan_loss_repro_fp16.ipynb
├── configs/
│   ├── lora_config.yaml
│   ├── qlora_bnb_config.yaml
│   └── dpo_config.yaml
└── project/
    ├── README.md
    └── results/
```

## Topics & Resource Directory

### Part 1 — Parameter-Efficient Fine-Tuning (PEFT)

| Topic | Key Concepts | What to Implement |
|---|---|---|
| LoRA | ΔW=BA low-rank decomposition, rank/α, target modules, zero-latency merge | Manual LoRA injection; PEFT LoraConfig; merge & ONNX export |
| QLoRA | NF4, double quantization, paged optimizers, BF16 compute / NF4 storage | BitsAndBytesConfig load; `prepare_model_for_kbit_training`; 7B on RTX 3090 |
| DoRA | Magnitude-direction decomposition, output calibration | `use_dora=True`; DoRA vs LoRA benchmark at equal params |

**Resources:**
- Paper: *LoRA* (Hu et al. 2021); *QLoRA* (Dettmers et al. 2023)
- Course: CMU 11-711 PEFT lecture
- Repo: `huggingface/peft`; `artidoro/qlora`
- Blog: Sebastian Raschka LoRA from scratch; Tim Dettmers blog

### Part 2 — Alignment Paradigms

| Topic | Key Concepts | What to Implement |
|---|---|---|
| RLHF + PPO | SFT→RM→RL pipeline, Bradley-Terry, PPO KL penalty, 4-model memory | Reward head training; TRL PPOTrainer loop; reward hacking monitoring |
| DPO | Implicit reward, 2-model memory, length bias | Preference dataset; TRL DPOTrainer; manual DPO loss |
| KTO & ORPO | Binary KTO signal, ORPO single-objective (no ref model) | TRL KTOTrainer; ORPO with MMLU forgetting check; noisy-label comparison |

**Resources:**
- Paper: *Direct Preference Optimization* (Rafailov et al. 2023)
- Course: Stanford CS224N Instruction Tuning & RLHF
- Repo: `huggingface/trl`
- Blog: Lilian Weng *Prompt Engineering, Fine-tuning & Alignment*

### Part 3 — Training Engineering

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Mixed Precision & Gradient Management | FP32/FP16/BF16/FP8, loss scaling, gradient checkpointing/accumulation | AMP training loop; NaN repro from FP16 overflow + fix |
| Dataset Packing & Quality | Padding waste, packing/cross-sample masking, template leakage | ConstantLengthDataset; n-gram contamination scan; template leakage detection |

**Resources:**
- Paper: *Mixed Precision Training* (Micikevicius et al. 2017)
- Course: CMU 11-868 Training Engineering module
- Repo: `axolotl-ai-cloud/axolotl`
- Blog: Maxime Labonne LLM Training & Fine-tuning Substack

## Core Engineering Project
**End-to-End Alignment Pipeline** — see [`project/README.md`](project/README.md)

> Note: the model produced here is reused in Module 12 (model card / system card /
> red-team target).
