# Core Engineering Project — End-to-End Alignment Pipeline

## Objective
Load a 7B model in NF4 via QLoRA, train a custom LoRA adapter with DPO on a packed
preference dataset, and compare across PEFT and alignment variants.

## Deliverables
1. **QLoRA setup** (`src/qlora_setup.py`): NF4 + double quant + BF16 compute,
   custom LoRA adapter injection.
2. **DPO training** on a packed preference dataset (`(prompt, chosen, rejected)`).
3. **Comparison tables**:
   - LoRA vs DoRA at same rank
   - DPO vs KTO vs ORPO on a noisy-label ablation
4. **Training instrumentation**: chosen/rejected log-probs, KL from reference,
   gradient norms, MMLU score every 100 steps.
5. **Merged adapter deployment**: verify zero added latency vs base model on
   identical prompts.

## Acceptance Checklist
- [ ] Model loads correctly in NF4 with double quantization, no NaN loss
- [ ] LoRA adapter injected on documented target modules (Q,K,V,O,FFN)
- [ ] DPO training converges; chosen/rejected log-prob divergence tracked
- [ ] Length-bias in DPO measured and reported
- [ ] LoRA vs DoRA table includes parameter counts and benchmark scores
- [ ] DPO vs KTO vs ORPO table includes noisy-label robustness results
- [ ] KL-from-reference, gradient norms, MMLU logged every 100 steps (plots included)
- [ ] Merged adapter verified to add zero inference latency

## Results
Place final report, plots, and tables in `results/`. The final merged model/adapter
checkpoint reference should be documented here for reuse in Module 12.
