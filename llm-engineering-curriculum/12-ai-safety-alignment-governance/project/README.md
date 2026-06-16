# Core Engineering Project — Layered Safety Evaluation Suite

## Objective
Build a complete safety evaluation and documentation suite for the Module 3
fine-tuned model.

## Deliverables
1. **Automated red-team pipeline**: 200 adversarial prompts across 5 OWASP LLM
   Top 10 categories; measure attack success rate.
2. **3-layer defense**: input classifier + aligned LLM + output classifier; find
   one prompt that bypasses all 3 layers.
3. **Green-list watermarking**: verify detection rate on 500 generated samples at
   100, 200, and 500 token lengths.
4. **Model card + system card**: for the Module 3 fine-tuned model, mapped to EU AI
   Act risk tier with justification.

## Acceptance Checklist
- [ ] 200 prompts generated across all 5 chosen OWASP categories
- [ ] Attack success rate computed and reported per category
- [ ] 3-layer defense implemented; each layer independently testable
- [ ] At least one bypass-all-3 prompt found and documented
- [ ] Watermark detection rate reported at 100/200/500 tokens
- [ ] Model card includes evaluation results and known failure modes
- [ ] System card documents retrieval/agent failure modes and mitigations (if applicable)
- [ ] EU AI Act risk tier mapping includes explicit justification

## Results
Place final report, model card, system card, and red-team logs in `results/`.
