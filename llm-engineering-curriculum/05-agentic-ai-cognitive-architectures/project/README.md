# Core Engineering Project — Framework-Free Cyclic State-Machine Agent

## Objective
Build a thread-safe cyclic agent (Plan → Act → Observe → Critique → Plan) in pure
Python with real tools, HITL gating, and safety mechanisms — no LangChain/CrewAI.

## Deliverables
1. **Cyclic state-machine agent** (`src/cyclic_state_machine.py`) with explicit
   typed states for Plan/Act/Observe/Critique.
2. **3 real tools** integrated: web search, code executor, file system — with
   graceful error recovery on tool failure.
3. **HITL**: agent pauses at high-risk tool calls and resumes only after explicit
   human approval via CLI.
4. **Semantic loop detector + step-budget enforcer**: embed last N states, trigger
   fallback if cosine similarity > threshold.
5. **Prompt injection recovery demo**: injected malicious tool output is detected
   and the agent recovers.

## Acceptance Checklist
- [ ] State machine has explicit typed states and deterministic transitions
- [ ] All 3 tools integrated with documented error-recovery paths
- [ ] HITL interruption correctly pauses execution and resumes on approval
- [ ] Parallel tool execution via asyncio with no race conditions on shared state
- [ ] Semantic loop detector triggers correctly on injected repetitive states
- [ ] Step-budget enforcer caps runaway loops
- [ ] Prompt injection simulation is detected and agent recovers without crashing

## Results
Place final report, logs, and demo transcripts in `results/`.
