# Core Engineering Project — Durable Multi-Agent Orchestration Grid

## Objective
Build a durable multi-agent orchestrator with checkpointing, failure recovery, and
full audit logging.

## Deliverables
1. **Durable orchestrator**: splits a research task into subtasks, dynamically
   spawns specialist worker agents, merges outputs.
2. **Per-step checkpointing**: kill an agent mid-task and verify automatic restart
   with state restoration.
3. **Deadlock detection** (dependency graph cycle detection) and **cascading
   failure containment** with local rollback.
4. **Full audit log**: all decisions, tool calls, handoffs, and timing — exported to
   structured JSON.

## Acceptance Checklist
- [ ] Orchestrator correctly decomposes task and merges worker outputs
- [ ] Checkpointing persists state at each step
- [ ] Mid-task kill + restart demonstrates correct state restoration
- [ ] Deadlock detector identifies cycles in agent dependency graph
- [ ] Cascading failure injected at one agent is contained via local rollback
- [ ] Audit log captures every decision, tool call, and handoff with timestamps
- [ ] Audit log exports as valid structured JSON

## Results
Place final report, audit log samples, and demo transcripts in `results/`.
