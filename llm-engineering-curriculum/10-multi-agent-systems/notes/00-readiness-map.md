# Section A — Module Readiness Map

## Part A1 — Prerequisite Dependency Tree

- **FROM MODULES 5 & 9 (required)**
  - Single-agent state machines and cyclic loops
  - Tool calling and error recovery patterns
- **DISTRIBUTED SYSTEMS THEORY**
  - CAP theorem
  - Idempotency
  - Eventual consistency
- **CONCURRENCY PRIMITIVES**
  - Deadlock; banker's algorithm intuition
  - Race conditions
  - asyncio.gather / task groups
- **WORKFLOW ORCHESTRATION CONCEPTS**
  - DAG-based orchestration (Airflow/Prefect)
  - Checkpoint and replay

## YOU ARE NOW READY FOR
Topologies → Handoffs → Context Isolation → Cascading Failures → Dynamic Swarms

## Part A2 — Prerequisite Detail Table

| Concept | Why You Need It | Where to Learn It |
|---|---|---|
| Deadlock detection algorithms | Multi-agent dependency deadlocks | CLRS Ch.22; OS textbook deadlock chapter |
| DAG orchestration (Airflow/Prefect) | Multi-agent topologies are workflow DAGs | Airflow Getting Started; Prefect quickstart |
| Idempotency | Durable checkpointing/retry requires this | Stripe Idempotency Keys blog |
| HMAC / crypto hashing | Zero-trust handoff signatures | Python hmac docs; Dan Boneh Crypto I (Wk2) |
| Module 5 project completion | Multi-agent extends single-agent patterns | Build/debug Module 5 agent first |

## Checklist
- [ ] I can explain CAP theorem and idempotency in my own words
- [ ] I understand deadlock detection via dependency cycles
- [ ] I understand DAG-based orchestration (Airflow/Prefect basics)
- [ ] I understand HMAC and message authentication
- [ ] I have a working single-agent cyclic system from Module 5
