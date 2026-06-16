# Module 10 — Advanced Multi-Agent Systems and Durable Swarms

> Architect decentralized, asynchronous, infinite-horizon multi-agent networks with
> durable state and autonomous coordination.

## Status
- [ ] Readiness map reviewed (`notes/00-readiness-map.md`)
- [ ] Part 1 notes + implementations
- [ ] Part 2 notes + implementations
- [ ] Core Engineering Project complete

## Folder Structure
```
10-multi-agent-systems/
├── README.md
├── notes/
│   ├── 00-readiness-map.md
│   ├── 01-agent-topologies.md          # orchestrator-worker, hierarchical, p2p, durable
│   ├── 02-handoffs-context-isolation.md
│   ├── 03-cascading-failures-rollback.md
│   └── 04-dynamic-swarms-lifecycle.md
├── src/
│   ├── orchestrator_worker.py
│   ├── hierarchical_supervision.py     # 3-level hierarchy
│   ├── durable_agent.py                # checkpoint/restart
│   ├── handoff_compression.py          # 512-token state summary
│   ├── hmac_handoff_signatures.py
│   ├── memory_tiers.py                 # in-context + Chroma + Neo4j
│   ├── cascading_failure_sim.py
│   ├── deadlock_detector.py            # dependency graph cycle detection
│   ├── dynamic_swarm_spawner.py
│   └── audit_log.py
└── project/
    ├── README.md
    └── results/
```

## Topics & Resource Directory

### Part 1 — Multi-Agent Topologies

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Agent Topologies | Orchestrator-worker, hierarchical supervision, peer-to-peer, durable execution | Orchestrator-worker w/ parallel subtasks; 3-level hierarchy; convert stateless→durable agent |
| Agent Handoffs & Context Isolation | Context isolation, state compression, HMAC handoff signatures, 3-tier memory | Handoff w/ 512-token summary; HMAC signature verify/reject; 3-tier memory (in-context + Chroma + Neo4j) |

**Resources:**
- Paper: *AutoGen* (Wu et al. 2023)
- Course: DeepLearning.AI Multi-Agent Systems with AutoGen
- Repo: `microsoft/autogen`
- Blog: Lilian Weng (multi-agent section)

### Part 2 — Failure Recovery & Swarm Dynamics

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Cascading Failures & Rollback | Compounding error rates, checkpoint rollback, deadlock detection, error propagation | Inject error at agent 3 of 5, measure propagation; checkpoint-based rollback; deadlock detector via dependency graph |
| Dynamic Swarms & Lifecycle | Dynamic spawning, termination consensus, drift tracking, audit logging | Dynamic swarm per retrieved document + merge; consensus termination (3-agent majority vote); full audit log |

**Resources:**
- Paper: *MemGPT* (Packer et al. 2023); *ChatDev* (Qian et al. 2023)
- Course: CMU 15-440 Distributed Systems (fault tolerance)
- Repo: `temporalio/temporal`; `langchain-ai/langgraph`
- Blog: Swyx *Anatomy of Autonomy*; Inngest Durable Functions

## Core Engineering Project
**Durable Multi-Agent Orchestration Grid** — see [`project/README.md`](project/README.md)
