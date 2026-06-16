# Module 5 — Agentic AI and Cognitive Architectures

> Design robust multi-turn reasoning loops using state machines, custom parsers, and
> safe execution boundaries.

## Status
- [ ] Readiness map reviewed (`notes/00-readiness-map.md`)
- [ ] Part 1 notes + implementations
- [ ] Part 2 notes + implementations
- [ ] Part 3 notes + implementations
- [ ] Core Engineering Project complete

## Folder Structure
```
05-agentic-ai-cognitive-architectures/
├── README.md
├── notes/
│   ├── 00-readiness-map.md
│   ├── 01-reasoning-patterns-react-reflexion-tot.md
│   ├── 02-state-machines-graphs.md
│   ├── 03-memory-context-management.md
│   ├── 04-tool-calling-hitl.md
│   └── 05-safety-failure-recovery.md
├── src/
│   ├── react_reflexion.py
│   ├── tree_of_thoughts.py
│   ├── cyclic_state_machine.py   # Plan->Act->Observe->Critique
│   ├── memory_strategies.py      # in-context/summary/sliding-window/external
│   ├── checkpoint_resume.py
│   ├── tool_call_parser.py
│   ├── hitl_interrupt.py
│   ├── prompt_injection_guard.py
│   └── semantic_loop_detector.py
├── notebooks/
│   └── reasoning_pattern_ablation.ipynb
└── project/
    ├── README.md
    └── results/
```

## Topics & Resource Directory

### Part 1 — Reasoning Frameworks

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Prompting & Reasoning Patterns | ReAct, Plan-and-Solve, Reflexion, Tree-of-Thoughts, Self-Discover | ReAct + Reflexion from scratch; 3-branch ToT search; ablate on GAIA/AgentBench |

**Resources:**
- Paper: *ReAct* (Yao et al. 2022); *Reflexion* (Shinn et al. 2023)
- Course: Stanford CS25 V4 LLM Agents (Shunyu Yao)
- Repo: `langchain-ai/langchain`
- Blog: Lilian Weng *LLM-Powered Autonomous Agents*

### Part 2 — State & Memory Management

| Topic | Key Concepts | What to Implement |
|---|---|---|
| State Machines & Graphs | Deterministic FSM, DAG vs cyclic graphs, LangGraph | Cyclic state-machine agent, no framework; loop detection (max_steps + similarity) |
| Memory & Context Management | In-context/summary/sliding-window/external memory, checkpointing | Benchmark 3 memory strategies; vector-based episodic memory; thread-safe checkpoint |

**Resources:**
- Paper: *MemGPT* (Packer et al. 2023)
- Course: DeepLearning.AI AI Agents in LangGraph
- Repo: `langchain-ai/langgraph`; `cpacker/MemGPT`
- Blog: LangChain blog; Harrison Chase newsletter

### Part 3 — Tool Use, HITL & Safety

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Tool Calling & HITL | JSON schema tools, function-call parser, HITL interrupt/resume | Robust tool-call parser; HITL async resume; parallel tool execution w/ asyncio |
| Safety & Failure Recovery | Prompt injection, context poisoning, loop detection, race conditions | Simulate prompt injection + sanitisation; semantic loop detector; deterministic fallback |

**Resources:**
- Paper: *Prompt Injection Attacks Against LLM-Integrated Applications* (Greshake et al. 2023)
- Course: OWASP LLM Top 10
- Repo: `protectai/rebuff`
- Blog: Simon Willison Prompt Injection

## Core Engineering Project
**Framework-Free Cyclic State-Machine Agent** — see [`project/README.md`](project/README.md)
