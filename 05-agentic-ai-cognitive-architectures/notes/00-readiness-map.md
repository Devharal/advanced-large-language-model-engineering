# Section A — Module Readiness Map

## Part A1 — Prerequisite Dependency Tree

- **FROM MODULES 1-4 (required)**
  - LLM generation and tool-use via structured prompting
  - RAG pipelines for agent memory and retrieval
- **PROMPTING FOUNDATIONS**
  - Chain-of-Thought (CoT)
  - Few-shot prompting
  - Structured output prompting (JSON/XML/schema)
- **TOOL USE BASICS**
  - Function calling (JSON schema)
  - Anthropic/OpenAI tool use API cycle
  - Error handling for tool calls
- **AGENT LOOP CONCEPT**
  - Observe-Think-Act loop
  - Context accumulation / window exhaustion

## YOU ARE NOW READY FOR
ReAct/Reflexion → State Machines → Memory Tiers → Tool Parsing → HITL → Safety

## Part A2 — Prerequisite Detail Table

| Concept | Why You Need It | Where to Learn It |
|---|---|---|
| Chain-of-Thought prompting | ReAct/Reflexion/ToT are extensions of CoT | Wei et al. 2022; Anthropic prompting guide |
| Function/tool calling API | Foundation of all agent tool use | Anthropic tool use docs; OpenAI function calling cookbook |
| Finite state machines (FSM) | Agent state graphs are FSMs | Any FSM tutorial; Wikipedia Finite Automata |
| Async Python (asyncio) | Parallel tool execution, HITL resume | Real Python Async IO; Python asyncio docs |
| Context window management | Long-horizon agents fail due to exhaustion | Lilian Weng LLM-Powered Autonomous Agents |

## Checklist
- [ ] I can write a CoT prompt and explain why it improves multi-step accuracy
- [ ] I've used a tool-call API (Anthropic/OpenAI) hands-on
- [ ] I understand FSM states/transitions
- [ ] I'm comfortable with asyncio.gather and async/await
- [ ] I understand why agents run out of context on long tasks
