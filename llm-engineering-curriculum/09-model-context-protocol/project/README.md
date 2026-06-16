# Core Engineering Project — Secure MCP Server + Agentic Loop

## Objective
Build a secure MCP server with sandboxed tools, connect it to Claude Desktop, and
implement a resilient tool-use agent.

## Deliverables
1. **MCP server** exposing: sandboxed code executor (Docker), read-only filesystem
   browser, and web search tool.
2. **stdio connection to Claude Desktop**; tool-use agent that reads code,
   refactors it, and runs unit tests.
3. **Credential isolation** (no keys in LLM context), output sanitisation, and
   prompt injection detection guard.
4. **Session-drop simulation**: verify agent resumes correctly with state restored
   from checkpoint.

## Acceptance Checklist
- [ ] All 3 tools functional and exposed via correct MCP primitives
- [ ] Code executor runs in Docker with restricted network + read-only mount
- [ ] Filesystem server enforces read-only access
- [ ] Agent successfully reads, refactors, and tests code end-to-end via stdio
- [ ] No API keys/credentials ever appear in LLM context (verified)
- [ ] Output sanitisation layer catches at least one simulated injection attempt
- [ ] Session drop mid-task is simulated and agent resumes from checkpoint correctly

## Results
Place final report, server logs, and demo transcripts in `results/`.
