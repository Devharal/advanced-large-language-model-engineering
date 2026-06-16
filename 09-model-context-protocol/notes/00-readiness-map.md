# Section A — Module Readiness Map

## Part A1 — Prerequisite Dependency Tree

- **FROM MODULE 5 (required)**
  - Tool/function calling JSON schema request-response cycle
  - Agent loops: how an LLM decides to call a tool and process results
- **NETWORK PROTOCOLS**
  - HTTP/REST: request-response, status codes, headers
  - WebSockets vs SSE
  - JSON-RPC 2.0
- **PROCESS & SUBPROCESS MANAGEMENT**
  - stdin/stdout/stderr (stdio transport)
  - Process spawning
- **SECURITY BASICS**
  - API key management (env vars, never hardcoded)
  - Docker containers: filesystem/network isolation

## YOU ARE NOW READY FOR
MCP Topology → Primitives → Transports (stdio/SSE) → Security → Enterprise N×M

## Part A2 — Prerequisite Detail Table

| Concept | Why You Need It | Where to Learn It |
|---|---|---|
| JSON-RPC 2.0 spec | Every MCP message is JSON-RPC | jsonrpc.org/specification |
| SSE | Remote MCP servers use SSE transport | MDN SSE docs; FastAPI SSE tutorial |
| Docker networking/volumes | Sandboxed MCP servers run in Docker | Docker Get Started; Play with Docker |
| Python asyncio | Session management, timeouts, reconnect | Real Python asyncio; asyncio.gather docs |
| Prior tool-use agent experience | MCP standardises tool-calling | Complete Module 5 project first |

## Checklist
- [ ] I've read the JSON-RPC 2.0 spec (10 pages)
- [ ] I understand SSE vs WebSockets
- [ ] I can run a Dockerized process with restricted network/filesystem
- [ ] I'm comfortable with asyncio for session/timeout handling
- [ ] I have completed the Module 5 cyclic agent project
