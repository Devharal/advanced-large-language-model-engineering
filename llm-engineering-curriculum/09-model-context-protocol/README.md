# Module 9 — Model Context Protocol and Standardized Tool Integration

> Standardize connection architectures between frontier models and external
> environments using uniform real-time protocols.

## Status
- [ ] Readiness map reviewed (`notes/00-readiness-map.md`)
- [ ] Part 1 notes + implementations
- [ ] Part 2 notes + implementations
- [ ] Part 3 notes + implementations
- [ ] Core Engineering Project complete

## Folder Structure
```
09-model-context-protocol/
├── README.md
├── notes/
│   ├── 00-readiness-map.md
│   ├── 01-mcp-fundamentals.md       # topology, JSON-RPC, primitives, sampling
│   ├── 02-transport-session.md      # stdio/SSE, reconnect, backoff
│   ├── 03-security-sandboxing.md
│   └── 04-enterprise-nxm.md
├── src/
│   ├── mcp_server_minimal.py        # 3 tools over stdio
│   ├── mcp_client.py                # initialize handshake, capability discovery
│   ├── sampling_primitive.py
│   ├── stdio_sse_transports.py
│   ├── session_reconnect.py
│   ├── output_sanitizer.py
│   └── multi_server_orchestrator.py
├── servers/
│   ├── code_executor_server/        # Dockerized sandbox
│   ├── filesystem_server/           # read-only mount
│   └── web_search_server/
└── project/
    ├── README.md
    └── results/
```

## Topics & Resource Directory

### Part 1 — MCP Architecture

| Topic | Key Concepts | What to Implement |
|---|---|---|
| MCP Fundamentals | Host/Client/Server, JSON-RPC 2.0, Resources/Tools/Prompts, Sampling | Minimal MCP server (3 tools, stdio) → Claude desktop; client w/ initialize handshake; sampling primitive |
| Transport & Session Management | stdio vs SSE, session state, exponential backoff | Implement both transports + failover; simulate session drop + reconnect; backoff on >5s timeouts |

**Resources:**
- Spec: Anthropic MCP Specification
- Course: Anthropic *Building with MCP* docs
- Repo: `modelcontextprotocol/servers`
- Blog: Anthropic *Introducing MCP*

### Part 2 — Security & Sandboxing

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Security & Runtime Sandboxing | Credential containment, Docker/Wasm sandbox, access control, injection via results | Dockerized MCP server (restricted network, read-only FS); output sanitisation; injection detection guard |

**Resources:**
- Paper: *Prompt Injection Attacks via Tool Results* (Greshake et al. 2023)
- Course: OWASP LLM Top 10
- Repo: `docker/docker-ce`
- Blog: Simon Willison MCP security

### Part 3 — Enterprise Integration

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Enterprise Scale & N×M Problem | N×M → N+M reduction, tool discovery, multi-server orchestration | Agent discovering/selecting from 5 MCP servers; multi-server session manager w/ credential isolation; latency overhead |

**Resources:**
- Paper: *MCP: A New Open Standard for AI Tool Integration* (Anthropic 2024)
- Course: Cursor/Windsurf MCP tutorials
- Repo: `punkpeye/awesome-mcp-servers`
- Blog: Every.to MCP Explained

## Core Engineering Project
**Secure MCP Server + Agentic Loop** — see [`project/README.md`](project/README.md)
