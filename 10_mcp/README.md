# Chapter 10 — Model Context Protocol (MCP)

> **You will master MCP — Anthropic's open protocol that lets LLMs talk to tools, databases, APIs, and file systems in a standardised, composable way.**

---

## Prerequisites

- [Chapter 08 — AI Applications](../08_ai_apps/README.md) *(agents, tool use)*
- [Chapter 06 — LLMs](../06_large_language_models/README.md) *(how LLMs work)*
- Packages: `mcp`, `anthropic`, `openai`, `httpx`, `pydantic`
- Install: `pip3 install mcp anthropic`

---

## Why MCP?

Before MCP every agent framework invented its own tool format:
- OpenAI function calling → JSON schema inside `tools=[]`
- LangChain → `BaseTool` class + its own registry
- LlamaIndex → different again

**MCP standardises this.** One protocol, any LLM, any tool.

```
Claude / GPT / Gemini
        ↓  (MCP client)
   MCP Protocol
        ↓  (MCP server)
  Your Tools / DBs / APIs
```

MCP servers expose three primitives:
| Primitive | What it is | Example |
|-----------|-----------|---------|
| **Tool** | A callable function | `get_weather(city)` |
| **Resource** | Read-only data/content | `file://notes.txt` |
| **Prompt** | Reusable prompt template | `summarise_email` |

---

## Learning Path

```
01_mcp_concepts/      ← Protocol architecture, transports, primitives
02_mcp_server/        ← Build MCP servers (tools, resources, prompts)
03_mcp_client/        ← Build MCP clients (connect + call tools)
04_mcp_with_claude/   ← End-to-end: Claude + your MCP server
05_labs/              ← Hands-on labs: weather, SQLite, filesystem
```

---

## Sub-chapter Breakdown

### `01_mcp_concepts/` — How MCP Works
Read this first — understand the protocol before writing code.

Topics: JSON-RPC messages, lifecycle, transports (stdio vs SSE), capabilities.

---

### `02_mcp_server/` — Build MCP Servers

| File | What it teaches |
|------|----------------|
| `01_hello_mcp.py` | Minimal server with one tool |
| `02_calculator_server.py` | Multiple tools with typed params |
| `03_resources_server.py` | Resources, prompts + tools together |

Run any server:
```bash
cd 10_mcp/02_mcp_server
python 01_hello_mcp.py
```

---

### `03_mcp_client/` — Build MCP Clients

| File | What it teaches |
|------|----------------|
| `01_basic_client.py` | Connect to server, list tools, call a tool |
| `02_openai_mcp_client.py` | Full loop: GPT-4 decides which tool to call |

Run a client (starts the server automatically):
```bash
cd 10_mcp/03_mcp_client
python 01_basic_client.py
```

---

### `04_mcp_with_claude/` — End-to-End with Claude

| File | What it teaches |
|------|----------------|
| `claude_mcp_app.py` | Anthropic SDK + MCP server in one script |
| `claude_desktop_config.json` | Plug your server into Claude Desktop |
| `openai_mcp_app.py` | Same pattern using OpenAI |

---

### `05_labs/` — Hands-On Labs

| Lab | What you build |
|-----|---------------|
| `lab_01_weather_mcp_server.py` | Weather tool + UV index + forecast |
| `lab_02_sqlite_mcp_server.py` | Full CRUD over a SQLite DB via MCP |
| `lab_03_filesystem_mcp_server.py` | Read/write/list files safely |
| `lab_04_multi_tool_agent.py` | Agent that uses all three servers |

---

## Key Concepts at a Glance

```python
# Server side — FastMCP makes it easy
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

mcp.run()  # starts stdio transport by default
```

```python
# Client side — connect and call
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

async with stdio_client(StdioServerParameters(command="python", args=["server.py"])) as (r, w):
    async with ClientSession(r, w) as session:
        await session.initialize()
        result = await session.call_tool("add", {"a": 3, "b": 4})
        print(result)  # 7
```

---

## Quick Install

```bash
# activate your venv first
source venv/bin/activate

pip3 install mcp anthropic
```

---

## MCP Ecosystem (Ready-Made Servers)

You don't always need to build a server — the ecosystem already has:

| Server | What it exposes |
|--------|----------------|
| `@modelcontextprotocol/server-filesystem` | Safe file operations |
| `@modelcontextprotocol/server-github` | GitHub repos, issues, PRs |
| `@modelcontextprotocol/server-postgres` | Query a Postgres DB |
| `@modelcontextprotocol/server-brave-search` | Web search |
| `mcp-server-sqlite` | SQLite queries |

These run as Node.js processes; your Python client talks to them via stdio.

---

## Next Step

After this chapter you have the full stack:

```
Python fundamentals (01) → Math (02) → ML (03) → Deep Learning (04)
→ NLP (05) → LLMs (06) → RAG (07) → AI Apps (08) → Projects (09)
→ MCP (10)  ← You are here
```

Head to **[Chapter 09 — Projects](../09_projects/README.md)** to build end-to-end systems using MCP.
