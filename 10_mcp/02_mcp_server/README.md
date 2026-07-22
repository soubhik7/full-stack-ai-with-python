# 02 — Building MCP Servers

> Build MCP servers using `FastMCP` — from a one-tool hello world to a multi-primitive production server.

---

## Setup

```bash
source venv/bin/activate
pip3 install mcp
```

---

## Files

| File | Concepts |
|------|---------|
| `01_hello_mcp.py` | Minimal server, one tool, stdio transport |
| `02_calculator_server.py` | Multiple tools, typed parameters, error handling |
| `03_resources_server.py` | Tools + Resources + Prompts together |

---

## Running a Server Directly

Servers communicate over stdio — you don't usually run them directly (the client spawns them).  
But you can test with the MCP CLI dev tool:

```bash
# Install the MCP CLI
pip3 install "mcp[cli]"

# Inspect your server (opens an interactive inspector)
mcp dev 02_mcp_server/01_hello_mcp.py

# Or run it raw (it blocks waiting for JSON-RPC on stdin)
python 02_mcp_server/01_hello_mcp.py
```

---

## FastMCP API Reference

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("server-name")

# ── Tools ──────────────────────────────────────────────────
@mcp.tool()
def my_tool(param: str) -> str:
    """Tool description — the LLM reads this to decide when to call."""
    return f"Result: {param}"

# ── Resources ──────────────────────────────────────────────
@mcp.resource("resource://my-data")
def my_resource() -> str:
    """Expose read-only data."""
    return "This is the data content"

# ── Prompts ────────────────────────────────────────────────
@mcp.prompt()
def my_prompt(topic: str) -> str:
    """A reusable prompt template."""
    return f"Explain {topic} in simple terms."

# ── Start ──────────────────────────────────────────────────
mcp.run()  # stdio by default
```

---

## Key Points

1. **Descriptions matter** — the LLM only sees the docstring to decide whether to call your tool.
2. **Type hints are required** — FastMCP uses them to build the JSON Schema for the LLM.
3. **Return strings** — FastMCP converts them to `TextContent` automatically.
4. **Raise exceptions normally** — MCP wraps them as `isError: true` responses.

---

## Next Step

Build a client to call these servers → **[03_mcp_client/](../03_mcp_client/README.md)**
