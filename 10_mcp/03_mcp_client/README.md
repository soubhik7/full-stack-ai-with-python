# 03 — Building MCP Clients

> Connect to MCP servers, discover tools, and call them — with and without an LLM in the loop.

---

## Files

| File | What it shows |
|------|--------------|
| `01_basic_client.py` | Connect to a server, list tools/resources/prompts, call a tool manually |
| `02_openai_mcp_client.py` | Full agentic loop: GPT-4 reads tools, picks one, calls it, returns answer |

---

## How the Client Works

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # 1. Define how to start the server
    server = StdioServerParameters(
        command="python",
        args=["path/to/server.py"],
    )

    # 2. Start the server process and open a connection
    async with stdio_client(server) as (read_stream, write_stream):

        # 3. Create a session
        async with ClientSession(read_stream, write_stream) as session:

            # 4. Handshake — REQUIRED before anything else
            await session.initialize()

            # 5. Discover capabilities
            tools = await session.list_tools()
            resources = await session.list_resources()
            prompts = await session.list_prompts()

            # 6. Call a tool
            result = await session.call_tool("greet", {"name": "Alice"})
            print(result.content[0].text)

asyncio.run(main())
```

---

## Running the Clients

Both clients start the server automatically as a subprocess.

```bash
# Basic client — connects to hello-server, lists everything, calls greet
python 10_mcp/03_mcp_client/01_basic_client.py

# OpenAI client — interactive chat that can call calculator tools
python 10_mcp/03_mcp_client/02_openai_mcp_client.py
```

---

## The Agentic Loop Pattern

```
User asks a question
        ↓
LLM receives question + list of MCP tools
        ↓
LLM decides: "I need to call add(3, 4)"
        ↓
Client calls session.call_tool("add", {"a": 3, "b": 4})
        ↓
Server executes, returns result
        ↓
Client sends result back to LLM
        ↓
LLM formulates final answer
```

This loop runs until the LLM stops requesting tool calls.

---

## Converting MCP Tools → LLM Format

Each LLM provider expects tools in a slightly different format.  
`02_openai_mcp_client.py` shows how to convert MCP tool schemas to OpenAI format:

```python
# MCP tool format
{"name": "add", "description": "...", "inputSchema": {"type": "object", ...}}

# OpenAI expects
{"type": "function", "function": {"name": "add", "description": "...", "parameters": {...}}}

# Conversion
openai_tools = [
    {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        },
    }
    for tool in mcp_tools
]
```

---

## Next Step

See MCP + Claude end-to-end → **[04_mcp_with_claude/](../04_mcp_with_claude/README.md)**
