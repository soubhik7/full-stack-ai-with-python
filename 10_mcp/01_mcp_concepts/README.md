# 01 — MCP Concepts: How the Protocol Works

> Understand the architecture before writing a single line of server code.

---

## What Is MCP?

**Model Context Protocol (MCP)** is an open standard (published by Anthropic, 2024) that defines how AI applications *connect to external tools and data*.

Before MCP:
```
App A: OpenAI function-calling JSON
App B: LangChain BaseTool + its own registry  
App C: Custom REST wrapper
```
After MCP:
```
App A, B, C  →  MCP  →  any server, any language, any tool
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      HOST PROCESS                          │
│                                                            │
│   ┌──────────────┐     JSON-RPC      ┌─────────────────┐   │
│   │  MCP Client  │ ◄──────────────► │   MCP Server     │   │
│   │  (your app)  │                   │  (your tools)   │   │
│   └──────────────┘                   └─────────────────┘   │
│                                                            │
│   The LLM sits inside the client side (or calls it)        │
└────────────────────────────────────────────────────────────┘
```

**MCP Client** = the LLM application (Claude Desktop, your Python script)  
**MCP Server** = the process that exposes tools/data  
**Transport** = how they talk (stdio or SSE)

---

## Transports

### 1. stdio (Standard I/O) — Most Common for Local Tools

```
Client ──stdin──► Server
Client ◄─stdout── Server
```

- Server runs as a subprocess of the client
- Client spawns the server, connects via stdin/stdout
- Best for local tools (file system, databases, calculators)
- No network port needed

```python
# Client launches server like this:
StdioServerParameters(command="python", args=["server.py"])

# Server just calls:
mcp.run()  # reads from stdin, writes to stdout
```

### 2. SSE (Server-Sent Events) — For Remote/Web Servers

```
Client ──HTTP POST──► Server  (sends requests)
Client ◄──SSE stream── Server  (receives responses)
```

- Server runs independently on a port
- Multiple clients can connect
- Best for cloud-deployed tools, shared infrastructure

```python
# Server
mcp.run(transport="sse", port=8000)

# Client
from mcp.client.sse import sse_client
async with sse_client("http://localhost:8000/sse") as (r, w):
    ...
```

---

## The Three Primitives

### 1. Tools — Callable Functions

The most common primitive. The LLM calls a tool to perform an action.

```json
{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "inputSchema": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "City name"}
    },
    "required": ["city"]
  }
}
```

**When is a tool called?**  
The LLM reads the tool's `description` and decides whether to invoke it based on user intent. Good descriptions are essential.

### 2. Resources — Read-Only Data

Resources expose data/content that the LLM can read but not modify.

```
URI examples:
  file:///home/user/notes.txt
  db://customers/table
  https://api.example.com/data
```

Resources have:
- A **URI** (unique identifier)
- A **MIME type** (text/plain, application/json, image/png…)
- **Contents** (text or binary blob)

### 3. Prompts — Reusable Templates

Prompt templates with parameters, stored server-side.

```json
{
  "name": "summarise_document",
  "description": "Summarise a document for a specific audience",
  "arguments": [
    {"name": "document", "description": "The document text"},
    {"name": "audience", "description": "Who the summary is for"}
  ]
}
```

---

## Message Lifecycle (How a Tool Call Works)

```
1. CLIENT sends:  initialize  (declare capabilities)
2. SERVER responds: initialized  (confirm capabilities)

3. CLIENT sends:  tools/list
4. SERVER responds: [{name, description, inputSchema}, ...]

5. LLM sees the tool list, user asks a question
6. LLM decides to call "get_weather" with {"city": "Delhi"}

7. CLIENT sends:  tools/call  {name: "get_weather", arguments: {city: "Delhi"}}
8. SERVER executes get_weather("Delhi")
9. SERVER responds: {content: [{type: "text", text: "20°C, Partly Cloudy"}]}

10. CLIENT gives result back to LLM
11. LLM incorporates result into its answer
```

All messages are **JSON-RPC 2.0** format:

```json
// Request (client → server)
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {"city": "Delhi"}
  }
}

// Response (server → client)
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [{"type": "text", "text": "20°C, Partly Cloudy"}],
    "isError": false
  }
}
```

---

## Capabilities Negotiation

During `initialize`, client and server declare what they support:

```json
// Client declares:
{
  "roots": {"listChanged": true},
  "sampling": {}
}

// Server declares:
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true},
  "prompts": {}
}
```

---

## Security Model

MCP servers should:
1. **Validate all inputs** — never execute raw strings as shell commands
2. **Sandbox file access** — only allow specific directories
3. **Require explicit permissions** — don't expose sensitive ops by default
4. **No credentials in tool responses** — return only what the LLM needs

---

## Common Mistakes

| Mistake | Why it's a problem |
|---------|-------------------|
| Vague tool descriptions | LLM won't know when to use the tool |
| Returning huge blobs | Wastes context window |
| No error handling in tools | Crashes cascade to the LLM |
| Forgetting `await session.initialize()` | Client/server never agree on capabilities |

---

## Ready to Build?

Move to **[02_mcp_server/](../02_mcp_server/README.md)** →
