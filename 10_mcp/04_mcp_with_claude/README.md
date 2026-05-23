# 04 — MCP with Claude

> Connect your MCP servers to Claude Desktop, Claude Code, and the Anthropic SDK.

---

## Files

| File | Purpose |
|------|---------|
| `claude_desktop_config.json` | Plug your servers into Claude Desktop |
| `openai_mcp_app.py` | Full app: GPT-4o + custom MCP server (no extra deps) |
| `anthropic_mcp_app.py` | Full app: Claude Sonnet + MCP server (requires `anthropic`) |

---

## Option A — Claude Desktop (No Code Required)

Claude Desktop has built-in MCP support. Add your server to its config file:

**macOS config path:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows config path:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

Copy the contents of `claude_desktop_config.json` from this folder, adjusting the Python path and script paths to match your machine.

```json
{
  "mcpServers": {
    "calculator": {
      "command": "/path/to/venv/bin/python",
      "args": ["/absolute/path/to/10_mcp/02_mcp_server/02_calculator_server.py"]
    },
    "notes": {
      "command": "/path/to/venv/bin/python",
      "args": ["/absolute/path/to/10_mcp/02_mcp_server/03_resources_server.py"]
    }
  }
}
```

Then restart Claude Desktop → your tools appear automatically.

---

## Option B — Claude Code (CLI)

Add MCP servers to Claude Code via:

```bash
# Add a server
claude mcp add calculator -- python /absolute/path/to/02_calculator_server.py

# List registered servers
claude mcp list

# Remove a server
claude mcp remove calculator
```

Or add to `.claude/settings.json` in your project:
```json
{
  "mcpServers": {
    "calculator": {
      "command": "python",
      "args": ["10_mcp/02_mcp_server/02_calculator_server.py"]
    }
  }
}
```

---

## Option C — Anthropic SDK (Python Code)

```bash
pip install anthropic mcp
```

The pattern: list MCP tools → pass to Claude → loop on `tool_use` blocks.

See `anthropic_mcp_app.py` for the full implementation.

```python
import anthropic
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

# Connect to MCP server
async with stdio_client(server_params) as (r, w):
    async with ClientSession(r, w) as session:
        await session.initialize()
        
        # Get tools and convert to Anthropic format
        mcp_tools = (await session.list_tools()).tools
        anthropic_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            }
            for t in mcp_tools
        ]
        
        # Pass to Claude
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=anthropic_tools,
            messages=[{"role": "user", "content": "What is 15% of 240?"}]
        )
        
        # Handle tool_use responses
        for block in response.content:
            if block.type == "tool_use":
                result = await session.call_tool(block.name, block.input)
                # ... send result back to Claude
```

---

## How Claude Desktop Discovers Tools

```
1. Claude Desktop reads claude_desktop_config.json on startup
2. For each server in "mcpServers": spawns the process
3. Runs initialize handshake with each server
4. Fetches tools/resources/prompts
5. Tools appear in Claude's UI as: 🔧 calculator, 🔧 notes, etc.
6. When you ask Claude "what is 15% of 240?" it automatically calls the MCP tool
```

---

## Security Considerations

- Never put your API keys inside an MCP server script
- Restrict file-system tools to specific directories
- Use `ALLOWED_DIRS` allowlists for filesystem tools
- Log all tool calls for auditing

---

## Next Step

→ **[05_labs/](../05_labs/README.md)** — Build real-world MCP servers for weather, databases, and file systems.
