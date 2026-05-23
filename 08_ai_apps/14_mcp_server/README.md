# App 14 — MCP Server Lab

> **Pattern:** Build and expose a custom MCP server — connect it to Claude Desktop or your own client.

---

## What You Build

A production-ready MCP server that exposes tools, resources, and prompts
for a **task management system**. Any MCP-compatible client (Claude Desktop,
your Python scripts, Claude Code) can connect to it.

```
┌─────────────────────┐   MCP (stdio)   ┌──────────────────┐
│  Claude Desktop /   │ ◄────────────── │   server.py      │
│  Your client.py     │                 │  (this app)      │
└─────────────────────┘                 └──────────────────┘
```

---

## Files

| File | What it does |
|------|-------------|
| `server.py` | Task manager MCP server (tools + resources + prompts) |
| `client.py` | Interactive client that connects to the server |

---

## Tools Exposed

| Tool | Description |
|------|-------------|
| `add_task(title, priority)` | Create a new task |
| `complete_task(task_id)` | Mark a task as done |
| `list_tasks(status)` | List pending/done/all tasks |
| `delete_task(task_id)` | Remove a task |
| `get_stats()` | Summary statistics |

## Resources

- `tasks://all` — All tasks as JSON
- `tasks://pending` — Pending tasks only
- `tasks://stats` — Completion statistics

## Prompts

- `daily_standup()` — Generate a standup update from pending tasks
- `prioritise_tasks()` — Ask Claude to help prioritise

---

## Run It

```bash
# Terminal 1 — Start the server
cd 08_ai_apps/14_mcp_server
python server.py

# Terminal 2 — Connect with the client
cd 08_ai_apps/14_mcp_server
python client.py
```

Or inspect with the MCP CLI:
```bash
mcp dev 08_ai_apps/14_mcp_server/server.py
```

Or add to Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "task-manager": {
      "command": "/path/to/venv/bin/python",
      "args": ["/absolute/path/to/08_ai_apps/14_mcp_server/server.py"]
    }
  }
}
```

---

## Concepts Demonstrated

- `FastMCP` server with all three primitives
- Persistent state in-memory (easily swappable to SQLite)
- Type-safe tool parameters
- Error handling with descriptive messages
- Resource URIs with path parameters

---

## Previous App

← [13 — Todo App](../13_todo/)

## Next App

→ [15 — Streaming](../15_streaming/)
