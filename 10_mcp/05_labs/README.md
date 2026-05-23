# 05 — Labs: Real-World MCP Servers

> Build production-grade MCP servers for common real-world use cases.

---

## Labs

| Lab | What you build | Key concepts |
|-----|---------------|-------------|
| `lab_01_weather_mcp_server.py` | Weather data server | External API, async tools, structured responses |
| `lab_02_sqlite_mcp_server.py` | Full SQLite CRUD server | Database tools, parameterised queries, resources |
| `lab_03_filesystem_mcp_server.py` | Safe file system server | Security, allowlists, path validation |
| `lab_04_multi_tool_agent.py` | Agent using all three labs | Multi-server client, tool routing |

---

## Running Each Lab

```bash
# Terminal 1 — Start a lab server (it blocks waiting for client)
python 10_mcp/05_labs/lab_01_weather_mcp_server.py

# Terminal 2 — Run the multi-tool agent that connects to all labs
python 10_mcp/05_labs/lab_04_multi_tool_agent.py
```

Or use the MCP inspector to explore any server interactively:
```bash
mcp dev 10_mcp/05_labs/lab_01_weather_mcp_server.py
```

---

## Lab 01 — Weather MCP Server

**What you build:** An MCP server that fetches real weather data.

**Tools exposed:**
- `get_current_weather(city)` — current conditions + temperature
- `get_forecast(city, days)` — N-day forecast
- `get_uv_index(city)` — UV index and recommendation

**What you learn:**
- Calling external REST APIs from a tool
- Returning structured JSON data
- Handling API errors gracefully

---

## Lab 02 — SQLite MCP Server

**What you build:** A full-featured database interface via MCP.

**Tools exposed:**
- `create_table(name, columns)` — create a table
- `insert_row(table, data)` — insert a record
- `query(sql)` — run a SELECT query
- `list_tables()` — list all tables
- `describe_table(name)` — show schema

**Resources exposed:**
- `db://schema` — full database schema
- `db://tables/{name}` — all rows from a table

**What you learn:**
- Database operations through MCP tools
- Exposing data as resources
- SQL injection prevention

---

## Lab 03 — Filesystem MCP Server

**What you build:** A safe file system interface that restricts access to a sandbox directory.

**Tools exposed:**
- `read_file(path)` — read a file's contents
- `write_file(path, content)` — write/create a file
- `list_directory(path)` — list files and folders
- `create_directory(path)` — create a directory
- `delete_file(path)` — delete a file (with confirmation)
- `file_info(path)` — size, modified date, permissions

**Security features:**
- All paths are validated against an `ALLOWED_ROOT`
- Path traversal attacks (`../../../etc/passwd`) are blocked
- Files > 10 MB are rejected

---

## Lab 04 — Multi-Tool Agent

**What you build:** A single agent that connects to all three lab servers simultaneously.

```
User: "What's the weather in Mumbai? Store it in a file called weather.txt, 
       then create a DB table called weather_log and insert today's reading."

Agent: [calls weather server] → 25°C, Partly Cloudy
       [calls filesystem server] → writes weather.txt
       [calls sqlite server] → creates table, inserts row
       [answers] "Done! Weather stored in file and database."
```

---

## Challenge: Build Your Own Server

After completing the labs, try building one of these:

1. **GitHub MCP Server** — tools for listing repos, issues, PRs (use `PyGithub`)
2. **Email MCP Server** — send/read emails via SMTP/IMAP
3. **PDF MCP Server** — read, search, and extract from PDFs (use `pypdf`)
4. **Notion MCP Server** — read and write to Notion pages (use Notion API)
5. **Slack MCP Server** — send messages, list channels, search messages

---

## Key Patterns Practiced

```python
# Pattern 1: Async external API call
@mcp.tool()
async def get_weather(city: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.weather.com/{city}")
        return resp.json()["temperature"]

# Pattern 2: Database query
@mcp.tool()
def query(sql: str) -> str:
    conn = sqlite3.connect("data.db")
    rows = conn.execute(sql).fetchall()
    return json.dumps(rows)

# Pattern 3: Secured file access
@mcp.tool()
def read_file(path: str) -> str:
    full = (ALLOWED_ROOT / path).resolve()
    if not str(full).startswith(str(ALLOWED_ROOT)):
        raise ValueError("Access denied: path outside allowed directory")
    return full.read_text()
```
