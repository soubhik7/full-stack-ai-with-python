"""
01_hello_mcp.py — Your First MCP Server
========================================
The simplest possible MCP server: one tool that echoes a greeting.

Run with:
    mcp dev 10_mcp/02_mcp_server/01_hello_mcp.py   # interactive inspector
    python 10_mcp/02_mcp_server/01_hello_mcp.py     # raw stdio (blocks)

The server exposes ONE tool:
    greet(name: str) -> str
"""

from mcp.server.fastmcp import FastMCP

# ── 1. Create the server ──────────────────────────────────────────────────────
#  The name appears in Claude Desktop, MCP Inspector, and log output.
mcp = FastMCP("hello-server")


# ── 2. Register a tool ───────────────────────────────────────────────────────
#  @mcp.tool() turns a plain Python function into an MCP tool.
#  The docstring becomes the tool's description (the LLM reads this).
#  Type hints become the JSON Schema that validates inputs.

@mcp.tool()
def greet(name: str) -> str:
    """
    Greet a person by name.

    Args:
        name: The person's name to greet.

    Returns:
        A friendly greeting string.
    """
    return f"Hello, {name}! Welcome to MCP 👋"


@mcp.tool()
def ping() -> str:
    """
    Check if the server is alive.

    Returns:
        A pong message confirming the server is running.
    """
    return "pong 🏓"


# ── 3. Start the server ───────────────────────────────────────────────────────
#  mcp.run() defaults to stdio transport — it reads JSON-RPC from stdin
#  and writes responses to stdout.
#  The client (Claude Desktop, your Python client) spawns this process.

if __name__ == "__main__":
    print("Starting hello-server (stdio transport)...")
    print("This server exposes tools: greet, ping")
    print("Connect a client to interact.")
    mcp.run()
