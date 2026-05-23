"""
01_basic_client.py — Basic MCP Client
======================================
Demonstrates the core MCP client workflow:

  1. Launch the hello-server as a subprocess
  2. Open a stdio connection
  3. Initialize the session (capability handshake)
  4. Discover tools, resources, and prompts
  5. Call tools manually
  6. Read a resource

Run:
    python 10_mcp/03_mcp_client/01_basic_client.py

No LLM involved — this is pure protocol exploration.
"""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Path to the server we want to connect to
SERVER_PATH = Path(__file__).parent.parent / "02_mcp_server" / "01_hello_mcp.py"


async def main():
    print("=" * 60)
    print("  MCP Basic Client — Protocol Explorer")
    print("=" * 60)

    # ── 1. Define server startup parameters ──────────────────────────────────
    server_params = StdioServerParameters(
        command=sys.executable,        # use the same Python as this script
        args=[str(SERVER_PATH)],        # server script path
        env=None,                       # inherit current environment
    )

    print(f"\n📡 Connecting to: {SERVER_PATH.name}")

    # ── 2. Open stdio connection (spawns the server process) ─────────────────
    async with stdio_client(server_params) as (read_stream, write_stream):

        # ── 3. Create session ─────────────────────────────────────────────────
        async with ClientSession(read_stream, write_stream) as session:

            # ── 4. Initialize (REQUIRED — capability handshake) ───────────────
            init_result = await session.initialize()
            print(f"✅ Connected to server: {init_result.serverInfo.name}")
            print(f"   Server version: {init_result.serverInfo.version}")

            # ── 5. List tools ─────────────────────────────────────────────────
            print("\n📋 Available Tools:")
            tools_response = await session.list_tools()
            for tool in tools_response.tools:
                print(f"  🔧 {tool.name}")
                print(f"     {tool.description}")
                if tool.inputSchema.get("properties"):
                    params = list(tool.inputSchema["properties"].keys())
                    print(f"     Parameters: {params}")

            # ── 6. List resources ─────────────────────────────────────────────
            print("\n📦 Available Resources:")
            resources_response = await session.list_resources()
            if resources_response.resources:
                for resource in resources_response.resources:
                    print(f"  📄 {resource.uri} — {resource.name}")
            else:
                print("  (no resources)")

            # ── 7. List prompts ───────────────────────────────────────────────
            print("\n💬 Available Prompts:")
            prompts_response = await session.list_prompts()
            if prompts_response.prompts:
                for prompt in prompts_response.prompts:
                    print(f"  📝 {prompt.name} — {prompt.description}")
            else:
                print("  (no prompts)")

            # ── 8. Call tools ─────────────────────────────────────────────────
            print("\n🚀 Calling Tools:")

            # Call greet
            print("\n  → Calling greet(name='Alice')")
            result = await session.call_tool("greet", {"name": "Alice"})
            if result.isError:
                print(f"  ❌ Error: {result.content[0].text}")
            else:
                print(f"  ✅ {result.content[0].text}")

            # Call greet again
            print("\n  → Calling greet(name='Bob')")
            result = await session.call_tool("greet", {"name": "Bob"})
            print(f"  ✅ {result.content[0].text}")

            # Call ping
            print("\n  → Calling ping()")
            result = await session.call_tool("ping", {})
            print(f"  ✅ {result.content[0].text}")

            # ── 9. Demo: call with wrong args (error handling) ────────────────
            print("\n⚠️  Error handling demo:")
            print("  → Calling greet() without required arg 'name'")
            try:
                result = await session.call_tool("greet", {})
                if result.isError:
                    print(f"  ❌ Server error: {result.content[0].text}")
                else:
                    print(f"  ✅ {result.content[0].text}")
            except Exception as e:
                print(f"  ❌ Client-side exception: {e}")

    print("\n✅ Session closed. Server process terminated.")
    print("\nNext: Try 02_openai_mcp_client.py to add an LLM into the loop!")


if __name__ == "__main__":
    asyncio.run(main())
