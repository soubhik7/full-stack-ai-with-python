"""
openai_mcp_app.py — Production-Style MCP App with OpenAI
=========================================================
A complete, production-style application demonstrating:
  - Connecting to MULTIPLE MCP servers simultaneously
  - Merging tool lists from multiple servers
  - Full agentic loop with GPT-4o
  - Graceful error handling and retry logic
  - Conversation history management

This app connects to both the calculator and notes servers.

Run:
    python 10_mcp/04_mcp_with_claude/openai_mcp_app.py

Requires: OPENAI_API_KEY in .env
"""

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv()

# ── Server paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent / "02_mcp_server"

SERVERS = {
    "calculator": StdioServerParameters(
        command=sys.executable,
        args=[str(BASE / "02_calculator_server.py")],
    ),
    "notes": StdioServerParameters(
        command=sys.executable,
        args=[str(BASE / "03_resources_server.py")],
    ),
}

openai_client = OpenAI()


class MultiServerMCPAgent:
    """
    An agent that connects to multiple MCP servers and uses GPT-4o
    to answer questions using any available tool from any server.
    """

    def __init__(self):
        self.sessions: dict[str, ClientSession] = {}
        self.tool_to_server: dict[str, str] = {}   # tool_name → server_name
        self.all_tools: list[dict] = []              # OpenAI-format tools
        self.messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to calculator and notes tools. "
                    "Use tools when needed. Be concise in your final answers."
                ),
            }
        ]

    async def connect(self, exit_stack: AsyncExitStack):
        """Connect to all servers and collect their tools."""
        print("🔌 Connecting to MCP servers...")
        for server_name, server_params in SERVERS.items():
            read, write = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            session = await exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions[server_name] = session
            print(f"  ✅ Connected: {server_name}")

            # Collect tools from this server
            tools_resp = await session.list_tools()
            for tool in tools_resp.tools:
                self.tool_to_server[tool.name] = server_name
                self.all_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": f"[{server_name}] {tool.description or ''}",
                        "parameters": tool.inputSchema,
                    },
                })

        print(f"\n🔧 Total tools available: {len(self.all_tools)}")
        for tool in self.all_tools:
            print(f"  • {tool['function']['name']}")

    async def _call_tool(self, name: str, args: dict) -> str:
        """Route a tool call to the correct server."""
        server_name = self.tool_to_server.get(name)
        if not server_name:
            return f"ERROR: Unknown tool '{name}'"

        session = self.sessions[server_name]
        try:
            result = await session.call_tool(name, args)
            if result.isError:
                return f"ERROR: {result.content[0].text}"
            return result.content[0].text
        except Exception as e:
            return f"ERROR calling {name}: {e}"

    async def chat(self, user_input: str) -> str:
        """
        Process one user message through the agentic loop.
        Returns the final answer string.
        """
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(10):  # max 10 iterations
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=self.messages,
                tools=self.all_tools,
                tool_choice="auto",
            )

            msg = response.choices[0].message
            self.messages.append(msg)

            if response.choices[0].finish_reason == "stop":
                return msg.content or ""

            if response.choices[0].finish_reason == "tool_calls":
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)

                    print(f"  🛠️  {name}({args})")
                    result = await self._call_tool(name, args)
                    print(f"  📊  {result}")

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

        return "Could not complete the request."


async def main():
    agent = MultiServerMCPAgent()

    async with AsyncExitStack() as stack:
        await agent.connect(stack)

        print("\n" + "=" * 60)
        print("  🤖 Multi-Server MCP Agent (GPT-4o)")
        print("  Tools: calculator + notes")
        print("  Type 'quit' to exit")
        print("=" * 60)
        print("\nTry these:")
        print("  'What is 17.5% of 3500?'")
        print("  'Create a note called \"MCP Tips\" about the stdio transport'")
        print("  'List my notes, then compute the square root of 256'")

        while True:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            answer = await agent.chat(user_input)
            print(f"\n🤖 Agent: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
