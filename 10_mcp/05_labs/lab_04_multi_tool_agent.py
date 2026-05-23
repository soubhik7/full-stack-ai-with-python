"""
lab_04_multi_tool_agent.py — Agent Using All Three Lab Servers
==============================================================
The capstone lab: one GPT-4o agent that simultaneously connects to:
  1. lab_01_weather_mcp_server.py  → weather tools
  2. lab_02_sqlite_mcp_server.py   → database tools
  3. lab_03_filesystem_mcp_server.py → file tools

Example conversation:
  "What's the weather in Mumbai?"
  "Store it in a file called today_weather.txt"
  "Create a DB table called weather_log and record today's reading"
  "Show me everything in the weather_log table"

Run:
    python 10_mcp/05_labs/lab_04_multi_tool_agent.py

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

LABS_DIR = Path(__file__).parent

SERVERS = {
    "weather": StdioServerParameters(
        command=sys.executable,
        args=[str(LABS_DIR / "lab_01_weather_mcp_server.py")],
    ),
    "database": StdioServerParameters(
        command=sys.executable,
        args=[str(LABS_DIR / "lab_02_sqlite_mcp_server.py")],
    ),
    "filesystem": StdioServerParameters(
        command=sys.executable,
        args=[str(LABS_DIR / "lab_03_filesystem_mcp_server.py")],
    ),
}

openai_client = OpenAI()


SYSTEM_PROMPT = """You are an intelligent assistant with access to three powerful tool sets:

1. **Weather tools** — get current weather, forecasts, UV index for any city
2. **Database tools** — create tables, insert/update/delete rows, run SELECT queries
3. **Filesystem tools** — read/write files, list directories, search files

Use these tools proactively when they help answer the user's request.
For complex tasks, chain multiple tool calls together.
Be concise in your final answers.

Safety rules:
- Never run destructive SQL without confirming with the user
- Never delete files without explicit user instruction
"""


class CapstoneAgent:
    def __init__(self):
        self.sessions: dict[str, ClientSession] = {}
        self.tool_to_server: dict[str, str] = {}
        self.all_tools: list[dict] = []
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def connect_all(self, exit_stack: AsyncExitStack):
        print("🔌 Connecting to all lab servers...")
        for name, params in SERVERS.items():
            try:
                r, w = await exit_stack.enter_async_context(stdio_client(params))
                session = await exit_stack.enter_async_context(ClientSession(r, w))
                await session.initialize()
                self.sessions[name] = session

                tools_resp = await session.list_tools()
                for tool in tools_resp.tools:
                    self.tool_to_server[tool.name] = name
                    self.all_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": f"[{name}] {tool.description or ''}",
                            "parameters": tool.inputSchema,
                        },
                    })
                print(f"  ✅ {name}: {len(tools_resp.tools)} tools")
            except Exception as e:
                print(f"  ⚠️  {name}: Failed to connect — {e}")

        total = len(self.all_tools)
        print(f"\n🔧 {total} tools available across {len(self.sessions)} servers")

    async def _call_tool(self, name: str, args: dict) -> str:
        server = self.tool_to_server.get(name)
        if not server:
            return f"Unknown tool: '{name}'"
        try:
            result = await self.sessions[server].call_tool(name, args)
            if result.isError:
                return f"ERROR: {result.content[0].text}"
            return result.content[0].text
        except Exception as e:
            return f"ERROR: {e}"

    async def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for turn in range(15):
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

            if response.choices[0].finish_reason == "tool_calls" and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    server = self.tool_to_server.get(name, "unknown")
                    print(f"  🛠️  [{server}] {name}({args})")

                    result_text = await self._call_tool(name, args)
                    # Truncate very long results for display
                    display = result_text[:200] + "..." if len(result_text) > 200 else result_text
                    print(f"  📊  {display}")

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })

        return "Reached iteration limit."


DEMO_QUESTIONS = [
    "What's the current weather in London?",
    "Now get the weather in Mumbai and compare it with London.",
    "Write a summary of both cities' weather to a file called weather_summary.txt",
    "Create a database table called weather_log with columns: id INTEGER PRIMARY KEY, city TEXT, temp_c INTEGER, conditions TEXT",
    "Insert the London and Mumbai weather data into the weather_log table",
    "Show me all records in the weather_log table",
    "List all files in the filesystem sandbox",
]


async def main():
    agent = CapstoneAgent()

    async with AsyncExitStack() as stack:
        await agent.connect_all(stack)

        print("\n" + "=" * 65)
        print("  🤖 Multi-Tool Capstone Agent (Weather + DB + Filesystem)")
        print("  Type 'demo' to run the demo sequence, 'quit' to exit")
        print("=" * 65)

        print("\nExample queries:")
        for q in DEMO_QUESTIONS[:4]:
            print(f"  • {q}")

        while True:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ("quit", "exit", "q"):
                break

            if user_input.lower() == "demo":
                print("\n🎬 Running demo sequence...\n")
                for question in DEMO_QUESTIONS:
                    print(f"\n👤 Demo: {question}")
                    answer = await agent.chat(question)
                    print(f"🤖 Agent: {answer}")
                continue

            if not user_input:
                continue

            answer = await agent.chat(user_input)
            print(f"\n🤖 Agent: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
