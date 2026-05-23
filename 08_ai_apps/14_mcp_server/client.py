"""
App 14 — MCP Task Manager Client
==================================
Interactive client for the task manager MCP server.
Connects via stdio, lists all capabilities, then provides
a chat loop where GPT-4o manages tasks on your behalf.

Run: python 08_ai_apps/14_mcp_server/client.py
Requires: OPENAI_API_KEY in .env
"""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv()

SERVER_PATH = Path(__file__).parent / "server.py"
openai_client = OpenAI()


def mcp_to_openai_tools(mcp_tools) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]


async def task_agent(session: ClientSession):
    """Interactive task management agent."""
    tools_resp = await session.list_tools()
    openai_tools = mcp_to_openai_tools(tools_resp.tools)

    print(f"\n🔧 Available tools: {[t.name for t in tools_resp.tools]}")

    SYSTEM = (
        "You are a task management assistant. "
        "Use the available tools to help the user manage their tasks. "
        "Always call list_tasks to understand the current state before making changes. "
        "Be proactive about checking stats when the user asks for a summary."
    )

    messages = [{"role": "system", "content": SYSTEM}]

    print("\n" + "="*55)
    print("  📋 Task Manager (powered by GPT-4o + MCP)")
    print("  Type 'quit' to exit")
    print("="*55)
    print("\nExamples:")
    print("  'Show me all my pending tasks'")
    print("  'Add a high priority task: Deploy to production'")
    print("  'Complete task1 and show stats'")
    print("  'What should I work on today?'")

    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # Agentic loop
        for _ in range(8):
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )

            msg = response.choices[0].message
            messages.append(msg)

            if response.choices[0].finish_reason == "stop":
                print(f"\n🤖 Assistant: {msg.content}")
                break

            if response.choices[0].finish_reason == "tool_calls" and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    result = await session.call_tool(name, args)
                    tool_text = result.content[0].text if result.content else ""
                    print(f"  🛠️  {name}({args}) → {tool_text[:80]}...")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_text,
                    })


async def main():
    print(f"Connecting to task manager server: {SERVER_PATH.name}")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    async with stdio_client(server_params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            print("✅ Connected!")
            await task_agent(session)


if __name__ == "__main__":
    asyncio.run(main())
