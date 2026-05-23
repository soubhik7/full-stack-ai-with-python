"""
02_openai_mcp_client.py — Agentic Loop: GPT-4 + MCP Tools
==========================================================
Full agentic pattern:

  User asks question
       ↓
  GPT-4 receives question + MCP tool schemas
       ↓
  GPT-4 calls a tool via tool_calls
       ↓
  Client calls session.call_tool()
       ↓
  Server executes, returns result
       ↓
  Result injected back into conversation
       ↓
  GPT-4 gives final answer

This client connects to the CALCULATOR server.

Run:
    python 10_mcp/03_mcp_client/02_openai_mcp_client.py

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

# Calculator server path
SERVER_PATH = Path(__file__).parent.parent / "02_mcp_server" / "02_calculator_server.py"

openai_client = OpenAI()


def mcp_tools_to_openai_format(mcp_tools) -> list[dict]:
    """
    Convert MCP tool definitions to OpenAI function-calling format.

    MCP format:
      Tool(name="add", description="...", inputSchema={...})

    OpenAI format:
      {"type": "function", "function": {"name": "add", "description": "...", "parameters": {...}}}
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in mcp_tools
    ]


async def run_agent(session: ClientSession, user_question: str) -> str:
    """
    Run one complete agentic turn: user question → tool calls → final answer.

    Args:
        session: Active MCP session with an initialized server.
        user_question: The user's natural language question.

    Returns:
        The LLM's final answer string.
    """
    # Get current tool list from server
    tools_response = await session.list_tools()
    openai_tools = mcp_tools_to_openai_format(tools_response.tools)

    print(f"\n🔍 Available tools: {[t.name for t in tools_response.tools]}")

    # Build conversation
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful calculator assistant. "
                "Use the provided tools to answer mathematical questions. "
                "Always show your work by calling the appropriate tool."
            ),
        },
        {"role": "user", "content": user_question},
    ]

    # ── Agentic loop ─────────────────────────────────────────────────────────
    iteration = 0
    max_iterations = 10  # safety limit

    while iteration < max_iterations:
        iteration += 1
        print(f"\n⚙️  LLM iteration {iteration}...")

        # Call GPT-4 with tools
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )

        message = response.choices[0].message
        messages.append(message)  # add assistant response to history

        # Check finish reason
        finish_reason = response.choices[0].finish_reason
        print(f"   Finish reason: {finish_reason}")

        if finish_reason == "stop":
            # LLM is done — return the final text answer
            return message.content or ""

        if finish_reason == "tool_calls":
            # Process each tool call the LLM requested
            if not message.tool_calls:
                break

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"   🛠️  Calling: {tool_name}({tool_args})")

                # Call the tool via MCP
                try:
                    mcp_result = await session.call_tool(tool_name, tool_args)
                    if mcp_result.isError:
                        tool_result_text = f"ERROR: {mcp_result.content[0].text}"
                    else:
                        tool_result_text = mcp_result.content[0].text
                except Exception as e:
                    tool_result_text = f"ERROR calling {tool_name}: {str(e)}"

                print(f"   📊 Result: {tool_result_text}")

                # Add tool result to conversation
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_text,
                    }
                )

    return "Max iterations reached without a final answer."


async def interactive_chat(session: ClientSession):
    """Run an interactive chat loop with the calculator agent."""
    print("\n" + "=" * 60)
    print("  🧮 MCP Calculator Agent (powered by GPT-4)")
    print("  Ask any math question — type 'quit' to exit")
    print("=" * 60)
    print("\nExample questions:")
    print("  'What is 15% of 240?'")
    print("  'What is the square root of 144?'")
    print("  'Convert 100 km to miles'")
    print("  'Add 3.14 and 2.86, then multiply by 10'")
    print("  'Compute stats for [12, 45, 7, 89, 34, 56]'")

    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        answer = await run_agent(session, user_input)
        print(f"\n🤖 Agent: {answer}")


async def main():
    print("Starting MCP + OpenAI Calculator Agent...")
    print(f"Server: {SERVER_PATH.name}")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("✅ Connected to calculator server")
            await interactive_chat(session)


if __name__ == "__main__":
    asyncio.run(main())
