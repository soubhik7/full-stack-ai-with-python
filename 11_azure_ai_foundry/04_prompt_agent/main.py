####################################################################################################
# LAB 4 — PROMPT AGENT (lightweight, stateless — no Azure-hosted Agent resource at all)
#
# WHY THIS MATTERS
#   Not every use case needs a persistent, server-managed agent. A "prompt agent" here means:
#   no create_agent() call, no agent_id, no thread_id — just a system prompt + a chat completion,
#   the same shape as a one-off API call. Cheaper and simpler when you don't need memory.
#
# HOW IT LINKS TO THE REST OF THE CHAPTER
#   Directly contrasts with 05_hosted_agent/main.py, which answers the exact same kind of
#   question but DOES remember prior turns because the thread lives on Azure. Run both and
#   compare the second question in each: here it has no memory, there it does.
#
# HOW IT WORKS
#   get_openai_client() (from AIProjectClient) returns a plain OpenAI-SDK client pointed at this
#   Foundry project — there is no Agents/Threads/Runs API involved anywhere in this file.
#   Function calling still works (FunctionTool generates the schema, same as 02_tools/), but
#   nothing executes it automatically: this file manually checks message.tool_calls, runs the
#   Python function itself, and appends the result as a role="tool" message before asking the
#   model to finish its answer — the loop the Agent Service automates away in later labs.
####################################################################################################
import json
import os

from azure.ai.agents.models import FunctionTool
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
DEPLOYMENT = os.environ["AZURE_AI_MODEL_DEPLOYMENT"]
SYSTEM_PROMPT = "You are a concise Crystal Hotels assistant."


def check_room_availability(room_type: str, check_in_date: str, check_out_date: str) -> str:
    """Check whether a Crystal Hotels room type is available for a date range.

    :param room_type: The room type, e.g. "Deluxe" or "Standard".
    :param check_in_date: Check-in date, e.g. "2026-06-20".
    :param check_out_date: Check-out date, e.g. "2026-06-23".
    """
    return f"{room_type} rooms are available from {check_in_date} to {check_out_date}."


TOOL_DEFINITIONS = FunctionTool(functions={check_room_availability}).definitions
LOCAL_FUNCTIONS = {"check_room_availability": check_room_availability}


def ask(openai_client, user_message: str) -> str:
    """A single, stateless prompt-agent turn: no agent_id, no thread_id, no server-side memory.
    Every call starts from this same system prompt — the caller owns the conversation history."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    response = openai_client.chat.completions.create(
        model=DEPLOYMENT, messages=messages, tools=TOOL_DEFINITIONS
    )
    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)
        for tool_call in message.tool_calls:
            fn = LOCAL_FUNCTIONS[tool_call.function.name]
            args = json.loads(tool_call.function.arguments)
            result = fn(**args)
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )
        response = openai_client.chat.completions.create(model=DEPLOYMENT, messages=messages)
        message = response.choices[0].message

    return message.content


def main():
    project_client = AIProjectClient(endpoint=ENDPOINT, credential=DefaultAzureCredential())
    openai_client = project_client.get_openai_client()

    print(ask(openai_client, "Is a Deluxe room available from 2026-06-20 to 2026-06-23?"))
    print(ask(openai_client, "What did I just ask about?"))  # no memory: each call is independent


if __name__ == "__main__":
    main()
