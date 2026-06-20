####################################################################################################
# LAB 2a — FUNCTION TOOL (custom Python function as an agent tool)
#
# WHY THIS MATTERS
#   An agent's instructions alone can't look up live facts (room availability, prices) — it can
#   only generate plausible-sounding text. A function tool is how you give it a real, callable
#   capability that you write and control, instead of trusting the model to "know" the answer.
#
# HOW IT LINKS TO THE REST OF THE CHAPTER
#   This is one tool, demonstrated in isolation. 03_toolbox/ bundles this same pattern (multiple
#   function tools + a built-in tool) into one reusable ToolSet. 04_prompt_agent/ shows the same
#   function-calling idea without the Agent Service at all (manual tool-call loop).
#
# HOW IT WORKS
#   FunctionTool({check_room_availability}) reads the function's type hints + docstring and
#   auto-generates the JSON schema the model needs to call it correctly — you never hand-write
#   that schema. enable_auto_function_calls({check_room_availability}) tells the Agent Service to
#   execute the Python function itself (locally, in this process) whenever the model requests it
#   during runs.create_and_process — no manual "catch the tool call, run it, send back the result"
#   loop required, unlike the raw OpenAI-style pattern in 04_prompt_agent/.
####################################################################################################
import os

from azure.ai.agents import AgentsClient
from azure.ai.agents.models import FunctionTool
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
DEPLOYMENT = os.environ["AZURE_AI_MODEL_DEPLOYMENT"]


def check_room_availability(room_type: str, check_in_date: str, check_out_date: str) -> str:
    """Check whether a Crystal Hotels room type is available for a date range.

    :param room_type: The room type, e.g. "Deluxe" or "Standard".
    :param check_in_date: Check-in date, e.g. "2026-06-20".
    :param check_out_date: Check-out date, e.g. "2026-06-23".
    """
    return f"{room_type} rooms are available from {check_in_date} to {check_out_date}."


def main():
    agents_client = AgentsClient(endpoint=ENDPOINT, credential=DefaultAzureCredential())
    function_tool = FunctionTool(functions={check_room_availability})
    agents_client.enable_auto_function_calls({check_room_availability})

    agent = agents_client.create_agent(
        model=DEPLOYMENT,
        name="function-tool-demo",
        instructions="You are a Crystal Hotels booking assistant. Use the available tool to answer.",
        tools=function_tool.definitions,
    )
    thread = agents_client.threads.create()
    agents_client.messages.create(
        thread_id=thread.id,
        role="user",
        content="Is a Deluxe room available from 2026-06-20 to 2026-06-23?",
    )
    agents_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

    reply = agents_client.messages.get_last_message_text_by_role(thread_id=thread.id, role="assistant")
    print(f"Agent: {reply.text.value}")

    agents_client.delete_agent(agent.id)
    agents_client.threads.delete(thread.id)


if __name__ == "__main__":
    main()
