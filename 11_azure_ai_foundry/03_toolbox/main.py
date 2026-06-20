####################################################################################################
# LAB 3 — TOOLBOX DEMO (one agent, multiple tools, asked in a single turn)
#
# WHY THIS MATTERS
#   The proof that a toolbox is useful isn't that it compiles — it's that one agent can pull
#   from several tools in the same answer without you writing per-tool wiring at call time.
#
# HOW IT LINKS TO THE REST OF THE CHAPTER
#   Imports build_toolbox()/TOOLBOX_FUNCTIONS from toolbox.py (same folder). The question below
#   deliberately needs BOTH a custom function (get_nightly_rate) and the built-in code
#   interpreter (the multiplication) to answer correctly — proving the bundle, not just one tool.
#
# HOW IT WORKS
#   build_toolbox() assembles the ToolSet; create_agent(toolset=toolset) attaches the whole thing
#   in one keyword argument. enable_auto_function_calls(TOOLBOX_FUNCTIONS) is what lets the
#   custom Python functions run locally and automatically during runs.create_and_process — the
#   code interpreter, by contrast, always runs server-side and needs no such registration.
####################################################################################################
import os

from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from toolbox import TOOLBOX_FUNCTIONS, build_toolbox

load_dotenv()

ENDPOINT = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
DEPLOYMENT = os.environ["AZURE_AI_MODEL_DEPLOYMENT"]


def main():
    agents_client = AgentsClient(endpoint=ENDPOINT, credential=DefaultAzureCredential())
    toolset = build_toolbox()
    agents_client.enable_auto_function_calls(TOOLBOX_FUNCTIONS)

    agent = agents_client.create_agent(
        model=DEPLOYMENT,
        name="toolbox-demo",
        instructions="You are a Crystal Hotels assistant. Use your tools to look up facts and compute totals.",
        toolset=toolset,
    )
    thread = agents_client.threads.create()
    agents_client.messages.create(
        thread_id=thread.id,
        role="user",
        content=(
            "What is the nightly rate for a Deluxe room, and what would 4 nights cost "
            "before tax? Use Python to compute the total."
        ),
    )
    agents_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

    reply = agents_client.messages.get_last_message_text_by_role(thread_id=thread.id, role="assistant")
    print(f"Agent: {reply.text.value}")

    agents_client.delete_agent(agent.id)
    agents_client.threads.delete(thread.id)


if __name__ == "__main__":
    main()
