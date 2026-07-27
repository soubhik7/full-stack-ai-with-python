# =============================================================================
# LAB 18 (NEW — added to fill an AI-103 syllabus gap, not an original course
# file): CONNECTED AGENTS — Foundry's native multi-agent DELEGATION pattern.
#
# This is different from Lab 11's SequentialBuilder (which PIPES agents one
# after another: A's output becomes B's input). Here, ONE orchestrator agent
# ("concierge") has other, already-created agents attached to it AS TOOLS -
# it decides, per user request, which specialist(s) to call, in whatever
# order/combination the conversation needs, then combines their answers
# itself. This is the pattern EXAM_NOTES.md lists as `ConnectedAgentTool`
# under "Orchestration patterns" (alongside sequential/concurrent/group-chat/
# handoff).
#
# This script is adapted from a REAL, working example already in this repo:
# `11_azure_ai_foundry/06_connected_agents/main.py` (a "Crystal Hotels"
# booking assistant) - ported here to fit this Labs folder's style and env
# vars, using the exact same SDK classes (`AgentsClient`, `ConnectedAgentTool`).
#
# Prerequisites: Azure AI Foundry project + `az login` + .env with
# PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME.
# Uses `azure-ai-agents` (already in root requirements.txt, added for
# chapter 11) rather than `azure-ai-projects` - a lower-level client aimed
# specifically at creating/running agents and threads directly.
# =============================================================================

import os                      # Env vars, clear-screen command
from dotenv import load_dotenv  # Loads .env file variables into the environment

from azure.ai.agents import AgentsClient                    # Lower-level agent/thread client (distinct from AIProjectClient)
from azure.ai.agents.models import ConnectedAgentTool        # Wraps an already-created agent so ANOTHER agent can call it as a tool
from azure.identity import DefaultAzureCredential            # Authenticates using your `az login` session


def main():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        load_dotenv()
        project_endpoint = os.getenv("PROJECT_ENDPOINT")
        model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

        agents_client = AgentsClient(
            endpoint=project_endpoint,
            credential=DefaultAzureCredential(),
        )

        # Create the two SPECIALIST agents first
        # Create the two specialist agents first
        #
        # Each is a small, single-purpose agent - notice there's no
        # coordination logic in either of them; they just answer their own
        # narrow question when asked.
        availability_agent = agents_client.create_agent(
            model=model_deployment,
            name="availability-checker",
            instructions=(
                "You check whether a hotel room type is available for given dates. "
                "Since this is a demo, invent a plausible-sounding availability answer "
                "(e.g. 'Yes, 3 Deluxe rooms are available for those dates.')."
            ),
        )
        payment_agent = agents_client.create_agent(
            model=model_deployment,
            name="payment-calculator",
            instructions=(
                "You calculate the total cost of a hotel stay, including a flat 12% tax. "
                "Given a nightly rate and number of nights, show the subtotal, tax, and total."
            ),
        )
        print(f"Created specialist agents: {availability_agent.name}, {payment_agent.name}")

        # Wrap each specialist agent as a ConnectedAgentTool
        # Wrap each specialist agent as a ConnectedAgentTool
        #
        # `id=` points at the specialist agent's own ID (it must already
        # exist - you can't connect to an agent that hasn't been created
        # yet). `name`/`description` are what the ORCHESTRATOR agent sees -
        # exactly like a FunctionTool's name/description, this is how the
        # model decides WHEN to delegate to this specialist.
        availability_tool = ConnectedAgentTool(
            id=availability_agent.id,
            name="check_availability",
            description="Check whether a room type is available at the hotel for given dates.",
        )
        payment_tool = ConnectedAgentTool(
            id=payment_agent.id,
            name="calculate_total_cost",
            description="Calculate the total cost of a hotel stay including tax.",
        )

        # Create the ORCHESTRATOR agent with both specialists attached
        # Create the orchestrator agent with both specialists attached
        #
        # Note `.definitions` (plural) - a ConnectedAgentTool exposes a LIST
        # of tool definitions (not the tool object itself) to pass into
        # `tools=`, which is why the two lists are concatenated with `+`.
        concierge = agents_client.create_agent(
            model=model_deployment,
            name="concierge",
            instructions=(
                "You are a hotel concierge. Use your connected agents to check room "
                "availability and calculate costs as needed, then combine their answers "
                "into one friendly reply to the guest."
            ),
            tools=availability_tool.definitions + payment_tool.definitions,
        )
        print(f"Created orchestrator agent: {concierge.name}\n")

        # This client uses the older thread/run model (not the
        # conversations/responses model the other agent labs use) - a
        # thread holds message history, and a "run" is one execution of the
        # agent against that thread.
        thread = agents_client.threads.create()

        user_question = (
            "I'd like a Deluxe room for 3 nights at $180/night starting next Friday - "
            "is it available, and what would it cost in total?"
        )
        print(f"USER: {user_question}\n")
        agents_client.messages.create(thread_id=thread.id, role="user", content=user_question)

        # create_and_process() runs the agent to completion in one call -
        # internally it starts the run, and whenever the orchestrator wants
        # to call a ConnectedAgentTool, the SDK automatically runs the
        # target specialist agent and feeds its reply back, all before this
        # call returns control to you (unlike the function-calling labs,
        # you don't manually loop over function_call items here).
        agents_client.runs.create_and_process(thread_id=thread.id, agent_id=concierge.id)

        reply = agents_client.messages.get_last_message_text_by_role(thread_id=thread.id, role="assistant")
        print(f"CONCIERGE: {reply.text.value if reply else '(no reply)'}")

        # Clean up all three agents created by this script
        # Clean up all three agents created by this script
        print("\nCleaning up agents...")
        agents_client.delete_agent(concierge.id)
        agents_client.delete_agent(availability_agent.id)
        agents_client.delete_agent(payment_agent.id)
        print("Done.")

    except Exception as ex:
        # Catch-all: print any error instead of crashing
        print(ex)


if __name__ == "__main__":
    main()
