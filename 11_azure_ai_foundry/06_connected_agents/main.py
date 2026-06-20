####################################################################################################
# LAB 6 — CONNECTED AGENTS (multi-agent orchestration via ConnectedAgentTool)
#
# WHY THIS MATTERS
#   One agent with a 500-line instruction prompt trying to do everything is hard to maintain and
#   hard to get right. Splitting work into focused sub-agents (each with a narrow job and short
#   instructions) and letting a parent agent delegate to them mirrors the real Crystal Hotels
#   pipeline in the study notes (§61, Figure 61.2): booking validation -> availability check ->
#   payment processing -> confirmation. This lab implements the middle two steps.
#
# HOW IT LINKS TO THE REST OF THE CHAPTER
#   Every agent here (sub-agents AND the parent) is built with the exact same create_agent/
#   thread/run calls as 05_hosted_agent/ — "connected" is just a tool type (ConnectedAgentTool)
#   wrapping an existing hosted agent, not a different kind of agent. This is also a different,
#   Foundry-specific flavor of multi-agent orchestration from the generic
#   08_ai_apps/17_multi_agent/ patterns elsewhere in this repo.
#
# HOW IT WORKS
#   availability_agent and payment_agent are created first, standalone. ConnectedAgentTool(id=...,
#   name=..., description=...) wraps each one so it can be listed in another agent's `tools=`
#   just like a function tool. The `concierge` agent never sees the sub-agents' instructions —
#   it just sees two tools named "check_availability" and "calculate_total_cost". When the model
#   decides to call one, Azure routes that call to the matching sub-agent and returns its reply
#   automatically; no manual execution loop, the same as the built-in tools in 02_tools/.
####################################################################################################
import os
import re

from azure.ai.agents import AgentsClient
from azure.ai.agents.models import ConnectedAgentTool
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

# Connected-agent runs occasionally have the model emit file-search-style citation
# markers even though no files are involved here. Strip them for clean output.
CITATION_MARKER = re.compile(r"【[^】]*】")

ENDPOINT = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
DEPLOYMENT = os.environ["AZURE_AI_MODEL_DEPLOYMENT"]


def main():
    agents_client = AgentsClient(endpoint=ENDPOINT, credential=DefaultAzureCredential())

    availability_agent = agents_client.create_agent(
        model=DEPLOYMENT,
        name="availability-checker",
        instructions=(
            "You check Crystal Hotels room availability. Deluxe and Standard rooms are always "
            "available unless the question mentions a Suite, which is sold out. Answer in one sentence."
        ),
    )
    payment_agent = agents_client.create_agent(
        model=DEPLOYMENT,
        name="payment-calculator",
        instructions=(
            "You calculate Crystal Hotels stay totals. Given a nightly rate, number of nights, and "
            "tax rate, compute the exact total. Answer in one sentence with the final number."
        ),
    )

    availability_tool = ConnectedAgentTool(
        id=availability_agent.id,
        name="check_availability",
        description="Check whether a room type is available at Crystal Hotels.",
    )
    payment_tool = ConnectedAgentTool(
        id=payment_agent.id,
        name="calculate_total_cost",
        description="Calculate the total cost of a Crystal Hotels stay including tax.",
    )

    concierge = agents_client.create_agent(
        model=DEPLOYMENT,
        name="concierge",
        instructions=(
            "You are the Crystal Hotels concierge. Use your connected agents to check availability "
            "and calculate costs, then combine their answers into one reply to the guest."
        ),
        tools=availability_tool.definitions + payment_tool.definitions,
    )

    thread = agents_client.threads.create()
    agents_client.messages.create(
        thread_id=thread.id,
        role="user",
        content=(
            "I want a Deluxe room for 5 nights at $189/night with 12% tax. "
            "Is it available, and what's the total cost?"
        ),
    )
    agents_client.runs.create_and_process(thread_id=thread.id, agent_id=concierge.id)

    reply = agents_client.messages.get_last_message_text_by_role(thread_id=thread.id, role="assistant")
    print(f"Concierge: {CITATION_MARKER.sub('', reply.text.value)}")

    agents_client.threads.delete(thread.id)
    for agent in (concierge, payment_agent, availability_agent):
        agents_client.delete_agent(agent.id)


if __name__ == "__main__":
    main()
