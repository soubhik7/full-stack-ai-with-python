####################################################################################################
# LAB 5 — HOSTED AGENT (Azure AI Foundry Agent Service — persistent agent + thread)
#
# WHY THIS MATTERS
#   Multi-turn assistants need memory of earlier turns. A hosted agent gets that for free:
#   Azure stores the agent's config and the thread's full message history server-side, under
#   IDs (agent_id, thread_id) you can reconnect to later from any process.
#
# HOW IT LINKS TO THE REST OF THE CHAPTER
#   Direct contrast with 04_prompt_agent/main.py: same hotel-assistant scenario, but there the
#   second question ("what did I just ask?") drew a blank — here the second question ("what
#   about late checkout?") correctly builds on the first because the thread remembers it.
#   06_connected_agents/ and 07_knowledge_rag/ both build on this exact create_agent/thread/run
#   pattern, just with extra tools attached.
#
# HOW IT WORKS
#   create_agent() registers the agent itself (its model + instructions) on Azure — separate
#   from any one conversation. threads.create() opens a conversation; messages.create() adds a
#   turn to it; runs.create_and_process() is what actually invokes the model against the whole
#   thread so far. Because both calls to ask() reuse the same thread.id, the second run sees the
#   first turn automatically — no message history is passed manually, unlike 04_prompt_agent/.
####################################################################################################
import os

from azure.ai.agents import AgentsClient
from azure.ai.agents.models import ListSortOrder
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
DEPLOYMENT = os.environ["AZURE_AI_MODEL_DEPLOYMENT"]


def ask(agents_client, thread_id: str, agent_id: str, user_message: str) -> str:
    """A turn in a persistent hosted agent thread. The thread (not the caller) carries history."""
    agents_client.messages.create(thread_id=thread_id, role="user", content=user_message)
    agents_client.runs.create_and_process(thread_id=thread_id, agent_id=agent_id)
    reply = agents_client.messages.get_last_message_text_by_role(thread_id=thread_id, role="assistant")
    return reply.text.value


def main():
    agents_client = AgentsClient(endpoint=ENDPOINT, credential=DefaultAzureCredential())

    agent = agents_client.create_agent(
        model=DEPLOYMENT,
        name="hosted-agent-demo",
        instructions="You are a concise Crystal Hotels assistant.",
    )
    thread = agents_client.threads.create()
    print(f"Created agent_id={agent.id}, thread_id={thread.id} — both persist on Azure until deleted.\n")

    print(ask(agents_client, thread.id, agent.id, "What time is checkout?"))
    # Same thread, no need to repeat context — the Agent Service remembers the prior turn.
    print(ask(agents_client, thread.id, agent.id, "And what about for late checkout requests?"))

    print("\nFull thread history (server-side, retrievable any time by thread_id):")
    for message in agents_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING):
        if message.text_messages:
            print(f"  [{message.role}] {message.text_messages[-1].text.value}")

    agents_client.delete_agent(agent.id)
    agents_client.threads.delete(thread.id)
    print(f"\nCleaned up: deleted {agent.name} (in production, hosted agents and threads are left running)")


if __name__ == "__main__":
    main()
