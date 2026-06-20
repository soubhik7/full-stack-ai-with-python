####################################################################################################
# LAB 7 — KNOWLEDGE / RAG GROUNDING (file search over an uploaded document)
#
# WHY THIS MATTERS
#   An agent's instructions can describe a persona, but real policy facts (exact cancellation
#   windows, fees) shouldn't be typed into a prompt — they live in documents that change, and a
#   model asked to recall them will happily guess a plausible-but-wrong number. File search
#   grounds answers in an actual uploaded file instead of the model's prior "knowledge".
#
# HOW IT LINKS TO THE REST OF THE CHAPTER
#   FileSearchTool slots into create_agent(tools=..., tool_resources=...) exactly like
#   CodeInterpreterTool did in 02_tools/code_interpreter_tool.py — same shape, different tool.
#   The instructions below are written deliberately strict ("you MUST call file_search") because
#   testing this lab live surfaced a real failure mode: with softer instructions, the model
#   answered confidently from general knowledge and skipped calling the tool entirely, giving a
#   plausible but wrong answer (24-hour cancellation window instead of this file's actual 72).
#
# HOW IT WORKS
#   files.upload_and_poll() pushes data/hotel_policy.md to Azure; vector_stores.create_and_poll()
#   chunks and embeds it. FileSearchTool(vector_store_ids=[...]) is what create_agent reads to
#   know which vector store to search. Cleanup deletes the agent, vector store, and file at the
#   end — unlike 05_hosted_agent/, there's no reason to keep a demo-only knowledge base around.
####################################################################################################
import os
import re

from azure.ai.agents import AgentsClient
from azure.ai.agents.models import FilePurpose, FileSearchTool
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
DEPLOYMENT = os.environ["AZURE_AI_MODEL_DEPLOYMENT"]
POLICY_DOC = os.path.join(os.path.dirname(__file__), "data", "hotel_policy.md")
CITATION_MARKER = re.compile(r"【[^】]*】")


def main():
    agents_client = AgentsClient(endpoint=ENDPOINT, credential=DefaultAzureCredential())

    uploaded_file = agents_client.files.upload_and_poll(file_path=POLICY_DOC, purpose=FilePurpose.AGENTS)
    vector_store = agents_client.vector_stores.create_and_poll(
        file_ids=[uploaded_file.id], name="crystal-hotels-policy"
    )
    file_search = FileSearchTool(vector_store_ids=[vector_store.id])

    agent = agents_client.create_agent(
        model=DEPLOYMENT,
        name="knowledge-rag-demo",
        instructions=(
            "You are a Crystal Hotels assistant. You have NO built-in knowledge of hotel policies. "
            "For every guest question, you MUST call the file_search tool to retrieve the relevant "
            "policy text before answering. Never answer from assumption."
        ),
        tools=file_search.definitions,
        tool_resources=file_search.resources,
    )
    thread = agents_client.threads.create()
    agents_client.messages.create(
        thread_id=thread.id,
        role="user",
        content="If I cancel my reservation 24 hours before check-in, what do I get charged?",
    )
    agents_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

    reply = agents_client.messages.get_last_message_text_by_role(thread_id=thread.id, role="assistant")
    print(f"Agent (grounded in hotel_policy.md): {CITATION_MARKER.sub('', reply.text.value)}")

    agents_client.threads.delete(thread.id)
    agents_client.delete_agent(agent.id)
    agents_client.vector_stores.delete(vector_store.id)
    agents_client.files.delete(uploaded_file.id)


if __name__ == "__main__":
    main()
