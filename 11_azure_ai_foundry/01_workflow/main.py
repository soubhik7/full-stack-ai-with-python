####################################################################################################
# LAB 1 — THE FOUNDRY WORKFLOW (Model Catalog -> Playground -> Agents -> Projects -> Deployments)
#
# WHY THIS MATTERS
#   Study notes §7.2/§7.3 describe Azure AI Foundry as five components walked in a fixed order:
#   pick a model, test it, configure an agent, organize it under a project, ship it. That's a
#   portal tour. This script is the same five steps done with code instead of clicks, so the
#   workflow is something you can run, not just read about.
#
# HOW IT LINKS TO THE REST OF THE CHAPTER
#   Step 2 (Playground) is the same stateless call built out fully in 04_prompt_agent/.
#   Step 3 (Agents) is the same create_agent -> thread -> run pattern built out fully in
#   05_hosted_agent/. This lab only needs to be read once — it's the map; 02-07 are the territory.
#
# HOW IT WORKS
#   AIProjectClient.get_openai_client() handles step 2 (no agent, no thread — just a chat call).
#   AgentsClient handles steps 3-4 (create_agent/threads/runs for "Agents", list_agents for
#   "Projects" — listing what's registered is the closest data-plane analogue to the portal's
#   Projects view). Step 5 has no SDK call: "Deployments" just is the endpoint already in use.
####################################################################################################
import os

from azure.ai.agents import AgentsClient
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
DEPLOYMENT = os.environ["AZURE_AI_MODEL_DEPLOYMENT"]
credential = DefaultAzureCredential()


def step_1_model_catalog():
    print("\n[1/5] Model Catalog — choose a model for the scenario")
    print(f"      Using deployment: {DEPLOYMENT} (selected in the Foundry portal's Model Catalog)")


def step_2_playground(openai_client):
    print("\n[2/5] Playground — test the model with a quick, stateless call")
    response = openai_client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": "In one short sentence, what is Azure AI Foundry?"}],
    )
    print(f"      Model says: {response.choices[0].message.content}")


def step_3_agents(agents_client):
    print("\n[3/5] Agents — configure and test an agent")
    agent = agents_client.create_agent(
        model=DEPLOYMENT,
        name="workflow-tour-agent",
        instructions="You are a concise assistant for Crystal Hotels. Answer in one sentence.",
    )
    thread = agents_client.threads.create()
    agents_client.messages.create(thread_id=thread.id, role="user", content="What time is checkout?")
    agents_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
    reply = agents_client.messages.get_last_message_text_by_role(thread_id=thread.id, role="assistant")
    print(f"      Agent says: {reply.text.value}")
    agents_client.threads.delete(thread.id)
    return agent


def step_4_projects(agents_client):
    print("\n[4/5] Projects — see everything organized under this project")
    agents = list(agents_client.list_agents())
    print(f"      {len(agents)} agent(s) currently registered in this project:")
    for a in agents:
        print(f"        - {a.name} ({a.id})")


def step_5_deployments():
    print("\n[5/5] Deployments — this is the production endpoint already in use")
    print(f"      Endpoint: {ENDPOINT}")
    print(f"      Deployment: {DEPLOYMENT}")


def main():
    project_client = AIProjectClient(endpoint=ENDPOINT, credential=credential)
    agents_client = AgentsClient(endpoint=ENDPOINT, credential=credential)

    step_1_model_catalog()
    step_2_playground(project_client.get_openai_client())
    agent = step_3_agents(agents_client)
    step_4_projects(agents_client)
    step_5_deployments()

    agents_client.delete_agent(agent.id)
    print(f"\nCleaned up: deleted {agent.name}")


if __name__ == "__main__":
    main()
