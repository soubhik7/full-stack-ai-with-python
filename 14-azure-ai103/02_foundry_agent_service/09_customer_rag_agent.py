import sys
from pathlib import Path

from azure.identity import AzureCliCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

PROJECT_ENDPOINT = config.project_endpoint

SYSTEM_PROMPT = """
You are a customer support assistant for CloudXeus Technology Services.

Answer the customer's question using ONLY the provided sources.

After your answer, cite the source URL you used.

If the sources do not contain the answer, say:
"I don't have that information in the available knowledge base."

Then suggest contacting support@cloudxeus.com.

Never invent policies, prices, refund rules, or timelines.
"""

project = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=AzureCliCredential()
)

agent = project.agents.create_version(
    agent_name=config.agent_name("cloudxeus-support-rag-agent"),
    definition=PromptAgentDefinition(
        model=config.model_deployment,
        instructions=SYSTEM_PROMPT,
    ),
)