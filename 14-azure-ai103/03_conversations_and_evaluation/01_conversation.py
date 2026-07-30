import sys
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

PROJECT_ENDPOINT = config.project_endpoint
AGENT_NAME = config.agent_name("cloudxeus-support-agent-conv")

client=AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential()
)

# Sara's thread
openai=client.get_openai_client()
sara=openai.conversations.create()
print(sara.id)

response=openai.responses.create(
    conversation=sara.id,
    extra_body={"agent_reference":{"name":AGENT_NAME,"type":"agent_reference"}},
    input="My order #4521 is late."
)

print(response.output_text)

response=openai.responses.create(
    conversation=sara.id,
    extra_body={"agent_reference":{"name":AGENT_NAME,"type":"agent_reference"}},
    input="Any update on it?"
)

print(response.output_text)