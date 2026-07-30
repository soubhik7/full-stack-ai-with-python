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
AGENT_NAME = config.agent_name("IT-HelpDesk-Agent")

client=AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential()
)

openai=client.get_openai_client()

response=openai.responses.create(
    extra_body={"agent_reference":{"name":AGENT_NAME,"type":"agent_reference"}},
    input="How do I reset my company password?"
)

print(response.output_text)