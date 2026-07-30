from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
import base64
import sys
from pathlib import Path

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

AGENT_NAME = config.agent_name("cloudxeus-support")

project = AIProjectClient(
    endpoint=config.project_endpoint,
    credential=DefaultAzureCredential(),
)

openai = project.get_openai_client()

with open("support.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

response = openai.responses.create(
    extra_body={"agent_reference": {"name": AGENT_NAME, "type": "agent_reference"}},
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "I'm getting this error connecting to the VPN. What does it mean?"},
                {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"},
            ],
        }
    ],
)

print(response.output_text)
print(response.model_extra["content_filters"])