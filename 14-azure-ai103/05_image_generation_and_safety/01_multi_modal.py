from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
import base64
import sys
from pathlib import Path

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

project = AIProjectClient(
    endpoint=config.project_endpoint,
    credential=DefaultAzureCredential(),
)

openai = project.get_openai_client()

with open("sales_data.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

response = openai.responses.create(
    model=config.model_deployment,
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text",
             "text": "Generate a summary based on the content in the attached image"},
            {"type": "input_image",
             "image_url": f"data:image/png;base64,{b64}"},
        ],
    }],
)

print(response.output_text)