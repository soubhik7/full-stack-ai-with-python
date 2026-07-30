from openai import OpenAI
import base64
import sys
from pathlib import Path
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

endpoint = config.openai_endpoint
deployment_name = config.openai_deployment
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")

client = OpenAI(
    base_url=endpoint,
    api_key=token_provider
)


script_dir = Path(__file__).resolve().parent
image_path = script_dir / "Agent_types.png"

if not image_path.exists():
    raise FileNotFoundError(f"Image file not found: {image_path}")

print(f"Loading image from: {image_path}")
with open(image_path, "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode("utf-8")

print("Sending multimodal request to Azure OpenAI...")
response = client.responses.create(
    model=deployment_name,
    instructions="You are a helpful assistant that reads and extracts information from images.",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_data}"
                },
                {
                    "type": "input_text",
                    "text": "Extract all the text from this image and present it in a structured, readable format."
                }
            ]
        }
    ]
)

print(f"answer: {response.output_text}")
