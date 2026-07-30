from openai import OpenAI
import base64
import sys
from pathlib import Path

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

IMAGE_DEPLOYMENT_NAME = config.image_deployment

client = OpenAI(
    base_url=config.image_endpoint,
    api_key=config.image_api_key
)


response=client.images.generate(
    model=IMAGE_DEPLOYMENT_NAME,
    prompt=(
        "Create a professional training image for an online course. "
        "Show a modern AI application dashboard with charts, documents, "
        "and an AI assistant helping business users. "
        "Use a clean corporate style suitable for a Microsoft Azure AI course."
    ),
    n=1,
    size="1024x1024",
    quality="medium",
    output_format="png"
)

image_base64 = response.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

output_path = Path("generated_image.png")
output_path.write_bytes(image_bytes)

print(f"Image saved to: {output_path}")
