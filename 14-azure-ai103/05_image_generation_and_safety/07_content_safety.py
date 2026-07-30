# pip3 install azure.ai.contentsafety
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeImageOptions, ImageData
import sys
from pathlib import Path

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

content_safety_client = ContentSafetyClient(
    endpoint=config.content_safety_endpoint,
    credential=AzureKeyCredential(config.content_safety_key)
)
IMAGE_PATH = "support.png"

with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()

image_request = AnalyzeImageOptions(
    image=ImageData(content=image_bytes)
)

moderation_result = content_safety_client.analyze_image(image_request)

print("Image moderation result:")
for category_result in moderation_result.categories_analysis:
    print(category_result.category, category_result.severity)