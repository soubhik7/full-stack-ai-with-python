import sys
from pathlib import Path

from openai import OpenAI
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


response = client.responses.create(
    model=deployment_name,
    instructions="You are a creative copywriter.",
    input="Write a two-sentence tagline for a new AI-powered productivity app.",
    #max_output_tokens=50,
    temperature=2
)


print(f"answer: {response.output_text}")
