import sys
from pathlib import Path

from openai import OpenAI

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

endpoint = config.openai_endpoint
deployment_name = config.openai_deployment
api_key = config.openai_api_key

client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)

response = client.responses.create(
    model=deployment_name,
    input="What are the three main benefits of using managed AI endpoints in the cloud?",
)

print(f"answer: {response.output_text}")


#--------------
#from openai import OpenAI
#from azure.identity import DefaultAzureCredential, get_bearer_token_provider

#endpoint = "https://integration-pulse-found-resource.services.ai.azure.com/openai/v1"
#deployment_name = "gpt-4.1"
#token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")

#client = OpenAI(
#    base_url=endpoint,
#    api_key=token_provider
#)

#response = client.responses.create(
#    model=deployment_name,
#    input="What is the capital of France?",
#)

#   print(f"answer: {response.output[0]}")
