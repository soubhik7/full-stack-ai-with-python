import sys
from pathlib import Path

from openai import OpenAI, BadRequestError
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

problem = """
A distributed e-commerce system is experiencing intermittent checkout failures 
during peak traffic. The failures appear random, affect roughly 3 percent of the transactions, 
and only occur when inventory checks and payment processing run concurrently. 
Identify the most likely root cause and propose a solution.
"""

try:
    response = client.responses.create(
        model=deployment_name,
        instructions="You are a senior software architect.",
        input=problem,
        reasoning={"effort": "high"}
    )
except BadRequestError as exc:
    if "reasoning.effort" in str(exc):
        response = client.responses.create(
            model=deployment_name,
            instructions="You are a senior software architect.",
            input=problem,
        )
    else:
        raise

print(f"answer: {response.output_text}")
