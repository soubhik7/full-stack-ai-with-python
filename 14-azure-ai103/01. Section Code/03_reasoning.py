from openai import OpenAI, BadRequestError
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

endpoint = "https://integration-pulse-found-resource.services.ai.azure.com/openai/v1"
deployment_name = "gpt-4.1"
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
