from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


endpoint = "https://integration-pulse-found-resource.services.ai.azure.com/openai/v1"
deployment_name = "gpt-4.1"
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")

client = OpenAI(
    base_url=endpoint,
    api_key=token_provider
)

response = client.responses.create(
    model=deployment_name,
    instructions="You are a helpful research assistant. Always cite your sources.",
    input="What are the latest developments in AI regulation in the European Union?",
    tools=[{"type": "web_search"}],
    tool_choice="auto"
)

print(f"answer: {response.output_text}")
