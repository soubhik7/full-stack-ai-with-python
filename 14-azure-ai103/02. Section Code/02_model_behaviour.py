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
    instructions="You are a creative copywriter.",
    input="Write a two-sentence tagline for a new AI-powered productivity app.",
    #max_output_tokens=50,
    temperature=2
)


print(f"answer: {response.output_text}")
