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
    instructions="You are a data analyst. Use Python to calculate precisely.",
    input="What is the compound interest on $10,000 at 5 percent annual rate over 10 years?",
    tools=[{"type": "code_interpreter", "container": {"type": "auto"}}]
)

# Inspect what happened under the hood
for item in response.output:
    if item.type == "code_interpreter_call":
        print("=== Python Code the Model Wrote ===")
        print(item.code)
        print("\n=== Output from Execution ===")
        print(item.outputs)
    elif item.type == "message":
        print("\n=== Final Answer ===")
        print(response.output_text)