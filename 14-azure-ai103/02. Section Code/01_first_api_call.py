from openai import OpenAI

endpoint = "https://integration-pulse-found-resource.openai.azure.com/openai/v1"
deployment_name = "gpt-4.1"
api_key = "<API_KEY>"

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
