####################################################################################################
# Foundry model (soubhik-demo-foundry-1 / soubhik-demo-gpt-mini, i.e. gpt-5-mini) calling the
# soubhik-demo-funcapp MCP server (GetOrderStatusTool, ListCustomerOrdersTool) as a tool.
#
# gpt-5-mini only supports MCP tools through the Responses API (the classic Assistants-style
# threads/runs Agent Service — azure-ai-agents' AgentsClient — rejected the "mcp" tool type with
# "This model only supports Responses API compatible tools"). So this uses the `openai` SDK's
# AzureOpenAI client against the Foundry resource's OpenAI-compatible endpoint instead.
####################################################################################################
import os

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

PROJECT_ENDPOINT = "https://soubhik-demo-foundry-1.services.ai.azure.com/api/projects/soubhik-demo-project"
DEPLOYMENT = "soubhik-demo-gpt-mini"

FUNCTION_APP_HOST = "soubhik-demo-funcapp.azurewebsites.net"
MCP_SYSTEM_KEY = os.environ["MCP_SYSTEM_KEY"]

mcp_tool = {
    "type": "mcp",
    "server_label": "soubhik_demo_funcapp_mcp",
    "server_url": f"https://{FUNCTION_APP_HOST}/runtime/webhooks/mcp/sse",
    "headers": {"x-functions-key": MCP_SYSTEM_KEY},
    "require_approval": "never",
}


def main():
    project_client = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential())
    client = project_client.get_openai_client()

    response = client.responses.create(
        model=DEPLOYMENT,
        input="What's the status of order ORD-1042?",
        tools=[mcp_tool],
    )

    for item in response.output:
        print(f"[{item.type}]", getattr(item, "name", "") or "")

    print("\nFinal answer:")
    print(response.output_text)


if __name__ == "__main__":
    main()
