####################################################################################################
# A Foundry model (gpt-5-mini-class deployment, see AZURE_AI_MODEL_DEPLOYMENT in .env) calling
# a Function App MCP server (GetOrderStatusTool, ListCustomerOrdersTool) as a tool.
#
# gpt-5-mini only supports MCP tools through the Responses API (the classic Assistants-style
# threads/runs Agent Service — azure-ai-agents' AgentsClient — rejected the "mcp" tool type with
# "This model only supports Responses API compatible tools"). So this uses the `openai` SDK's
# AzureOpenAI client against the Foundry resource's OpenAI-compatible endpoint instead.
####################################################################################################
import sys
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

PROJECT_ENDPOINT = config.project_endpoint
DEPLOYMENT = config.model_deployment

FUNCTION_APP_HOST = config.function_app_host
MCP_SYSTEM_KEY = config.mcp_system_key

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
