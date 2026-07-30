import os

import requests
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

load_dotenv()

credential = DefaultAzureCredential()

PROJECT_RESOURCE_ID = (
    "/subscriptions/a2a03c7b-e15c-45c6-9ddd-dfa0552e2007"
    "/resourceGroups/soubhik-demo-rg"
    "/providers/Microsoft.CognitiveServices/accounts/soubhik-demo-foundry-1"
    "/projects/soubhik-demo-project"
)

PROJECT_CONNECTION_NAME = "soubhik-demo-funcapp-mcp-connection"

FUNCTION_APP_HOST = "soubhik-demo-funcapp.azurewebsites.net"
MCP_SYSTEM_KEY = os.environ["MCP_SYSTEM_KEY"]

mcp_endpoint = f"https://{FUNCTION_APP_HOST}/runtime/webhooks/mcp/sse"

bearer_token_provider = get_bearer_token_provider(
    credential,
    "https://management.azure.com/.default"
)

headers = {
    "Authorization": f"Bearer {bearer_token_provider()}",
    "Content-Type": "application/json"
}

response = requests.put(
    f"https://management.azure.com{PROJECT_RESOURCE_ID}/connections/{PROJECT_CONNECTION_NAME}"
    "?api-version=2025-10-01-preview",
    headers=headers,
    json={
        "name": PROJECT_CONNECTION_NAME,
        "type": "Microsoft.MachineLearningServices/workspaces/connections",
        "properties": {
            "authType": "CustomKeys",
            "category": "RemoteTool",
            "target": mcp_endpoint,
            "isSharedToAll": True,
            "credentials": {
                "keys": {
                    "x-functions-key": MCP_SYSTEM_KEY
                }
            },
            "metadata": {
                "ApiType": "Azure",
                "McpTransport": "sse"
            }
        }
    }
)

print(response.status_code)
print(response.text)

response.raise_for_status()

print(f"Connection '{PROJECT_CONNECTION_NAME}' created or updated successfully.")
