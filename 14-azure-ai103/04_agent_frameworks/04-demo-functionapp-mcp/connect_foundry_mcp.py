import sys
from pathlib import Path

import requests
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

credential = DefaultAzureCredential()

PROJECT_RESOURCE_ID = config.project_resource_id

PROJECT_CONNECTION_NAME = config.project_mcp_connection_name("soubhik-demo-funcapp-mcp-connection")

FUNCTION_APP_HOST = config.function_app_host
MCP_SYSTEM_KEY = config.mcp_system_key

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
