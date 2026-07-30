import sys
from pathlib import Path

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.ai.contentunderstanding import ContentUnderstandingClient

endpoint = config.content_understanding_endpoint
api_key = config.content_understanding_key

client=ContentUnderstandingClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(api_key) if api_key else DefaultAzureCredential()
)

file_path = "support_ticket_portal.png"

with open(file_path, "rb") as file:
    poller=client.begin_analyze_binary(
        analyzer_id="prebuilt-imageSearch",
        binary_input=file,
        content_type="image/png"
    )
result = poller.result()
print("Analysis completed.")

content = result["contents"][0]

print("\n--- Image Analysis Result ---")
print(content)

client.close()