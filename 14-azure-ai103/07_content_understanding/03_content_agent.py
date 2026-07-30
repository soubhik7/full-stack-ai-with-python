import sys
from pathlib import Path

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

from azure.core.credentials import AzureKeyCredential
from azure.ai.contentunderstanding import ContentUnderstandingClient
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

endpoint = config.content_understanding_endpoint
api_key = config.content_understanding_key

client=ContentUnderstandingClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(api_key) if api_key else DefaultAzureCredential()
)

file_path = "cloudxeus_sample_invoice.pdf"

with open(file_path, "rb") as file:
    poller=client.begin_analyze_binary(
        analyzer_id="prebuilt-invoice",
        binary_input=file,
        content_type="application/pdf"
    )

result = poller.result()
print("Analysis completed.")

content = result["contents"][0]
fields = content["fields"]

invoice_details = ""

for field_name, field_data in fields.items():
    invoice_details += f"{field_name}: {field_data}\n"

# Calling the agent

PROJECT_ENDPOINT = config.project_endpoint
AGENT_NAME = config.agent_name("cloudxeus-support")
project = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(),
)

openai = project.get_openai_client()

response = openai.responses.create( 
    extra_body={
        "agent_reference": {
            "name": AGENT_NAME,
            "type": "agent_reference"
        }
    },
    input=f"""
Review this invoice using only the extracted fields below.

Your task:
1. Summarize the invoice in business-friendly language.
2. Provide an approval status.
3. Identify any issues found.
4. Recommend the next step.

Extracted invoice fields:
{invoice_details}
"""
)

print("\n--- Agent Invoice Review ---")
print(response.output_text)
