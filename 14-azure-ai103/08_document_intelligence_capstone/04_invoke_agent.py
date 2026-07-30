import json
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
from azure.ai.contentunderstanding.models import AnalysisInput
from azure.ai.projects import AIProjectClient

CONTENT_UNDERSTANDING_ENDPOINT = config.content_understanding_endpoint
CONTENT_UNDERSTANDING_KEY = config.content_understanding_key

ANALYZER_ID = config.cu_analyzer_id

INVOICE_BLOB_URL = "https://stcxai103capdeus01.blob.core.windows.net/course-products/student-invoices/CloudXeus_Invoice_INV-CX-2026-1003.pdf"

PROJECT_ENDPOINT = config.project_endpoint
AGENT_NAME = config.agent_name("CloudXeus-Invoice-Intelligence-Agent")


credential = (
    AzureKeyCredential(CONTENT_UNDERSTANDING_KEY)
    if CONTENT_UNDERSTANDING_KEY
    else DefaultAzureCredential()
)

content_client = ContentUnderstandingClient(
    endpoint=CONTENT_UNDERSTANDING_ENDPOINT,
    credential=credential
)

poller = content_client.begin_analyze(
    analyzer_id=ANALYZER_ID,
    inputs=[
        AnalysisInput(url=INVOICE_BLOB_URL)
    ]
)

invoice_result = poller.result()

print("Invoice analysis completed.")

project_client = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential()
)

openai_client = project_client.get_openai_client()

prompt = f"""
Create a complete invoice intelligence report.

Use the extracted invoice data below.
When you need product information, use the CloudXeus product knowledge base.

Include:
1. Invoice overview
2. Products purchased
3. Product explanations from the knowledge base
4. Product match validation
5. Finance summary
6. Customer-friendly explanation
7. Recommended sales follow-up

Extracted invoice data:
{json.dumps(invoice_result.as_dict(), indent=2)}
"""

response = openai_client.responses.create(
    extra_body={
        "agent_reference": {
            "name": AGENT_NAME,
            "type": "agent_reference"
        }
    },
    input=prompt
)

print("\n----- Agent Report -----\n")
print(response.output_text)