import sys
from pathlib import Path

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions

CONTENT_SAFETY_ENDPOINT = config.content_safety_endpoint
CONTENT_SAFETY_KEY = config.content_safety_key

endpoint = config.document_intelligence_endpoint
key = config.document_intelligence_key

document_url = "https://stcxai103capdeus01.blob.core.windows.net/course-products/student-invoices/CloudXeus_Invoice_INV-CX-2026-1001.pdf"

client = DocumentIntelligenceClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key),
    api_version="2024-11-30"
)

poller = client.begin_analyze_document(
    model_id="prebuilt-layout",
    body=AnalyzeDocumentRequest(url_source=document_url),
    output_content_format="markdown"
)

document_result = poller.result()

extracted_text = document_result.content

content_safety_client = ContentSafetyClient(
    endpoint=CONTENT_SAFETY_ENDPOINT,
    credential=AzureKeyCredential(CONTENT_SAFETY_KEY)
)

text_request = AnalyzeTextOptions(
    text=extracted_text
)

moderation_result = content_safety_client.analyze_text(text_request)

print("\nContent Safety result:")
should_block = False

for category_result in moderation_result.categories_analysis:
    print(category_result.category, category_result.severity)
if category_result.severity >= 4:
        should_block = True


if should_block:
    print("\nBlocked: unsafe content was detected in the product PDF.")
else:
    print("\nAllowed: the product PDF text can be passed to the agent.")

