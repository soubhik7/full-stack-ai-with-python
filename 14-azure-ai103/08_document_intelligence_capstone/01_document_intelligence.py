# pip3 install azure.ai.documentintelligence
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

endpoint = config.document_intelligence_endpoint
key = config.document_intelligence_key

document_url = "https://stcxai103capdeus01.blob.core.windows.net/course-products/student-invoices/CloudXeus_Invoice_INV-CX-2026-1002.pdf"

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

result = poller.result()

print(result.content)