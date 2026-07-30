# pip3 install azure.ai.textanalytics
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import sys
from pathlib import Path

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

client = TextAnalyticsClient(
    endpoint=config.language_endpoint,
    credential=AzureKeyCredential(config.language_key),
)

documents = [
    "Hi, this is Sarah Chen from Acme Logistics. You can reach me at "
    "sarah.chen@acmelogistics.com or call 312-555-1234 regarding ticket TKT-1042.",
]

response=client.recognize_pii_entities(documents,language="en")
results = [doc for doc in response if not doc.is_error]

for idx, doc in enumerate(results):
    print(f"--- Document {idx + 1} ---")
    print(f"Redacted text: {doc.redacted_text}")
    print("Detected entities:")
    for entity in doc.entities:
        print(f"  [{entity.category}] '{entity.text}'  (confidence: {entity.confidence_score:.2f})")
    print()