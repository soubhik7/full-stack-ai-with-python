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
    "Hi, this is Sarah Chen from Acme Logistics writing in again about "
    "ticket TKT-1042. Our Gold-tier SLA promises a 4 hour response time, "
    "and we are now at hour 6 with no update. The VPN client keeps "
    "dropping every 10 minutes on our Windows fleet since the rollout of "
    "CloudXeus Connect v3.2 last Tuesday. If this isn't resolved by end "
    "of day Friday we will be requesting the $500 SLA breach credit "
    "outlined in our contract.",
]

response=client.recognize_entities(documents,language="en")
results = [doc for doc in response if not doc.is_error]

for idx, doc in enumerate(results):
    print(f"--- Document {idx + 1}: Prebuilt NER Results ---")
    for entity in doc.entities:
        subcat = f" / {entity.subcategory}" if entity.subcategory else ""
        print(f"  [{entity.category}{subcat}] '{entity.text}'  (confidence: {entity.confidence_score:.2f})")
    print()