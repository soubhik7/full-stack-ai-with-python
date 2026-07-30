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
    "Hi, this is Sarah Chen from Acme Logistics regarding ticket TKT-1042.",
    "Bonjour, je vous écris au sujet du ticket TKT-1042 concernant notre VPN.",
    "こんにちは、TKT-1042のチケットについてVPNの問題をご連絡しています。",
    "OK"
]

response=client.detect_language(documents)
results = [doc for doc in response if not doc.is_error]

for idx, doc in enumerate(results):
    primary = doc.primary_language
    print(f"--- Document {idx + 1}: \"{documents[idx][:40]}...\" ---")
    print(f"  Detected language: {primary.name} ({primary.iso6391_name})")
    print(f"  Confidence score: {primary.confidence_score:.2f}")
    print()

