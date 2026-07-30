import requests
import sys
from pathlib import Path

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

API_VERSION = "2025-10-01-preview"


def translate_text(text, targets, source_language):
    headers = {
        "Ocp-Apim-Subscription-Key": config.translator_key,
        "Content-Type": "application/json"
    }
    url = f"{config.translator_endpoint}translator/text/translate?api-version={API_VERSION}"
    body = {
        "inputs": [
            {
                "Text": text,
                "language": source_language,
                "targets": targets
            }
        ]
    }

    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    return response.json()


def main():
    text = (
        "Our VPN keeps dropping every 10 minutes since the last update. "
        "This is affecting our whole sales team."
    )
    targets = [
        {"language": "fr"},
        {"language": "ja"},
        {"language": "es"},
    ]
    source_language = "en"

    try:
        result = translate_text(text, targets, source_language)

        for t in result["value"][0]["translations"]:
            print(f"[{t['language']}] {t['text']}")

    except Exception as e:
        print(f"Translation failed: {e}")


if __name__ == "__main__":
    main()