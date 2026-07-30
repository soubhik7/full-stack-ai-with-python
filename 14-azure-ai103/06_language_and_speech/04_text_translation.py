from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
import sys
from pathlib import Path

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

deployment_name = config.model_deployment

project = AIProjectClient(
    endpoint=config.project_endpoint,
    credential=DefaultAzureCredential(),
)

openai = project.get_openai_client()

source_text = """
Hi team, our VPN keeps dropping every 10 minutes since the last update.
This is affecting our whole sales team and we need this fixed today.
"""

def translate(text: str, target_language: str) -> str:
    system_prompt = f"""
    You are a professional translator. Translate the user's text into
    {target_language}. Preserve the original line breaks and formatting.
    Preserve the tone and register of the original (e.g. formal, urgent,
    casual) rather than producing a flat, literal, word-for-word translation.

    Respond in exactly this plain-text format, with no extra commentary:

    SOURCE LANGUAGE: <detected source language>
    TRANSLATION: <the translated text>
    """

    response = openai.responses.create(
        model=deployment_name,
        input=[
            {"type": "message", "role": "system", "content": system_prompt},
            {"type": "message", "role": "user", "content": text}
        ]
    )
    return response.output_text

for target in ["French", "Japanese"]:
    print(f"--- Translating to {target} ---")
    print(translate(source_text, target))
    print()