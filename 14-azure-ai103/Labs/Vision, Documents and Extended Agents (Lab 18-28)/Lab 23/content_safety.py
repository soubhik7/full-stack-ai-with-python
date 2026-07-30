# =============================================================================
# LAB 23 (NEW — added to fill an AI-103 syllabus gap, not an original course
# file): AZURE AI CONTENT SAFETY — the dedicated moderation SERVICE, not the
# "read content_filters off an agent's response" pattern. Both approaches are
# real and used side-by-side in this chapter's `05_image_generation_and_safety/` and
# `08_document_intelligence_capstone/` folders; this lab covers the dedicated-client half
# (Responsible AI is a Domain-1 objective AND a theme across the whole
# AI-103 exam).
#
# Prerequisites: an Azure AI Content Safety resource + .env with
# CONTENT_SAFETY_ENDPOINT and CONTENT_SAFETY_KEY (same variable names this
# chapter's top-level README already documents).
# Requires: pip install azure-ai-contentsafety (not in the root requirements.txt).
# =============================================================================

import os                      # Env vars, clear-screen command
import sys
from pathlib import Path

from azure.core.credentials import AzureKeyCredential                          # Content Safety authenticates with a subscription key, not az login
from azure.ai.contentsafety import ContentSafetyClient                          # The dedicated Content Safety SDK client
from azure.ai.contentsafety.models import AnalyzeTextOptions, AnalyzeImageOptions, ImageData  # Request shapes for text vs. image moderation

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config


def analyze_text(client, text):
    """Moderate a piece of text and print a severity score per harm category."""
    request = AnalyzeTextOptions(text=text)
    result = client.analyze_text(request)

    # Each harm category (Hate, SelfHarm, Sexual, Violence) gets its OWN
    # severity score, 0 (safe) to 7 (most severe) - unlike a single
    # pass/fail flag, this lets an app decide different thresholds per
    # category (e.g. block Violence at a lower severity than Sexual).
    print(f'\nText: "{text}"')
    should_block = False
    for category_result in result.categories_analysis:
        print(f"  {category_result.category}: severity {category_result.severity}")
        if category_result.severity >= 4:   # a common "block" threshold - tune per your app's risk tolerance
            should_block = True
    print(f"  -> {'BLOCK' if should_block else 'allow'}")
    return should_block


def analyze_image(client, image_path):
    """Moderate an image file the same way, using raw image bytes."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    request = AnalyzeImageOptions(image=ImageData(content=image_bytes))
    result = client.analyze_image(request)

    print(f"\nImage: {image_path}")
    should_block = False
    for category_result in result.categories_analysis:
        print(f"  {category_result.category}: severity {category_result.severity}")
        if category_result.severity >= 4:
            should_block = True
    print(f"  -> {'BLOCK' if should_block else 'allow'}")
    return should_block


def main():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        endpoint = config.content_safety_endpoint
        key = config.content_safety_key

        client = ContentSafetyClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )

        # A deliberately mixed batch: one clearly benign line, one mildly
        # aggressive line - so you can see the severity scores actually
        # differ rather than everything reading 0.
        sample_texts = [
            "Thank you so much for your help today, I really appreciate it!",
            "If you don't fix this immediately I will destroy your whole team.",
        ]
        for text in sample_texts:
            analyze_text(client, text)

        # Optional: analyze a local image if one is provided in this folder
        image_path = os.path.join(os.path.dirname(__file__), "sample_image.png")
        if os.path.exists(image_path):
            analyze_image(client, image_path)
        else:
            print(f"\n(Skipping image moderation - no sample_image.png found at {image_path})")

    except Exception as ex:
        # Catch-all: print any error instead of crashing
        print(ex)


if __name__ == "__main__":
    main()
