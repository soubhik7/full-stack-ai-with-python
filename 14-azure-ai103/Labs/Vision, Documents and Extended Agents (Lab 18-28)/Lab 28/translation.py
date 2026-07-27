# =============================================================================
# LAB 28 (NEW — added to fill an AI-103 syllabus gap, not an original course
# file): TRANSLATION — the last topic named in the "Develop natural language
# solutions" learning path ("...translate text & speech") that had no lab
# anywhere in this folder. Adapted from this repo's own real, working
# `06. Section Code/04_text_translation.py` and `09_text_translation.py`,
# which demonstrate BOTH approaches side by side - exactly the trade-off
# AI-103 tests (Domain 4: "know when to reach for an LLM prompt vs. a
# dedicated service").
#
# Prerequisites: for the dedicated-service half, an Azure AI Translator
# resource + .env with AZURE_TRANSLATOR_ENDPOINT and AZURE_TRANSLATOR_KEY;
# for the LLM half, a Foundry project + `az login` + .env with
# PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME.
# =============================================================================

import os                      # Env vars, clear-screen command
import requests                # Plain HTTP calls - the Translator service is called via REST, not a dedicated Python SDK package
from dotenv import load_dotenv  # Loads .env file variables into the environment

from azure.ai.projects import AIProjectClient       # Talks to your Foundry project (LLM-prompted translation half)
from azure.identity import DefaultAzureCredential   # Authenticates using your `az login` session


# ----------------------------------------------------------------------------
# Approach 1: LLM-prompted translation — no Translator service at all
# ----------------------------------------------------------------------------
def translate_with_llm(openai_client, model_deployment, text, target_language):
    """Ask a chat model to translate directly. Fast to set up (one prompt,
    no extra Azure resource), and the model can follow nuanced instructions
    ("keep it formal", "translate idioms naturally") - but quality and
    consistency depend entirely on the model, with no dedicated
    translation-quality guarantees.
    """
    system_prompt = (
        f"You are a professional translator. Translate the user's text into "
        f"{target_language}. Reply with ONLY the translated text, no explanation."
    )
    response = openai_client.responses.create(
        model=model_deployment,
        input=[
            {"type": "message", "role": "system", "content": system_prompt},
            {"type": "message", "role": "user", "content": text},
        ],
    )
    return response.output_text


# ----------------------------------------------------------------------------
# Approach 2: Azure AI Translator — the dedicated service, called via REST
# ----------------------------------------------------------------------------
def translate_with_translator_service(endpoint, key, text, target_languages, source_language="en"):
    """Azure AI Translator has no dedicated Python SDK package in common use
    here - it's called as a plain REST API. Its one distinguishing feature
    vs. asking an LLM: a SINGLE call can translate into MANY target
    languages at once (the `targets` array below), each returned with a
    confidence-scored result, which is far cheaper than N separate LLM
    calls for N languages.
    """
    headers = {
        "Ocp-Apim-Subscription-Key": key,   # Translator's own auth header - distinct from Bearer tokens or AzureKeyCredential
        "Content-Type": "application/json",
    }
    url = f"{endpoint}translator/text/translate?api-version=2025-10-01-preview"
    targets = [{"language": lang} for lang in target_languages]
    body = {"inputs": [{"Text": text, "language": source_language, "targets": targets}]}

    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()   # raises an exception for a 4xx/5xx response, instead of silently returning bad data
    return response.json()


def main():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        load_dotenv()

        text = "Could you please confirm my reservation for tomorrow evening?"

        # --- Approach 1: LLM-prompted ---
        project_endpoint = os.getenv("PROJECT_ENDPOINT")
        model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")
        if project_endpoint and model_deployment:
            print("=" * 60)
            print("APPROACH 1: LLM-prompted translation")
            print("=" * 60)
            project_client = AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential())
            openai_client = project_client.get_openai_client()

            translated = translate_with_llm(openai_client, model_deployment, text, "French")
            print(f'Original (en): "{text}"')
            print(f"Translated (fr): {translated}")
        else:
            print("Skipping Approach 1 - set PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME to try it.")

        # --- Approach 2: dedicated Translator service ---
        translator_endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
        translator_key = os.getenv("AZURE_TRANSLATOR_KEY")
        if translator_endpoint and translator_key:
            print("\n" + "=" * 60)
            print("APPROACH 2: Azure AI Translator (dedicated service, REST)")
            print("=" * 60)
            result = translate_with_translator_service(
                translator_endpoint, translator_key, text,
                target_languages=["fr", "ja", "es"],
            )
            print(f'Original (en): "{text}"')
            for translation in result["value"][0]["translations"]:
                print(f"  [{translation['language']}] {translation['text']}")
        else:
            print("\nSkipping Approach 2 - set AZURE_TRANSLATOR_ENDPOINT and AZURE_TRANSLATOR_KEY to try it.")
            print("Notice how ONE call above (Approach 2) can translate to THREE languages at")
            print("once, where the LLM approach needs one call PER target language.")

    except Exception as ex:
        # Catch-all: print any error instead of crashing
        print(ex)


if __name__ == "__main__":
    main()
