# =============================================================================
# LAB 1 (NEW — added to fill an AI-103 syllabus gap, not an original course
# file): "Plan and prepare AI development" + your first connection test.
#
# The AI-103 course map's first learning-path module is "Plan/prepare AI
# development" followed by "model catalog: select, deploy, evaluate" — before
# writing any chat app (Lab 2/3) or agent (Lab 6+), you need to prove your
# Foundry project, deployed model, and Azure credentials are all wired up
# correctly. This lab is that smoke test.
#
# Note on the "model catalog" part of this module: there is no SDK call in
# this repo (or, as far as this lab's author could verify, in the public
# `azure-ai-projects` SDK) that lists the Foundry model catalog itself —
# selecting and deploying a model is a Foundry PORTAL action (Model Catalog →
# Deploy), not something you script. What you CAN verify in code is that the
# deployment you picked in the portal actually works — which is what this
# script does. This mirrors the real pattern used in this repo's
# `11_azure_ai_foundry/00_setup/verify_connection.py`.
#
# Prerequisites: an Azure AI Foundry project with a chat model deployed
# (pick one in the portal's Model Catalog first) + `az login` + .env with
# PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME.
# =============================================================================

import os                      # Env vars, clear-screen command
from dotenv import load_dotenv  # Loads .env file variables into the environment

from azure.ai.projects import AIProjectClient       # Talks to your Foundry project
from azure.identity import DefaultAzureCredential   # Authenticates using your `az login` session


def main():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    load_dotenv()
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

    if not project_endpoint or not model_deployment:
        print("Error: set PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME in your .env file first.")
        print("PROJECT_ENDPOINT is your Foundry project's endpoint URL.")
        print("MODEL_DEPLOYMENT_NAME must match a model you've already deployed")
        print("in the Foundry portal's Model Catalog (Model Catalog -> Deploy).")
        return

    print(f"Step 1/3: Connecting to Foundry project at {project_endpoint} ...")
    # DefaultAzureCredential automatically uses your `az login` session - no
    # API key stored anywhere in this script.
    credential = DefaultAzureCredential()
    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=credential,
    )
    print("  Connected.")

    print(f"\nStep 2/3: Confirming the deployment '{model_deployment}' actually responds ...")
    # get_openai_client() wraps the project connection in an OpenAI-compatible
    # client - the same one every other lab in this folder uses for
    # .responses.create()/.chat.completions.create().
    openai_client = project_client.get_openai_client()

    # The simplest possible round-trip: one prompt, one reply. If this
    # succeeds, your endpoint, deployment name, and Azure identity are all
    # correctly configured for every other lab in this folder.
    completion = openai_client.chat.completions.create(
        model=model_deployment,
        messages=[
            {"role": "user", "content": "Reply with exactly one word: 'ready'."}
        ],
    )
    reply = completion.choices[0].message.content
    print(f"  Model replied: {reply}")

    print("\nStep 3/3: Summary")
    print("=" * 60)
    print(f"  Project endpoint : {project_endpoint}")
    print(f"  Model deployment : {model_deployment}")
    print("  Auth method      : DefaultAzureCredential (az login / Entra ID)")
    print("  Status           : ✅ ready — you can move on to Lab 2/3")
    print("=" * 60)


if __name__ == "__main__":
    main()
