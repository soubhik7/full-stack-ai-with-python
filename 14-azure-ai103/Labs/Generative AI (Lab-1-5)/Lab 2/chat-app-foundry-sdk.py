# =============================================================================
# LAB 2 (NEW — added to fill an AI-103 syllabus gap, not an original course
# file): "Chat app with Foundry SDK + Responses API" — the AI-103 course map
# names this as its own module, distinct from Lab 3's four chat variants.
#
# The difference from Lab 3: every Lab 3 script builds its client with
# `OpenAI(base_url=azure_openai_endpoint, api_key=token_provider)` - pointing
# a generic OpenAI SDK client directly at an Azure endpoint. THIS lab instead
# goes through the actual **Foundry SDK** (`azure-ai-projects`'
# `AIProjectClient`), the same client every agent lab (6 onward) in this
# folder uses via `project_client.get_openai_client()`. Architecturally this
# matters: the Foundry-SDK route is what lets you later swap in agent
# references, project-level connections (Azure AI Search, MCP tools), and
# managed authentication without changing your chat code - Lab 3's approach
# cannot do any of that.
#
# Prerequisites: Azure AI Foundry project + deployed chat model + `az login`
# + .env with PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME.
# =============================================================================

import os                      # Env vars, clear-screen command
from dotenv import load_dotenv  # Loads .env file variables into the environment

from azure.ai.projects import AIProjectClient       # The Foundry SDK's project client
from azure.identity import DefaultAzureCredential   # Authenticates using your `az login` session


def main():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        load_dotenv()
        project_endpoint = os.getenv("PROJECT_ENDPOINT")
        model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

        # Connect through the Foundry SDK
        # Connect through the Foundry SDK
        #
        # This is the one line that makes this a "Foundry SDK" chat app
        # rather than a plain OpenAI-client app: AIProjectClient represents
        # your whole Foundry PROJECT (which can hold multiple model
        # deployments, connections, and agents), not just one endpoint.
        project_client = AIProjectClient(
            endpoint=project_endpoint,
            credential=DefaultAzureCredential(),
        )

        # get_openai_client() hands you back an OpenAI-compatible client that's
        # already wired up to this specific project - you call .responses.create()
        # on it exactly like Lab 3 does, but authentication and endpoint
        # resolution are handled by the project client instead of manual
        # base_url/token_provider plumbing.
        openai_client = project_client.get_openai_client()

        # Track responses
        # Same Responses API memory trick as Lab 3: None on the first turn,
        # then the previous response's ID on every turn after that.
        last_response_id = None

        print("Foundry SDK chat app ready. Type 'quit' to exit.\n")

        while True:
            input_text = input('\nEnter a prompt (or type "quit" to exit): ')
            if input_text.lower() == "quit":
                break
            if len(input_text) == 0:
                print("Please enter a prompt.")
                continue

            # Get a response
            # Identical call shape to Lab 3's chat-app-responseapi.py - the
            # only thing that changed is HOW openai_client was constructed
            # above (via the Foundry SDK, not a raw base_url).
            response = openai_client.responses.create(
                model=model_deployment,
                instructions="You are a helpful AI assistant that answers questions and provides information.",
                input=input_text,
                previous_response_id=last_response_id,
            )
            print(response.output_text)
            last_response_id = response.id

    except Exception as ex:
        # Catch-all: print any error instead of crashing
        print(ex)


if __name__ == "__main__":
    main()
