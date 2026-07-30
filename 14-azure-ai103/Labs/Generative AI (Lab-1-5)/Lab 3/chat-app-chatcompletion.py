# =============================================================================
# LAB 3 (variant 1 of 4): Simple console chatbot using the CLASSIC
# "Chat Completions" API (client.chat.completions.create).
#
# This is the OLDER of the two Azure OpenAI API styles covered in this lab.
# Compare this file line-by-line with chat-app-responseapi.py to see how the
# newer "Responses API" simplifies things (no manual message list, built-in
# conversation memory via previous_response_id).
#
# What you need before running this:
#   - An Azure AI Foundry project with a chat model deployed (e.g. gpt-4o).
#   - You must be logged in with `az login` (this script uses your Azure CLI
#     identity for auth - no API key is used).
#   - A .env file (in this folder or a parent folder) containing:
#       AZURE_OPENAI_ENDPOINT=<your Foundry/Azure OpenAI endpoint URL>
#       MODEL_DEPLOYMENT=<the name of your deployed chat model>
# =============================================================================

import os                      # Standard library: read environment variables, run OS commands (clear screen)
import sys
from pathlib import Path

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config                    # Centralized Azure credential/config module

# import namespaces
# import namespaces
from openai import OpenAI                                              # The official OpenAI Python SDK client
from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # Azure Entra ID (formerly Azure AD) authentication helpers


def main():
    # Clear the console so old output doesn't clutter the screen each run
    # 'cls' is the Windows clear command, 'clear' is the macOS/Linux one
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        # Get configuration settings
        azure_openai_endpoint = config.openai_endpoint  # e.g. https://<your-resource>.openai.azure.com/openai/v1
        model_deployment = config.openai_deployment      # e.g. "gpt-4o" - must match a deployment name in your Foundry project

        # Initialize the OpenAI client
        # Initialize the OpenAI client
        #
        # Instead of a hardcoded API key, we use DefaultAzureCredential, which
        # automatically picks up your `az login` session (or managed identity,
        # environment variables, etc. - it tries several methods in order).
        # get_bearer_token_provider wraps that credential so the OpenAI SDK can
        # call it every time it needs a fresh access token (tokens expire and
        # auto-refresh behind the scenes).
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://ai.azure.com/.default"  # the "scope" - which Azure resource we're requesting a token for
        )

        # Note: this uses the plain `OpenAI` client (not `AzureOpenAI`).
        # Because Azure OpenAI now exposes an OpenAI-compatible endpoint, you
        # can point the standard OpenAI SDK at it with `base_url=`, and pass
        # the bearer token in place of an API key.
        openai_client = OpenAI(
            base_url=azure_openai_endpoint,
            api_key=token_provider
        )


        # Loop until the user wants to quit
        # This is the main chat loop - keeps asking for input until "quit"
        while True:
            input_text = input('\nEnter a prompt (or type "quit" to exit): ')
            if input_text.lower() == "quit":
                break                                  # exit the while loop -> program ends
            if len(input_text) == 0:
                print("Please enter a prompt.")
                continue                                # skip back to the top of the loop, ask again

            # Get a response
            # Get a response
            #
            # This is the classic Chat Completions call. Notice that we send
            # the ENTIRE conversation as a `messages` list every single time -
            # there's no server-side memory here. In this simple version we
            # only ever send a "system" instruction plus the single latest
            # user message, so the model has NO memory of earlier turns in
            # the loop (each question is answered in isolation).
            completion = openai_client.chat.completions.create(
                model=model_deployment,               # which deployed model to use
                messages=[
                    {
                        "role": "system",              # sets the assistant's behavior/persona
                        "content": "You are a helpful AI assistant that answers questions and provides information."
                    },
                    {
                        "role": "user",                # the actual question from the person typing
                        "content": input_text
                    }
                ]
            )
            # The reply text lives at completion.choices[0].message.content
            # ("choices" is a list because you can ask the model for multiple
            # alternative completions at once - here we only ever get one).
            print(completion.choices[0].message.content)

    except Exception as ex:
        # Catch-all: prints any error (bad endpoint, auth failure, etc.) instead of crashing with a traceback
        print(ex)

if __name__ == '__main__':
    # Only run main() when this file is executed directly (not when imported)
    main()
