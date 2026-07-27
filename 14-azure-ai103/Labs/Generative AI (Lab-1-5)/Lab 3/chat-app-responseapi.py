# =============================================================================
# LAB 3 (variant 2 of 4): Simple console chatbot using the NEWER
# "Responses" API (client.responses.create) instead of Chat Completions.
#
# Key difference from chat-app-chatcompletion.py: you don't build/send a
# `messages` array yourself. Instead you pass plain `input` text, and Azure
# keeps track of the conversation FOR you on the server side - you just tell
# it which previous response to continue from via `previous_response_id`.
# That's what gives this version real multi-turn memory (the chat-completion
# version did not have memory).
#
# Prerequisites (same as chat-app-chatcompletion.py):
#   - Azure AI Foundry project with a chat model deployed.
#   - Logged in via `az login`.
#   - .env with AZURE_OPENAI_ENDPOINT and MODEL_DEPLOYMENT.
# =============================================================================

import os                      # Standard library: env vars, OS-level clear-screen command
from dotenv import load_dotenv  # Loads variables from a .env file into the environment

# import namespaces
# import namespaces
from openai import OpenAI                                              # Official OpenAI Python SDK
from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # Entra ID auth helpers


def main():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        # Get configuration settings
        load_dotenv()
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        model_deployment = os.getenv("MODEL_DEPLOYMENT")

        # Initialize the OpenAI client
        # Initialize the OpenAI client
        #
        # Same auth pattern as the Chat Completions version: get a token
        # provider tied to your `az login` identity instead of an API key.
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://ai.azure.com/.default"
        )

        openai_client = OpenAI(
            base_url=azure_openai_endpoint,
            api_key=token_provider
        )

        # Track responses
        # This starts as None (no previous conversation yet). After the
        # first reply we save its ID here so the NEXT call can say
        # "continue from that response" and the model will remember
        # everything said so far in this session.
        last_response_id = None

        # Loop until the user wants to quit
        while True:
            input_text = input('\nEnter a prompt (or type "quit" to exit): ')
            if input_text.lower() == "quit":
                break
            if len(input_text) == 0:
                print("Please enter a prompt.")
                continue

            # Get a response
            # Get a response
            #
            # `instructions` plays the same role as the "system" message did
            # in the Chat Completions version - it sets the assistant's
            # persona/behavior for this turn.
            # `input` is just the plain user text - no list of role/content
            # dicts needed.
            # `previous_response_id=last_response_id` is the trick that gives
            # this chatbot memory: on the very first turn it's None (fresh
            # conversation), but from the second turn onward it points back
            # at the prior response so Azure can recall the full history.
            response = openai_client.responses.create(
                        model=model_deployment,
                        instructions="You are a helpful AI assistant that answers questions and provides information.",
                        input=input_text,
                        previous_response_id=last_response_id,
            )
            # `output_text` is a convenience property that concatenates all
            # the text parts of the reply into one string.
            print(response.output_text)
            # Remember this response's ID so the next loop iteration can
            # chain off of it and keep the conversation going.
            last_response_id = response.id
    except Exception as ex:
        # Catch-all: print any error instead of crashing
        print(ex)

if __name__ == '__main__':
    main()
