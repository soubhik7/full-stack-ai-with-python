# =============================================================================
# LAB 3 (variant 3 of 4): Same chatbot as chat-app-responseapi.py, but with
# STREAMING enabled - the reply prints word-by-word/token-by-token as the
# model generates it, instead of waiting for the whole answer to finish.
#
# This is the pattern you'd use for a "typing" effect in a real chat UI so
# users see progress immediately rather than staring at a blank screen.
#
# Prerequisites: same as the other Lab 3 files - Azure AI Foundry project +
# deployed chat model + `az login` + .env with AZURE_OPENAI_ENDPOINT and
# MODEL_DEPLOYMENT.
# =============================================================================

import os                      # Standard library: env vars, clear-screen command
from dotenv import load_dotenv  # Loads .env file variables into the environment

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
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://ai.azure.com/.default"
        )

        openai_client = OpenAI(
            base_url=azure_openai_endpoint,
            api_key=token_provider
        )

        # Track responses
        # Same idea as the non-streaming version: remembers the last
        # response's ID so the conversation has memory across turns.
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
            # The only new thing here vs. the non-streaming version is
            # `stream=True`. Instead of returning one finished `response`
            # object, the SDK now returns an iterator ("stream") of small
            # EVENT objects that arrive as the model generates text.
            stream = openai_client.responses.create(
                        model=model_deployment,
                        instructions="You are a helpful AI assistant that answers questions and provides information.",
                        input=input_text,
                        previous_response_id=last_response_id,
                        stream=True
            )
            # Loop over every event as it arrives from the server.
            for event in stream:
                # "response.output_text.delta" events each carry one small
                # chunk ("delta") of newly generated text - print it
                # immediately, without a newline, so the words appear to
                # stream across the screen in real time.
                if event.type == "response.output_text.delta":
                    print(event.delta, end="")
                # "response.completed" fires once, at the very end, and
                # carries the FULL final response object - that's where we
                # grab the response ID to remember for the next turn (the
                # delta events don't have it).
                elif event.type == "response.completed":
                    last_response_id = event.response.id
            # Print a newline after the streamed reply finishes, so the next
            # prompt starts on a fresh line.
            print()
    except Exception as ex:
        # Catch-all: print any error instead of crashing
        print(ex)

if __name__ == '__main__':
    main()
