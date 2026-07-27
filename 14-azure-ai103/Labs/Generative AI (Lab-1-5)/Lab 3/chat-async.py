# =============================================================================
# LAB 3 (variant 4 of 4): Same chatbot as chat-app-responseapi.py, but written
# with Python's ASYNC/AWAIT syntax instead of plain synchronous calls.
#
# Why this matters: a synchronous script blocks (does nothing else) while it
# waits for the network call to Azure to come back. An async script can, in
# principle, do other work while waiting (e.g. handle multiple users'
# requests at once in a web server). This lab file is a minimal example of
# what an async version of the same chatbot looks like - notice every line
# that talks to Azure now has `await` in front of it.
#
# Prerequisites: same as the other Lab 3 files - Azure AI Foundry project +
# deployed chat model + `az login` + .env with AZURE_OPENAI_ENDPOINT and
# MODEL_DEPLOYMENT.
# =============================================================================

import os                      # Standard library: env vars, clear-screen command
from dotenv import load_dotenv  # Loads .env file variables into the environment

# import namespaces for async
# import namespaces for async
import asyncio                                     # Python's built-in async/await event-loop library
from openai import AsyncOpenAI                     # The ASYNC version of the OpenAI SDK client (note the "Async" prefix)
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider  # ".aio" = the async versions of these Azure auth helpers


async def main():
    # `async def` means this function can contain `await` calls and must
    # itself be run through an async event loop (see asyncio.run(main())
    # at the very bottom of this file).

    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        # Get configuration settings
        load_dotenv()
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        model_deployment = os.getenv("MODEL_DEPLOYMENT")

        # Initialize an async OpenAI client
        # Initialize an async OpenAI client
        #
        # We keep a direct reference to `credential` (rather than only the
        # token_provider built from it) because async credentials hold open
        # network resources that must be explicitly closed later - see the
        # `finally` block at the bottom of this function.
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
        credential, "https://ai.azure.com/.default"
        )

        async_client = AsyncOpenAI(
            base_url=azure_openai_endpoint,
            api_key=token_provider
        )


        # Track responses
        # Same conversation-memory trick as the other Responses API examples:
        # None on the first turn, then the previous response's ID afterward.
        last_response_id = None

        # Loop until the user wants to quit
        while True:
            input_text = input('\nEnter a prompt (or type "quit" to exit): ')
            if input_text.lower() == "quit":
                break
            if len(input_text) == 0:
                print("Please enter a prompt.")
                continue

            # Await an asynchronous response
            # Await an asynchronous response
            #
            # Same call shape as the synchronous Responses API example, but
            # since `async_client` is the async client, this call returns a
            # coroutine that we must `await` - Python pauses this function
            # here (without blocking the whole program) until the network
            # response comes back from Azure.
            response = await async_client.responses.create(
                        model=model_deployment,
                        instructions="You are a helpful AI assistant that answers questions and provides information.",
                        input=input_text,
                        previous_response_id=last_response_id
            )
            assistant_text = response.output_text
            print("Assistant:", assistant_text)
            # Remember this response's ID for the next turn, exactly like the
            # synchronous Responses API version does.
            last_response_id = response.id


    except Exception as ex:
        # Catch-all: print any error instead of crashing
        print(ex)

    finally:
        # Close the async client session
        # Close the async client session
        #
        # Async credentials open background network connections (e.g. to
        # refresh tokens) that need to be explicitly released. This runs
        # whether the try block succeeded or raised an exception, so we
        # never leak the connection.
        await credential.close()


if __name__ == '__main__':
    # asyncio.run() creates an event loop, runs main() to completion inside
    # it, and then closes the loop - this is the standard entry point for a
    # top-level async script.
    asyncio.run(main())
