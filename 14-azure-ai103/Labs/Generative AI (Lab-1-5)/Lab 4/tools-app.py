# =============================================================================
# LAB 4: File-search tool exercise.
#
# ⚠️ THIS FILE IS AN UNFINISHED SKELETON, NOT A WORKING SCRIPT. It is the
# lab's STARTING POINT - several sections below are deliberately left blank
# (only a comment marks where code belongs) so that you, the student, fill
# them in. As shipped, running this script will connect to nothing and the
# "get a response" section does nothing (the try body under the loop is
# empty, so every prompt is silently ignored).
#
# GOAL OF THE LAB: teach Azure OpenAI's built-in "file search" tool - you
# upload documents, Azure indexes them into a vector store, and then you can
# ask questions that get answered by searching those documents (similar in
# spirit to chapter 07_rag/, but using Azure's managed retrieval instead of a
# hand-built RAG pipeline).
#
# WHAT YOU'D NEED TO ADD, roughly in this order (see the "# TODO" markers
# inline below for exactly where):
#   1. Create the OpenAI client using the same DefaultAzureCredential +
#      get_bearer_token_provider pattern used in Lab 3's chat-app files.
#   2. Create a vector store: openai_client.vector_stores.create(name=...)
#   3. Upload one or more local files into it, e.g.:
#        with open("some_doc.pdf", "rb") as f:
#            openai_client.vector_stores.files.upload_and_poll(
#                vector_store_id=vector_store.id, file=f)
#   4. Inside the loop, call:
#        response = openai_client.responses.create(
#            model=model_deployment,
#            input=input_text,
#            previous_response_id=last_response_id,
#            tools=[{"type": "file_search", "vector_store_ids": [vector_store.id]}],
#        )
#      then print response.output_text and save response.id into
#      last_response_id, exactly like the Lab 3 Responses API examples do.
#
# Prerequisites once completed: Azure AI Foundry project + deployed chat
# model + `az login` + .env with AZURE_OPENAI_ENDPOINT and MODEL_DEPLOYMENT,
# plus at least one local document to upload.
# =============================================================================

import os                      # Standard library: env vars, clear-screen command
import sys
from pathlib import Path
import glob                     # Pattern-matches files on disk (e.g. glob.glob("*.pdf")) - imported for use when uploading a folder of files, but not yet used below

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config                    # Centralized Azure credential/config module

# Import namespaces
# TODO: import the OpenAI SDK client and the Azure auth helpers here, e.g.:
#   from openai import OpenAI
#   from azure.identity import DefaultAzureCredential, get_bearer_token_provider



def main():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        # Get configuration settings
        azure_openai_endpoint = config.openai_endpoint  # your Azure OpenAI / Foundry endpoint
        model_deployment = config.openai_deployment      # your deployed chat model's name

        # Initialize the OpenAI client
        # TODO: build the token_provider + OpenAI(base_url=..., api_key=...)
        # client exactly like the Lab 3 chat-app files do. Nothing is created
        # here yet, so the variable `openai_client` does not exist below -
        # you must define it before the rest of this script will run.



        # Create vector store and upload files
        # TODO:
        #   1. vector_store = openai_client.vector_stores.create(name="lab4-docs")
        #   2. Upload your sample file(s) into it with
        #      openai_client.vector_stores.files.upload_and_poll(...)
        #   3. Keep hold of vector_store.id - you'll need it below when
        #      building the file_search tool.



        # Track conversation state
        # Same Responses API memory pattern as Lab 3: None on the first
        # question, then the previous response's ID on every question after
        # that, so the model remembers earlier turns.
        last_response_id = None

        # Loop until the user wants to quit
        while True:
            input_text = input('\nEnter a question (or type "quit" to exit): ')
            if input_text.lower() == "quit":
                break
            if len(input_text) == 0:
                print("Please enter a question.")
                continue

            # Get a response using tools
            # TODO: call openai_client.responses.create(...) with a
            # `tools=[{"type": "file_search", "vector_store_ids": [vector_store.id]}]`
            # argument so the model can search your uploaded documents before
            # answering. Then print response.output_text and update
            # last_response_id = response.id, same as Lab 3.
            #
            # As written, this section does nothing - every question you
            # type is silently thrown away and the loop just asks again.



    except Exception as ex:
        # Catch-all: print any error instead of crashing
        print(ex)

if __name__ == '__main__':
    main()
