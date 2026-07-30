# =============================================================================
# LAB 6: Chat client for an agent that was ALREADY CREATED IN THE FOUNDRY
# PORTAL (this script only connects to it - it does not define any tools
# itself). The point of this lab is handling everything an agent can send
# back besides plain text: code-interpreter-generated charts, downloadable
# files referenced by citation, and images - and saving them locally so you
# can actually open them.
#
# Needs a pre-built portal agent: create an agent in the Foundry portal
# first (give it the code interpreter tool so it can produce charts/files),
# then set AGENT_NAME in your .env to match its name (defaults to
# "it-support-agent" if not set).
#
# Prerequisites: Azure AI Foundry project + a portal-created agent +
# `az login` + .env with PROJECT_ENDPOINT (and optionally AGENT_NAME).
# =============================================================================

import base64                      # Decodes base64 text back into raw image bytes
import sys
from pathlib import Path           # Cross-platform, object-oriented file path handling

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config                     # Centralized Azure credential/config module

from azure.ai.projects import AIProjectClient       # Talks to your Foundry project (agents, conversations, etc.)
from azure.identity import DefaultAzureCredential   # Authenticates using your `az login` session


OUTPUT_DIR = Path("agent_outputs")  # Every file the agent generates gets saved into this local folder


def get_output_path(filename):
    """Create a unique path for generated files."""
    # Make sure the output folder exists (does nothing if it's already there)
    OUTPUT_DIR.mkdir(exist_ok=True)
    # Strip any directory info from the agent-provided filename - we only
    # want the bare file name (e.g. "chart.png"), not a path, to avoid
    # writing outside OUTPUT_DIR.
    file_name = Path(filename).name
    stem = Path(file_name).stem or "output"   # filename without its extension, e.g. "chart"
    suffix = Path(file_name).suffix           # just the extension, e.g. ".png"
    output_path = OUTPUT_DIR / file_name

    # If a file with that name already exists (e.g. you asked for two charts
    # in the same session), keep appending a counter until the name is free,
    # so nothing gets silently overwritten: chart.png -> chart_1.png -> chart_2.png ...
    counter = 1
    while output_path.exists():
        output_path = OUTPUT_DIR / f"{stem}_{counter}{suffix}"
        counter += 1

    return output_path


def save_bytes(file_bytes, filename):
    """Save binary content to a local file."""
    output_path = get_output_path(filename)
    # "wb" = write binary mode, required because file_bytes is raw bytes,
    # not text (charts, PDFs, etc. are all binary data).
    with open(output_path, "wb") as file_handle:
        file_handle.write(file_bytes)
    return output_path


def save_image(image_data, filename):
    """Save base64 image data to a file."""
    # The agent sends generated images as base64-encoded TEXT (a common way
    # to embed binary data inside JSON). base64.b64decode converts that text
    # back into the actual raw image bytes before we write it to disk.
    return save_bytes(base64.b64decode(image_data), filename)


def download_container_file(openai_client, annotation, downloaded_files):
    """Download a cited container file once and return its local path."""
    # When the agent's code interpreter creates a file (e.g. a CSV or a
    # chart it references in text), the response doesn't include the file
    # itself - it includes a *citation* pointing at a "container" (a
    # sandboxed workspace) where the file lives. This function fetches the
    # actual bytes for one such citation.
    #
    # A single file can be cited multiple times in the same message, so we
    # cache by (container_id, file_id) to avoid downloading it twice.
    cache_key = (annotation.container_id, annotation.file_id)
    if cache_key in downloaded_files:
        return downloaded_files[cache_key]

    # Ask the Foundry/OpenAI-compatible API for the raw content of this
    # specific file inside this specific sandbox container.
    file_content = openai_client.containers.files.content.retrieve(
        file_id=annotation.file_id,
        container_id=annotation.container_id,
    )
    output_path = save_bytes(
        file_content.read(),
        annotation.filename or f"{annotation.file_id}.bin",  # fall back to a generic name if none was given
    )
    downloaded_files[cache_key] = output_path
    return output_path


def format_output_text(content_item, openai_client, downloaded_files):
    """Replace sandbox file citations with local file paths."""
    # The raw text may contain inline citation markers pointing at sandbox
    # files (e.g. "here's your report【4:0†report.csv】"). This function
    # downloads each cited file and swaps the citation marker for a readable
    # "filename (saved to path)" note, so the printed text is actually useful
    # to a human instead of showing internal citation syntax.
    text = content_item.text or ""
    replacements = []
    referenced_files = set()

    for annotation in content_item.annotations or []:
        # Skip any annotation that isn't a reference to a sandbox container
        # file (there can be other annotation types we don't care about here).
        if getattr(annotation, "type", "") != "container_file_citation":
            continue

        output_path = download_container_file(openai_client, annotation, downloaded_files)
        replacement_text = f"{annotation.filename} (saved to {output_path})"
        referenced_files.add(output_path)

        # Prefer precise start/end character positions when the API gives
        # them - that lets us splice the replacement in at the exact spot.
        start_index = getattr(annotation, "start_index", None)
        end_index = getattr(annotation, "end_index", None)
        if start_index is not None and end_index is not None:
            replacements.append((start_index, end_index, replacement_text))
            continue

        # Fallback: if there's no index info, just do a plain text search-
        # and-replace of the citation marker string.
        annotated_text = getattr(annotation, "text", "")
        if annotated_text:
            text = text.replace(annotated_text, replacement_text)

    # Apply the indexed replacements in REVERSE order (highest start_index
    # first). This matters: once you splice text at one position, every
    # index after it shifts - going in reverse means earlier positions are
    # still valid when we get to them.
    for start_index, end_index, replacement_text in sorted(replacements, reverse=True):
        text = f"{text[:start_index]}{replacement_text}{text[end_index:]}"

    return text, referenced_files


def main():
    # Initialize the project client
    project_endpoint = config.project_endpoint
    agent_name = config.agent_name("it-support-agent")  # falls back to this default if not set in .env

    if not project_endpoint:
        print("Error: PROJECT_ENDPOINT environment variable not set")
        print("Please set it in your .env file or environment")
        return

    print("Connecting to Microsoft Foundry project...")
    # DefaultAzureCredential automatically uses your `az login` session (or
    # managed identity / env vars, whichever is available) - no API key needed.
    credential = DefaultAzureCredential()
    project_client = AIProjectClient(
        credential=credential,
        endpoint=project_endpoint
    )

    # Get the OpenAI client for Responses API
    # AIProjectClient wraps a plain OpenAI-compatible client so you can call
    # .conversations / .responses just like in the Lab 3 examples, but
    # already authenticated against your Foundry project.
    openai_client = project_client.get_openai_client()

    # Get the agent created in the portal
    # This is why the lab is called "portal-created agent" - we don't
    # define instructions or tools here, we just look up an agent that
    # someone already configured (with a code interpreter tool, etc.) in
    # the Foundry web UI, by its name.
    print(f"Loading agent: {agent_name}")
    agent = project_client.agents.get(agent_name=agent_name)
    print(f"Connected to agent: {agent.name} (id: {agent.id})")

    # Create a conversation
    # A "conversation" is the server-side container that holds message
    # history - it's created once here, then every user turn below adds a
    # new item to this same conversation so the agent has memory.
    conversation = openai_client.conversations.create(items=[])
    print(f"Conversation created (id: {conversation.id})")

    # Chat loop
    print("\n" + "="*60)
    print("IT Support Agent Ready!")
    print("Ask questions, request data analysis, or get help.")
    print("Type 'exit' to quit.")
    print("="*60 + "\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Add user message to conversation
        # This appends your typed message as a new item in the shared
        # conversation - the agent will see it (and everything before it)
        # when we ask for a response next.
        openai_client.conversations.items.create(
            conversation_id=conversation.id,
            items=[{"type": "message", "role": "user", "content": user_input}]
        )

        # Get response from agent
        print("\n[Agent is thinking...]")
        # `extra_body={"agent_reference": ...}` is what tells the Responses
        # API "answer as THIS specific portal agent" (with its own
        # instructions/tools) rather than a bare model. `input=""` because
        # the actual user message was already added to the conversation
        # above - we're just asking the agent to respond to what's there.
        response = openai_client.responses.create(
            conversation=conversation.id,
            extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            input=""
        )

        # Display response and save any generated files locally
        # The agent's reply can be made up of several different "output
        # item" types (a plain message, an image, etc.) - this loop walks
        # through all of them and handles each kind appropriately.
        handled_output = False
        downloaded_files = {}
        referenced_files = set()
        image_count = 0

        if hasattr(response, "output") and response.output:
            for item in response.output:
                item_type = getattr(item, "type", "")

                if item_type == "message" and getattr(item, "content", None):
                    # A normal chat message - may itself contain several
                    # content parts (text, citations, etc.)
                    for content_item in item.content:
                        if getattr(content_item, "type", "") != "output_text":
                            continue

                        # Resolve any file citations into "saved to <path>"
                        # notes before printing (see format_output_text above).
                        formatted_text, message_files = format_output_text(
                            content_item,
                            openai_client,
                            downloaded_files,
                        )
                        referenced_files.update(message_files)

                        if formatted_text:
                            print(f"\nAgent: {formatted_text}\n")
                            handled_output = True

                elif hasattr(item, "text") and item.text:
                    # A fallback for any output item that just has a plain
                    # `.text` attribute but isn't a full "message" type.
                    print(f"\nAgent: {item.text}\n")
                    handled_output = True

                elif item_type == "image":
                    # The agent generated an image directly (e.g. a chart
                    # from code interpreter) rather than citing a file -
                    # decode and save it under a generic "chart_N.png" name.
                    image_count += 1
                    filename = f"chart_{image_count}.png"

                    if hasattr(item, "image") and hasattr(item.image, "data"):
                        file_path = save_image(item.image.data, filename)
                        print(f"\n[Agent generated a chart - saved to: {file_path}]")
                    else:
                        print("\n[Agent generated an image]")
                    handled_output = True

            # Any file that was downloaded via a citation but never actually
            # appeared inline in the printed text (edge case) still gets
            # reported here so nothing generated by the agent goes unnoticed.
            for file_path in downloaded_files.values():
                if file_path not in referenced_files:
                    print(f"\n[Agent generated a file - saved to: {file_path}]")
                    handled_output = True

        # Last-resort fallback: if none of the structured output handling
        # above produced anything to show, just print the SDK's convenience
        # `output_text` property (concatenated plain text of the reply).
        if not handled_output and hasattr(response, "output_text") and response.output_text:
            print(f"\nAgent: {response.output_text}\n")

if __name__ == "__main__":
    main()
