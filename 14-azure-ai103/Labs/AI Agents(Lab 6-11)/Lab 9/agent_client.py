# =============================================================================
# LAB 9: The SIMPLEST possible "talk to an existing Foundry agent" script -
# no tools, no function calling, no streaming. If Lab 6/7/8's extra
# machinery feels overwhelming, read this file first.
#
# Needs a pre-built portal agent: create an agent in the Foundry portal
# (this script doesn't define one), then set PROJECT_ENDPOINT and
# AGENT_NAME in your .env to point at it.
#
# The pattern here - connect, get agent, create conversation, loop sending
# messages - is the backbone every other agent-chat lab in this folder
# builds on top of.
# =============================================================================

import os                          # Env vars
from dotenv import load_dotenv      # Loads .env file variables into the environment
from azure.identity import DefaultAzureCredential   # Authenticates using your `az login` session (no API key)
from azure.ai.projects import AIProjectClient       # Talks to your Foundry project

# Load environment variables
load_dotenv()
project_endpoint = os.getenv("PROJECT_ENDPOINT")
agent_name = os.getenv("AGENT_NAME")

# Validate configuration
# Fail fast with a clear message rather than a confusing error later if
# either required setting is missing from .env.
if not project_endpoint or not agent_name:
    raise ValueError("PROJECT_ENDPOINT and AGENT_NAME must be set in .env file")

print(f"Connecting to project: {project_endpoint}")
print(f"Using agent: {agent_name}\n")

# TODO: Connect to the project and create a conversation
# Add your code here to:
# 1. Create DefaultAzureCredential
# 2. Create AIProjectClient with endpoint
# 3. Get the OpenAI client
# 4. Get the agent by name
# 5. Create a new conversation
#
# NOTE: despite the "TODO" comment above (left over from how this lab was
# originally handed out as an exercise), the code that actually does all
# five of those steps is already written below - this file is a COMPLETE,
# working example, not a skeleton you need to fill in.

# Connect to the project and agent
# `exclude_environment_credential` / `exclude_managed_identity_credential`
# tell DefaultAzureCredential to SKIP those auth methods and go straight to
# your `az login` (Azure CLI) session - useful when you have leftover
# environment variables or a managed identity that would otherwise be tried
# first and might fail or pick the wrong identity.
credential = DefaultAzureCredential(
    exclude_environment_credential=True,
    exclude_managed_identity_credential=True
)
project_client = AIProjectClient(
    credential=credential,
    endpoint=project_endpoint
)

# Get the OpenAI client
# Wraps the Foundry project connection in an OpenAI-compatible client so we
# can call .conversations / .responses the same way as every other lab.
openai_client = project_client.get_openai_client()

# Get the agent
# Looks up the agent you created in the Foundry portal by name - this
# script never creates or modifies the agent itself.
agent = project_client.agents.get(agent_name=agent_name)
print(f"Connected to agent: {agent.name} (id: {agent.id})\n")

# Create a new conversation
# A conversation is the server-side container that remembers message
# history for this session - created once here, reused for every turn below.
conversation = openai_client.conversations.create(items=[])
print(f"Created conversation (id: {conversation.id})\n")


# Conversation history for context (client-side tracking)
# This list is a LOCAL copy of the conversation, kept only so
# display_conversation_history() can print a transcript - it is NOT what
# gives the agent memory (the server-side `conversation` object already
# does that). Think of this as a convenience for the user, not the model.
conversation_history = []


def send_message_to_agent(user_message):
    """
    Send a message to the agent and handle the response using the conversations API.
    """
    try:
        print("\nAgent: ", end="", flush=True)

        # TODO: Add user message to conversation and get response
        # Add your code here to:
        # 1. Add the user message to the conversation using conversations.items.create()
        # 2. Create a response using responses.create() with agent reference
        # 3. Extract and display the response text
        # 4. Check for and display any citations
        # Your code will go here
        #
        # (Again, the TODO comment is left over from the exercise version of
        # this lab - the actual implementation follows immediately below.)

        # Add user message to the conversation
        openai_client.conversations.items.create(
            conversation_id=conversation.id,
            items=[{"type": "message", "role": "user", "content": user_message}],
        )

        # Store in conversation history (client-side)
        conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Create a response using the agent
        # `extra_body={"agent_reference": ...}` is what makes this call use
        # OUR named agent's instructions/tools instead of a bare model.
        # `input=""` because the actual user text was already added to the
        # conversation above.
        response = openai_client.responses.create(
            conversation=conversation.id,
            extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            input=""
        )

        # Check if the response output contains an MCP approval request
        # If the portal agent has an MCP tool configured (similar to Lab 8's
        # agent.py), the model might pause here waiting for us to approve a
        # tool call before it can finish answering.
        approval_request = None
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'type') and item.type == 'mcp_approval_request':
                    approval_request = item
                    break

        # Handle approval request if present
        if approval_request:
            print(f"[Approval required for: {approval_request.name}]\n")
            print(f"Server: {approval_request.server_label}")

            # Parse and display the arguments (optional, for transparency)
            # Show the human exactly what arguments the agent wants to call
            # the tool with, so they can make an informed yes/no decision.
            import json
            try:
                args = json.loads(approval_request.arguments)
                print(f"Arguments: {json.dumps(args, indent=2)}\n")
            except:
                # If the arguments aren't valid JSON for some reason, just
                # show the raw string instead of crashing.
                print(f"Arguments: {approval_request.arguments}\n")

            # Prompt user for approval
            approval_input = input("Approve this action? (yes/no): ").strip().lower()

            if approval_input in ['yes', 'y']:
                print("Approving action...\n")

                # Create approval response item
                approval_response = {
                    "type": "mcp_approval_response",
                    "approval_request_id": approval_request.id,
                    "approve": True
                }
            else:
                print("Action denied.\n")

                # Create denial response item
                approval_response = {
                    "type": "mcp_approval_response",
                    "approval_request_id": approval_request.id,
                    "approve": False
                }

            # Add the approval response to the conversation
            # This is a slightly different mechanism than Lab 8's agent.py
            # (which sent the approval via responses.create(input=...)) -
            # here the approval is added as a conversation item instead.
            openai_client.conversations.items.create(
                conversation_id=conversation.id,
                items=[approval_response]
            )

            # Get the actual response after approval/denial
            # Ask the agent again now that the approval/denial has been recorded.
            response = openai_client.responses.create(
                conversation=conversation.id,
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                input=""
            )



        # Extract the response text
        if response and response.output_text:
            response_text = response.output_text

            print(f"{response_text}\n")

            # Check for citations if available
            # Some agents (e.g. ones with a knowledge-base/file-search tool)
            # attach source citations to their answer - print them if present.
            if hasattr(response, 'citations') and response.citations:
                print("\nSources:")
                for citation in response.citations:
                    print(f"  - {citation.content if hasattr(citation, 'content') else 'Knowledge Base'}")

            # Store in conversation history (client-side)
            conversation_history.append({
                "role": "assistant",
                "content": response_text
            })

            return response_text
        else:
            print("No response received.\n")
            return None
    except Exception as e:
        # Catch-all: print any error instead of crashing the whole chat loop
        print(f"\n\nError: {str(e)}\n")
        return None


def display_conversation_history():
    """
    Display the full conversation history.
    """
    # Purely a convenience for the user - prints the LOCAL
    # conversation_history list built up in send_message_to_agent() above.
    print("\n" + "="*60)
    print("CONVERSATION HISTORY")
    print("="*60 + "\n")

    for turn in conversation_history:
        role = turn["role"].upper()
        content = turn["content"]
        print(f"{role}: {content}\n")

    print("="*60 + "\n")


def main():
    """
    Main interaction loop.
    """
    print("Contoso Product Expert Agent")
    print("Ask questions about our outdoor and camping products.")
    print("Type 'history' to see conversation history, or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("\nEnding conversation...")
                break

            if user_input.lower() == 'history':
                display_conversation_history()
                continue

            # Send message and get response
            send_message_to_agent(user_input)

        except KeyboardInterrupt:
            # Lets Ctrl+C exit the loop cleanly instead of showing a traceback.
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}\n")

    print("\nConversation ended.")


if __name__ == "__main__":
    main()
