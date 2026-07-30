# =============================================================================
# LAB 8 (part A - hosted/remote MCP tool): an agent whose only tool is a
# REMOTE MCP (Model Context Protocol) server run by Microsoft - not code you
# write yourself. MCP is a standard way for an AI agent to discover and call
# tools exposed by any compliant server, over HTTP or other transports.
#
# This demo also shows the "human approval" safety pattern: because the tool
# is marked require_approval="always", the agent can never silently call it -
# every single tool call comes back to us as a pending approval request that
# our code must explicitly say yes/no to before the agent can proceed.
#
# Contrast with client.py + server.py in this same folder, which build and
# consume a LOCAL MCP server instead of a hosted one.
#
# Prerequisites: Azure AI Foundry project + `az login` + .env with
# PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME.
# =============================================================================

import sys
from pathlib import Path

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config                    # Centralized Azure credential/config module

# Add references
# Add references
from azure.identity import DefaultAzureCredential                  # Authenticates using your `az login` session
from azure.ai.projects import AIProjectClient                      # Talks to your Foundry project
from azure.ai.projects.models import PromptAgentDefinition, MCPTool  # MCPTool describes a remote MCP server as an agent tool
from openai.types.responses.response_input_param import McpApprovalResponse, ResponseInputParam  # Typed shape for approving/denying an MCP tool call

# Load configuration from the centralized azure_config module
project_endpoint = config.project_endpoint
model_deployment = config.model_deployment

# Connect to the agents client
# Connect to the agents client
# `with (... as a, ... as b, ... as c):` opens all three context managers
# together, auto-closing every connection when the block ends.
with (
    DefaultAzureCredential() as credential,
    AIProjectClient(endpoint=project_endpoint, credential=credential) as project_client,
    project_client.get_openai_client() as openai_client,
):

    # Initialize agent MCP tool
    # Initialize agent MCP tool
    #
    # This points at Microsoft Learn's own public MCP server, which exposes
    # tools for searching Microsoft/Azure documentation. `server_label` is
    # just a local nickname you choose to refer to this tool by.
    # `require_approval="always"` means every call the agent wants to make
    # to this server must be explicitly approved by our code below before
    # it's actually executed - a safety net so an agent can't silently hit
    # an external service on your behalf.
    mcp_tool = MCPTool(
        server_label="api-specs",
        server_url="https://learn.microsoft.com/api/mcp",
        require_approval="always",
    )

    # Create a new agent with the MCP tool
    # Create a new agent with the MCP tool
    agent = project_client.agents.create_version(
        agent_name="MyAgent",
        definition=PromptAgentDefinition(
            model=model_deployment,
            instructions="You are a helpful agent that can use MCP tools to assist users. Use the available MCP tools to answer questions and perform tasks.",
            tools=[mcp_tool],
        ),
    )
    print(f"Agent created (id: {agent.id}, name: {agent.name}, version: {agent.version})")

    # Create conversation thread
    # Create a conversation thread
    conversation = openai_client.conversations.create()
    print(f"Created conversation (id: {conversation.id})")

    # Send initial request that will trigger the MCP tool
    # Send initial request that will trigger the MCP tool
    #
    # This is a single fixed question (not an interactive loop like other
    # labs) - it's phrased so the model will almost certainly want to
    # search Microsoft Learn docs to answer accurately, which is exactly
    # what triggers the MCP tool call we want to observe.
    response = openai_client.responses.create(
        conversation=conversation.id,
        input="Give me the Azure CLI commands to create an Azure Container App with a managed identity.",
    extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
    )

    # Process any MCP approval requests that were generated
    # Process any MCP approval requests that were generated
    # The agent may issue several tool calls, each needing its own approval,
    # so we loop until there are none left.
    while True:
        # Collect any MCP approval requests from the latest response
        input_list: ResponseInputParam = []
        for item in response.output:
            # An "mcp_approval_request" item appears whenever the model
            # wants to call the MCP tool but is blocked pending our OK.
            if item.type == "mcp_approval_request":
                if item.server_label == "api-specs" and item.id:
                    # Automatically approve the MCP request to allow the agent to proceed
                    # In a real app you might show this to a human and ask
                    # "allow this?" - here we auto-approve every request
                    # since this is a demo of a Microsoft-owned, read-only
                    # documentation server.
                    input_list.append(
                        McpApprovalResponse(
                            type="mcp_approval_response",
                            approve=True,
                            approval_request_id=item.id,   # ties this approval to the specific pending request
                        )
                    )

        # No more approvals needed -> the agent has produced its final response
        if not input_list:
            break

        print("Final input:")
        print(input_list)

        # Send the approval response back and retrieve the next response
        # `previous_response_id=response.id` continues the same reasoning
        # chain - the model picks up exactly where it left off, now able to
        # actually execute the approved tool call(s).
        response = openai_client.responses.create(
            input=input_list,
            previous_response_id=response.id,
            extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
        )

    print(f"\nAgent response: {response.output_text}")
    # Clean up resources by deleting the agent version
    # Clean up resources by deleting the agent version
    # This script created the agent itself, so it deletes it at the end
    # rather than leaving it behind in the Foundry project.
    project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
    print("Agent deleted")
