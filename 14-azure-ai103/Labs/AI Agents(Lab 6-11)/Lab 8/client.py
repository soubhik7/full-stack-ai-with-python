# =============================================================================
# LAB 8 (part B - local MCP server): the CLIENT half. Run THIS file, not
# server.py directly - this script launches server.py itself as a
# subprocess, talks to it over MCP, and bridges its tools into a Foundry
# agent as ordinary FunctionTools.
#
# This is the general pattern for connecting ANY MCP server (not just this
# toy one) to a Foundry agent that only natively understands FunctionTool:
# 1. Connect to the MCP server and list its tools.
# 2. Wrap each MCP tool as a FunctionTool "shell" (name + description only).
# 3. When the agent calls one of those FunctionTools, forward the call over
#    MCP to the real server and hand the result back.
#
# Prerequisites: Azure AI Foundry project + `az login` + .env with
# PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME + `pip install mcp fastmcp`.
# =============================================================================

import os                          # Env vars, clear-screen command
import asyncio                     # Everything here is async because MCP communication is I/O-bound (subprocess pipes)
import json                        # Parses the model's function-call arguments
from dotenv import load_dotenv      # Loads .env file variables into the environment
from contextlib import AsyncExitStack  # Manages multiple async context managers (server connection, MCP session) and closes them all together on shutdown
from azure.ai.projects import AIProjectClient                       # Talks to your Foundry project
from azure.ai.projects.models import FunctionTool                   # Describes a callable tool to the agent
from azure.identity import DefaultAzureCredential                   # Authenticates using your `az login` session
from azure.ai.projects.models import PromptAgentDefinition, FunctionTool  # Agent definition + FunctionTool (imported twice here, harmless duplicate from the original course file)
from openai.types.responses.response_input_param import FunctionCallOutput, ResponseInputParam  # Typed shapes for returning function results to the model

# Add references
# Add references
from mcp import ClientSession, StdioServerParameters   # Core MCP client classes: a session (talks the MCP protocol) and server launch parameters
from mcp.client.stdio import stdio_client                # Starts an MCP server as a subprocess and connects to it over stdin/stdout

# Clear the console
os.system('cls' if os.name=='nt' else 'clear')

# Load environment variables from .env file
load_dotenv()
project_endpoint = os.getenv("PROJECT_ENDPOINT")
model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

async def connect_to_server(exit_stack: AsyncExitStack):
    # Describes HOW to launch the MCP server: run `python server.py` with no
    # extra environment variables. This assumes server.py lives in the same
    # folder you run this script from.
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
        env=None
    )

    # Start the MCP server
    # Start the MCP server
    # `exit_stack.enter_async_context(...)` starts the subprocess and wires
    # up its stdin/stdout as a transport, while registering it with the
    # exit_stack so it's automatically torn down (subprocess killed, pipes
    # closed) when exit_stack.aclose() runs at the end of main().
    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport   # a pair of async read/write streams talking to the subprocess

    # Create an MCP client session
    # Create an MCP client session
    # ClientSession speaks the actual MCP protocol (handshake, tool
    # discovery, tool calls) on top of those raw read/write streams.
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))
    await session.initialize()   # performs the MCP handshake with the server

    # List available tools
    # List available tools
    # Asks the running server.py "what tools do you have?" - this returns
    # get_inventory_levels and get_weekly_sales, as decorated in server.py.
    response = await session.list_tools()
    tools = response.tools
    print("\nConnected to server with tools:", [tool.name for tool in tools])

    return session

async def chat_loop(session):

    # Connect to the agents client
    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client,
    ):

        # Get the mcp tools available from the server
        response = await session.list_tools()
        tools = response.tools

        # Build a function for each tool
        # Build a function for each tool
        #
        # This is a small "factory" function: it returns a NEW async
        # function each time it's called, one per MCP tool, that knows how
        # to forward a call to that specific tool name over the MCP
        # session. This closure pattern is needed because a plain loop
        # variable (`tool_name`) would otherwise be shared/overwritten
        # across all the generated functions - the factory captures each
        # name correctly.
        def make_tool_func(tool_name):
            async def tool_func(**kwargs):
                # Forwards the call over MCP to the real server.py tool,
                # passing along whatever keyword arguments the model supplied.
                result = await session.call_tool(tool_name, kwargs)
                return result

            tool_func.__name__ = tool_name
            return tool_func

        # Store the functions in a dictionary for easy access when processing function calls
        # e.g. {"get_inventory_levels": <async function>, "get_weekly_sales": <async function>}
        functions_dict = {tool.name: make_tool_func(tool.name) for tool in tools}

        # Create FunctionTool definitions for the agent
        # Create FunctionTool definitions for the agent
        #
        # The Foundry agent only understands FunctionTool, not MCP directly -
        # so for each MCP tool discovered, we build a matching FunctionTool
        # "shell" with an empty parameter schema (these two demo tools take
        # no arguments) so the agent knows the tool exists and can call it.
        mcp_function_tools: FunctionTool = []
        for tool in tools:
            function_tool = FunctionTool(
                name=tool.name,
                description=tool.description,
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                strict=True
            )
            mcp_function_tools.append(function_tool)

        # Create the agent
        # Create the agent
        #
        # Instructions here encode simple business rules (when to recommend
        # restock vs. clearance) that the agent applies to whatever data it
        # gets back from the get_inventory_levels/get_weekly_sales tools.
        agent = project_client.agents.create_version(
            agent_name="inventory-agent",
            definition=PromptAgentDefinition(
                model=model_deployment,
                instructions="""
                You are an inventory assistant. Here are some general guidelines:
                - Recommend restock if item inventory < 10  and weekly sales > 15
                - Recommend clearance if item inventory > 20 and weekly sales < 5
                """,
                tools=mcp_function_tools
            ),
        )

        # Create a thread for the chat session
        conversation = openai_client.conversations.create()

        # Create an input list to hold function call outputs to send back to the model
        input_list: ResponseInputParam = []

        while True:
            user_input = input("Enter a prompt for the inventory agent. Use 'quit' to exit.\nUSER: ").strip()
            if user_input.lower() == "quit":
                print("Exiting chat.")
                break

            # Send a prompt to the agent
            openai_client.conversations.items.create(
                conversation_id=conversation.id,
                items=[{"type": "message", "role": "user", "content": user_input}],
            )

            # Retrieve the agent's response, which may include function calls to the MCP server tools
            response = openai_client.responses.create(
                conversation=conversation.id,
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                input=input_list,
            )

            # Check the run status for failures
            if response.status == "failed":
                print(f"Response failed: {response.error}")

            # Process function calls
            # Process function calls
            #
            # Same dispatch pattern as Lab 7, but generalized: instead of a
            # hardcoded if/elif chain per function name, we look up the
            # right async wrapper in functions_dict (built above from
            # whatever tools the MCP server happened to expose).
            for item in response.output:
                if item.type == "function_call":
                    # Retrieve the matching function tool
                    function_name = item.name
                    kwargs = json.loads(item.arguments)
                    required_function = functions_dict.get(function_name)

                    # Invoke the function
                    # This actually calls the wrapper from make_tool_func(),
                    # which forwards the call to server.py over MCP and
                    # awaits its reply.
                    output = await required_function(**kwargs)

                    # Append the output text
                    input_list.append(
                    FunctionCallOutput(
                        type="function_call_output",
                        call_id=item.call_id,
                        # MCP tool results come back as a structured object
                        # with a `.content` list; `[0].text` pulls out the
                        # first content item's plain text so it can be sent
                        # back to the model as a simple string.
                        output=output.content[0].text,
                    )
                    )

            # Send function call outputs back to the model and retrieve a response
            # Send function call outputs back to the model and retrieve a response
            if input_list:
                response = openai_client.responses.create(
                        input=input_list,
                        previous_response_id=response.id,
                        extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                )
            print(f"Agent response: {response.output_text}")

        # Delete the agent when done
        print("Cleaning up agents:")
        project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
        print("Deleted inventory agent.")


async def main():
    import sys   # imported here but not actually used below - a leftover from the original course file
    # AsyncExitStack collects every "with"-like resource we open (the MCP
    # server subprocess, the MCP session) so they can all be cleanly closed
    # together in the `finally` block, regardless of how the chat loop ends.
    exit_stack = AsyncExitStack()
    try:
        session = await connect_to_server(exit_stack)
        await chat_loop(session)
    finally:
        # Kills the server.py subprocess and closes the MCP session/pipes.
        await exit_stack.aclose()

if __name__ == "__main__":
    asyncio.run(main())
