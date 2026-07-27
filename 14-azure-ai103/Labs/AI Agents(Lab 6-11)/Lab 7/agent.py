# =============================================================================
# LAB 7: A complete, self-contained example of FUNCTION CALLING (a.k.a.
# "tools") with a Foundry agent, built entirely in code (no portal setup
# needed, unlike Lab 6). This is the clearest lab to study first if you're
# new to agent tools.
#
# The agent plays the role of an astronomy-observations assistant that can:
#   - look up the next visible astronomical event for a location
#   - calculate telescope rental cost
#   - generate a text report combining both
# Each of those is a plain Python function in functions.py - this file's job
# is to (1) describe those functions to the model as "tools" it's allowed to
# call, (2) actually run them when the model asks, and (3) send the results
# back so the model can give a final natural-language answer.
#
# Needs: a `data/` folder next to this file (see functions.py's header
# comment for the exact file formats) - create it yourself, it's not
# included in this repo.
#
# Prerequisites: Azure AI Foundry project + `az login` + .env with
# PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME.
# =============================================================================

import os                      # Env vars, clear-screen command
import json                    # Parses/builds JSON (tool arguments and results are JSON strings)
from dotenv import load_dotenv  # Loads .env file variables into the environment

# Add references
# Add references
from azure.ai.projects import AIProjectClient                      # Talks to your Foundry project
from azure.ai.projects.models import FunctionTool                  # Describes a Python function to the agent as a callable "tool"
from azure.identity import DefaultAzureCredential                  # Authenticates using your `az login` session
from azure.ai.projects.models import PromptAgentDefinition, FunctionTool  # Defines an agent's model/instructions/tools (FunctionTool imported twice here in the original course file - harmless duplicate import)
from openai.types.responses.response_input_param import FunctionCallOutput, ResponseInputParam  # Typed shapes for sending function results back to the model
from functions import next_visible_event, calculate_observation_cost, generate_observation_report  # The actual Python functions this agent can call - see functions.py

def main():
    # Clear the console
    os.system('cls' if os.name=='nt' else 'clear')

    # Load environment variables from .env file
    load_dotenv()
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

    # Connect to the project client
    # Connect to the project client
    #
    # `with (... as a, ... as b, ... as c):` opens all three "context
    # managers" together and automatically closes/cleans them up (network
    # connections etc.) when the `with` block ends - equivalent to nesting
    # three separate `with` statements.
    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client,
    ):

        # Define the event function tool
        # Define the event function tool
        #
        # A FunctionTool is essentially a JSON-schema description of a
        # Python function's name, purpose, and parameters - it tells the
        # model "this tool exists, here's what it does, here's what
        # arguments it needs" WITHOUT giving the model the actual code.
        # The model can only ever request that this tool be called with
        # specific arguments; running the real Python function is still up
        # to us (see the "Process function calls" section below).
        event_tool = FunctionTool(
            name="next_visible_event",                      # must match the Python function name in functions.py
            description="Get the next visible event in a given location.",  # helps the model decide WHEN to use this tool
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "continent to find the next visible event in (e.g. 'north_america', 'south_america', 'australia')",
                    },
                },
                "required": ["location"],
                "additionalProperties": False,   # rejects any extra fields the model might try to invent
            },
            strict=True,   # forces the model to follow this schema exactly (no missing/extra fields)
        )

        # Define the observation cost function tool
        # Define the observation cost function tool
        cost_tool = FunctionTool(
            name="calculate_observation_cost",
            description="Calculate the cost of an observation based on the telescope tier, number of hours, and priority level.",
            parameters={
                "type": "object",
                "properties": {
                    "telescope_tier": {
                        "type": "string",
                        "description": "the tier of the telescope (e.g. 'standard', 'advanced', 'premium')",
                    },
                    "hours": {
                        "type": "number",
                        "description": "the number of hours for the observation",
                    },
                    "priority": {
                        "type": "string",
                        "description": "the priority level of the observation (e.g. 'low', 'normal', 'high')",
                    },
                },
                "required": ["telescope_tier", "hours", "priority"],
                "additionalProperties": False,
            },
            strict=True,
        )

        # Define the observation report generation function tool
        # Define the observation report generation function tool
        report_tool = FunctionTool(
            name="generate_observation_report",
            description="Generate a report summarizing an astronomical observation",
            parameters={
                "type": "object",
                "properties": {
                    "event_name": {
                        "type": "string",
                        "description": "the name of the astronomical event being observed",
                    },
                    "location": {
                        "type": "string",
                        "description": "the location of the observer",
                    },
                    "telescope_tier": {
                        "type": "string",
                        "description": "the tier of the telescope used for the observation (e.g. 'standard', 'advanced', 'premium')",
                    },
                    "hours": {
                        "type": "number",
                        "description": "the number of hours the telescope was used for the observation",
                    },
                    "priority": {
                        "type": "string",
                        "description": "the priority level of the observation (e.g. 'low', 'normal', 'high')",
                    },
                    "observer_name": {
                        "type": "string",
                        "description": "the name of the person who conducted the observation",
                    },
                },
                "required": ["event_name", "location", "telescope_tier", "hours", "priority", "observer_name"],
                "additionalProperties": False,
            },
            strict=True,
        )

        # Create a new agent with the function tools
        # Create a new agent with the function tools
        #
        # Unlike Lab 6 (which looked up an agent already made in the
        # portal), here we CREATE a brand-new, versioned agent entirely in
        # code - its model, instructions, and the three tools above are all
        # defined right here.
        agent = project_client.agents.create_version(
            agent_name="astronomy-agent",
            definition=PromptAgentDefinition(
                model=model_deployment,
                instructions=
                    """You are an astronomy observations assistant that helps users find
                    information about astronomical events and calculate telescope rental costs.
                    Use the available tools to assist users with their inquiries.""",
                tools=[event_tool, cost_tool, report_tool],
            ),
        )

        # Create a thread for the chat session
        # Create a thread for the chat session
        # (Despite the comment saying "thread", this API calls it a
        # "conversation" - same concept as Lab 6: a server-side container
        # that holds the message history for this session.)
        conversation = openai_client.conversations.create()

        # Create a list to hold function call outputs that will be sent back as input to the agent
        # Create a list to hold function call outputs that will be sent back as input to the agent
        # This list is reused across the whole chat loop - each turn appends
        # any function results the model asked for, then sends them back.
        input_list: ResponseInputParam = []

        while True:
            user_input = input("Enter a prompt for the astronomy agent. Use 'quit' to exit.\nUSER: ").strip()
            if user_input.lower() == "quit":
                print("Exiting chat.")
                break

            # Send a prompt to the agent
            # Send a prompt to the agent
            openai_client.conversations.items.create(
                conversation_id=conversation.id,
                items=[{"type": "message", "role": "user", "content": user_input}],
            )

            # Retrieve the agent's response, which may include function calls
            # Retrieve the agent's response, which may include function calls
            #
            # `extra_body={"agent_reference": ...}` tells the API to answer
            # using OUR agent (with its tools/instructions) rather than a
            # bare model. `input=input_list` starts empty on the first turn,
            # but on later turns may carry leftover function outputs.
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
            # The model doesn't run Python itself - when it decides a tool
            # is needed, the response contains a "function_call" item naming
            # which tool and with what arguments. WE are responsible for
            # actually calling the matching real Python function and
            # collecting its return value.
            for item in response.output:
                if item.type == "function_call":
                    # Retrieve the matching function tool
                    function_name = item.name
                    result = None
                    # Dispatch by name to the matching real Python function,
                    # unpacking the model-supplied JSON arguments as keyword
                    # arguments (json.loads turns the JSON string back into
                    # a dict, and **dict unpacks it as name=value pairs).
                    if item.name == "next_visible_event":
                        result = next_visible_event(**json.loads(item.arguments))
                    elif item.name == "calculate_observation_cost":
                        result = calculate_observation_cost(**json.loads(item.arguments))
                    elif item.name == "generate_observation_report":
                        result = generate_observation_report(**json.loads(item.arguments))

                    # Append the output text
                    # Package the function's return value as a
                    # FunctionCallOutput, tagged with the same call_id the
                    # model used to request it - that's how the model knows
                    # which of (possibly several) tool calls this result
                    # answers.
                    input_list.append(
                        FunctionCallOutput(
                            type="function_call_output",
                            call_id=item.call_id,
                            output=result,
                        )
                    )

            # Send function call outputs back to the model and retrieve a response
            # Send function call outputs back to the model and retrieve a response
            #
            # Only make this second call if there actually WERE function
            # calls to answer. `previous_response_id=response.id` tells the
            # API "continue from that response" so the model can now use
            # the tool results to write its final natural-language answer.
            if input_list:
                response = openai_client.responses.create(
                    input=input_list,
                    previous_response_id=response.id,
                    extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                )
            # Display the agent's response
            print(f"AGENT: {response.output_text}")

        # Delete the agent when done
        # Delete the agent when done
        #
        # Unlike Lab 6 (which reused a permanent portal agent), this agent
        # was created fresh by THIS script, so it's good practice to clean
        # it up when the chat session ends - otherwise you'd accumulate a
        # new "astronomy-agent" version every time you run this file.
        project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
        print("Deleted agent.")

if __name__ == '__main__':
    main()
