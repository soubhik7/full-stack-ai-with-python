# =============================================================================
# LAB 10: Introduces Microsoft's newer, higher-level "Agent Framework" SDK
# (`agent_framework` / `agent_framework.foundry`) as an alternative to using
# `azure-ai-projects` directly (as Labs 6-9 did). Agent Framework wraps a lot
# of the boilerplate (creating conversations, dispatching function calls,
# etc.) behind a simpler `Agent(...)` object and a `@tool` decorator.
#
# Scenario: an "expense claim" assistant that reads a text file of expenses,
# drafts an email requesting reimbursement, and "sends" it (really just
# prints it - no real email is sent).
#
# Needs a `data.txt` file in this same folder (not included) - plain text
# describing some expenses, e.g.:
#   Taxi to airport - $45.00
#   Hotel (2 nights) - $210.00
#
# Requires: pip install agent-framework agent-framework-azure-ai (imported
# as `agent_framework` / `agent_framework.foundry` - not in the root
# requirements.txt).
#
# Prerequisites: Azure AI Foundry project + `az login` (via AzureCliCredential)
# + .env with PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME.
# =============================================================================

import os                          # Env vars, clear-screen command
import asyncio                     # Agent Framework is async end-to-end
from pathlib import Path           # Cross-platform file path handling
from dotenv import load_dotenv      # Loads .env file variables into the environment

# Add references
# Add references
from agent_framework import tool, Agent                 # `tool` decorator turns a function into an agent tool; `Agent` is the main assistant object
from agent_framework.foundry import FoundryChatClient    # Connects Agent Framework to your Azure AI Foundry project/model
from azure.identity import AzureCliCredential            # Authenticates specifically via your `az login` session
from pydantic import Field                                # Used to attach descriptions to tool parameters
from typing import Annotated                               # Lets us attach that Field metadata to a type hint

load_dotenv()

async def main():
    # Clear the console
    os.system('cls' if os.name=='nt' else 'clear')

    # Load the expnses data file
    # Path(__file__).parent = the folder this script lives in, so `data.txt`
    # is always found regardless of what directory you ran python from.
    script_dir = Path(__file__).parent
    file_path = script_dir / 'data.txt'
    with file_path.open('r') as file:
        data = file.read() + "\n"

    # Ask for a prompt
    # Shows the loaded expense data to the user as context, then asks what
    # they'd like done with it (e.g. "submit this as a claim").
    user_prompt = input(f"Here is the expenses data in your file:\n\n{data}\n\nWhat would you like me to do with it?\n\n")

    # Run the async agent code
    await process_expenses_data(user_prompt, data)

async def process_expenses_data(prompt, expenses_data):

    # Create a foundry chat client
    # Create a foundry chat client
    # This is Agent Framework's equivalent of AIProjectClient +
    # get_openai_client() combined - one object that knows how to talk to
    # your Foundry project's deployed model.
    client = FoundryChatClient(
        project_endpoint=os.getenv("PROJECT_ENDPOINT"),
        model=os.getenv("MODEL_DEPLOYMENT_NAME"),
        credential=AzureCliCredential()
    )

    # Initialize an agent with the tool and instructions
    # Initialize an agent with the tool and instructions
    #
    # `async with Agent(...) as agent:` creates a temporary agent scoped to
    # this block - it's automatically cleaned up when the block ends,
    # similar in spirit to the `with (...)` blocks in the azure-ai-projects
    # labs, but here it's a single object instead of three.
    # `tools=[submit_claim]` - notice we pass the plain Python function
    # directly (decorated with @tool below), not a hand-written JSON schema
    # like the FunctionTool examples in Labs 7/8 - Agent Framework derives
    # the schema automatically from the function's type hints and Field
    # descriptions.
    async with (
        Agent(
            client=client,
            name="ExpenseClaimAgent",
            instructions="""You are an AI assistant for expense claim submission.
                        At the user's request, create an expense claim and use the plug-in function to send an email to expenses@contoso.com with the subject 'Expense Claim`and a body that contains itemized expenses with a total.
                        Then confirm to the user that you've done so. Don't ask for any more information from the user, just use the data provided to create the email.""",
            tools=[submit_claim],
        ) as agent,
    ):

        # Use the agent to process the expenses data
        # Use the agent to process the expenses data
        try:
            # Add the input prompt to a list of messages to be submitted
            # Combines what the user asked for with the raw expense data, in
            # one message, so the agent has everything it needs in a single turn.
            prompt_messages = [f"{prompt}: {expenses_data}"]
            # Invoke the agent for the specified thread with the messages
            # `agent.run(...)` is Agent Framework's all-in-one call: it
            # sends the prompt, lets the agent decide to call submit_claim
            # if needed, runs that tool automatically, and returns the final
            # answer - all the manual "check for function_call items, run
            # them, send results back" plumbing from Labs 7/8 happens
            # internally here.
            response = await agent.run(prompt_messages)
            # Display the response
            print(f"\n# Agent:\n{response}")
        except Exception as e:
            # Something went wrong
            print (e)


# Create a tool function for the email functionality
# Create a tool function for the email functionality
#
# `@tool(approval_mode="never_require")` marks this function as an agent
# tool that the agent can call WITHOUT asking a human for approval first -
# the opposite end of the trust spectrum from Lab 8's MCP tool, which
# required approval="always". `Annotated[str, Field(description=...)]` is
# how Agent Framework figures out the tool's parameter schema and
# descriptions automatically, without you writing raw JSON schema by hand.
@tool(approval_mode="never_require")
def submit_claim(
    to: Annotated[str, Field(description="Who to send the email to")],
    subject: Annotated[str, Field(description="The subject of the email.")],
    body: Annotated[str, Field(description="The text body of the email.")]):
        # This doesn't actually send an email - it just prints what WOULD
        # be sent, which is enough to demonstrate the tool-calling flow
        # without needing real email credentials.
        print("\nTo:", to)
        print("Subject:", subject)
        print(body, "\n")


if __name__ == "__main__":
    asyncio.run(main())
