# =============================================================================
# LAB 11: Multi-agent ORCHESTRATION using Microsoft Agent Framework's
# SequentialBuilder. Instead of one agent doing everything, this pipelines
# THREE small, narrowly-instructed agents one after another - the output of
# each becomes the input to the next.
#
# Scenario: customer feedback comes in -> agent 1 summarizes it -> agent 2
# classifies it -> agent 3 suggests a next action. This "chain of small
# specialists" pattern tends to be more reliable than asking one big agent
# to do all three things in a single prompt.
#
# Requires: pip install agent-framework agent-framework-azure-ai (same as
# Lab 10 - not in the root requirements.txt).
#
# Prerequisites: Azure AI Foundry project + `az login` + .env with
# AZURE_AI_PROJECT_ENDPOINT and AZURE_AI_MODEL_DEPLOYMENT_NAME.
# ⚠️ Note the exact names: this repo's root .env (see CLAUDE.md) already
# defines AZURE_AI_PROJECT_ENDPOINT and AZURE_AI_MODEL_DEPLOYMENT (no
# "_NAME" suffix) for chapter 11's labs - this script expects the "_NAME"
# variant, so you may need to add that extra variable.
# =============================================================================

import asyncio                     # Agent Framework is async end-to-end
import os                          # Env vars
from typing import cast            # Pure type-hinting helper (tells the type checker "trust me, this is a list[Message]") - no runtime effect
from dotenv import load_dotenv      # Loads .env file variables into the environment

# Add references
# Add references
from agent_framework import Message                              # The type representing one chat message (role + text + author)
from agent_framework.foundry import FoundryChatClient             # Connects Agent Framework to your Foundry project/model
from agent_framework.orchestrations import SequentialBuilder      # Builds a pipeline that runs multiple agents one after another
from azure.identity import AzureCliCredential                      # Authenticates via your `az login` session

load_dotenv()

async def main():
    # Agent instructions
    # Each agent below gets ONE narrow job and an example of the output
    # format expected - this keeps every step predictable and easy to debug
    # individually, instead of one agent trying to do everything at once.
    summarizer_instructions="""
    Summarize the customer's feedback in one short sentence. Keep it neutral and concise.
    Example output:
    App crashes during photo upload.
    User praises dark mode feature.
    """

    classifier_instructions="""
    Classify the feedback as one of the following: Positive, Negative, or Feature request.
    """

    action_instructions="""
    Based on the summary and classification, suggest the next action in one short sentence.
    Example output:
    Escalate as a high-priority bug for the mobile team.
    Log as positive feedback to share with design and marketing.
    Log as enhancement request for product backlog.
    """

    # Create the chat client
    # Create the chat client
    # One shared client, reused to build all three agents below (they can
    # all use the same underlying model deployment, just with different
    # instructions).
    credential = AzureCliCredential()
    chat_client = FoundryChatClient(
        credential=credential,
        project_endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
        model=os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME"),
    )

    # Create agents
    # Create agents
    # `chat_client.as_agent(...)` is a shorthand for building an Agent tied
    # to this client, without the `async with Agent(...) as agent:` block
    # style used in Lab 10 - simpler because these three agents don't need
    # any tools, just instructions.
    summarizer_agent = chat_client.as_agent(
        name="summarizer",
        instructions=summarizer_instructions,
    )

    classifier_agent = chat_client.as_agent(
        name="classifier",
        instructions=classifier_instructions,
    )

    action_agent = chat_client.as_agent(
        name="action",
        instructions=action_instructions,
    )

    # Initialize the current feedback
    # Initialize the current feedback
    # A hardcoded sample - this script always processes the SAME feedback
    # text; it's meant to demonstrate the orchestration pattern, not take
    # live user input.
    feedback="""
    I use the dashboard every day to monitor metrics, and it works well overall.
    But when I'm working late at night, the bright screen is really harsh on my eyes.
    If you added a dark mode option, it would make the experience much more comfortable.
    """

    # Build sequential orchestration
    # Build sequential orchestration
    #
    # `participants=[...]` lists the agents IN THE ORDER they should run:
    # summarizer first, then classifier, then action. `output_from="all"`
    # means the final result includes every agent's output (not just the
    # last one), so we can print the whole chain of reasoning below.
    workflow = SequentialBuilder(
        participants=[summarizer_agent, classifier_agent, action_agent],
        output_from="all",
    ).build()

    # Run and collect outputs
    # Run and collect outputs
    #
    # `workflow.run(...)` feeds the initial feedback text to the FIRST
    # agent, then automatically pipes that agent's reply as the input to
    # the SECOND agent, and so on - this is the whole point of
    # SequentialBuilder: you don't have to manually pass each agent's output
    # into the next agent's call yourself.
    result = await workflow.run(f"Customer feedback: {feedback}")
    outputs = result.get_outputs()   # one entry per agent in the pipeline

    # Display outputs
    # Display outputs
    #
    # Walks through each agent's response in order and prints every message
    # it produced, numbered, with the agent's name (or "assistant"/"user" as
    # a fallback) so you can see the whole summarize -> classify -> act chain.
    i = 1
    for response in outputs:
        for msg in cast(list[Message], response.messages):
            name = msg.author_name or ("assistant" if msg.role == "assistant" else "user")
            print(f"{'-' * 60}\n{i:02d} [{name}]\n{msg.text}")
            i += 1


if __name__ == "__main__":
    asyncio.run(main())
