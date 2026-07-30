# =============================================================================
# LAB 13: The simplest possible "one-shot" Foundry agent client - no loop,
# no memory, no tools. Read one prompt from the user, send it to a
# pre-built portal agent, print the reply, exit. Good as a quick smoke test
# that your agent name and credentials are set up correctly before moving on
# to the more elaborate labs.
#
# Needs a pre-built portal agent: create an agent in the Foundry portal
# first, then set FOUNDRY_ENDPOINT and AGENT_NAME in your .env to match it.
# =============================================================================

import os                       # Env vars, clear-screen command
import sys
from pathlib import Path

# Import namespaces
# import namespaces
from azure.identity import DefaultAzureCredential   # Authenticates using your `az login` session
from azure.ai.projects import AIProjectClient        # Talks to your Foundry project

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config


def main():
    try:
        # Clear the console
        os.system('cls' if os.name == 'nt' else 'clear')

        # Get Configuration Settings
        foundry_endpoint = config.project_endpoint
        agent_name = config.agent_name("text-agent")

        # Get project client
        # Get project client
        # No API key - DefaultAzureCredential uses your `az login` identity.
        project_client = AIProjectClient(
            endpoint=foundry_endpoint,
            credential=DefaultAzureCredential(),
        )


        # Get an OpenAI client
        # Get an OpenAI client
        # Wraps the project connection so we can call the OpenAI-compatible
        # .responses.create() API against it.
        openai_client = project_client.get_openai_client()


        # Use the agent to get a response
        # Use the agent to get a response
        #
        # Note there's no `openai_client.agents.get(...)` call here to load
        # the agent object first (unlike Labs 6/9/16) - this script just
        # passes the agent's NAME straight into `agent_reference` and lets
        # the API look it up implicitly.
        prompt = input("User prompt: ")

        response = openai_client.responses.create(
            input=[{"role": "user", "content": prompt}],
            extra_body={"agent_reference": {"name": agent_name, "type": "agent_reference"}},
        )

        print(f"{agent_name}: {response.output_text}")


    except Exception as ex:
        # Catch-all: print any error instead of crashing
        print(ex)

if __name__ == "__main__":
    main()
