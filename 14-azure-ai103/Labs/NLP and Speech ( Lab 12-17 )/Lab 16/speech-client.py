# =============================================================================
# LAB 16: ⚠️ Despite the filename ("speech-client.py"), this code contains
# NO audio capture or synthesis logic at all - it's a plain TEXT chat loop,
# structurally identical to Lab 9's agent_client.py / Lab 13's
# text-agent.py, just with a `while True` loop added around a single
# question-and-answer exchange.
#
# The "speech" part of this lab almost certainly lives on the AGENT ITSELF,
# configured in the Foundry portal (e.g. an agent with a voice/telephony
# channel enabled), not in this Python file. Treat this script as "how to
# send text to that agent from Python once it exists" rather than as a
# speech-processing example - if you're looking for actual audio code, see
# Lab 15 (dedicated Speech SDK) or Lab 17 (real-time VoiceLive).
#
# Needs a pre-built portal agent: set FOUNDRY_ENDPOINT and AGENT_NAME in
# your .env to match an agent you've already created.
# =============================================================================

import os                       # Env vars, clear-screen command
import sys
from pathlib import Path

# import namespaces
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
        agent_name = config.agent_name("speech-client-agent")


        # Get project client
        # Get project client
        project_client = AIProjectClient(
            endpoint=foundry_endpoint,
            credential=DefaultAzureCredential(),
        )

        # Get an OpenAI client
        # Get an OpenAI client
        openai_client = project_client.get_openai_client()

        # Main loop
        while True:
            # Get user input
            prompt = input("User prompt (or 'quit'): ")
            if prompt == "quit" or len(prompt) == 0:
                break
            else:
                # Use the agent to get a response
                # Use the agent to get a response
                #
                # Same "reference the agent by name, no memory across
                # turns" pattern as Lab 13 - every question here is answered
                # independently; there's no conversation object being reused
                # to give the agent memory of earlier questions in this loop.
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
