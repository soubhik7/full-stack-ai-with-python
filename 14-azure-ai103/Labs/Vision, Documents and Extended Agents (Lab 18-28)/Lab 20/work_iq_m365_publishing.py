# =============================================================================
# LAB 20 (NEW — added to fill an AI-103 syllabus gap, not an original course
# file).
#
# ⚠️⚠️⚠️ PREVIEW / BEST-EFFORT — UNVERIFIED CODE ⚠️⚠️⚠️
# EXAM_NOTES.md ("8 · Building agents with Foundry") names Microsoft 365
# integration as the second "headline feature AI-102 never had": you can
# PUBLISH a Foundry agent so it's usable inside Microsoft Teams and
# Microsoft 365 Copilot, and give it (with the signed-in user's own
# permissions) access to their workplace data - mail, files, chats - via
# **Work IQ**.
#
# In practice, "publishing an agent to Teams/M365 Copilot" is primarily a
# PORTAL / admin-center action (Teams Admin Center app registration, a
# Microsoft 365 Copilot agent manifest), not a single Python SDK call this
# author could find or verify anywhere in this repo or a stable public
# package. What follows is illustrative pseudocode of the CONCEPT, using
# plausible naming borrowed from the (real, but only partially-Python-mature
# at the time of writing) Microsoft 365 Agents SDK / Bot Framework lineage -
# it is NOT proven to run. Check Microsoft Learn's current Microsoft 365
# Agents SDK / Foundry "publish to Teams" documentation before relying on
# any of the class/method names below.
#
# Prerequisites (conceptual, once you verify real API names): a Foundry
# agent already created (see Lab 6-11) + Microsoft 365 admin/developer
# access to register an app in Teams Admin Center + Work IQ data
# permissions consented by the end user.
# =============================================================================

import os
from dotenv import load_dotenv

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential


def main():
    os.system('cls' if os.name == 'nt' else 'clear')

    load_dotenv()
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    agent_name = os.getenv("AGENT_NAME")

    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(),
    )

    # This part IS real and proven: looking up an existing agent, exactly
    # like Lab 6/9/13/16 do.
    agent = project_client.agents.get(agent_name=agent_name)
    print(f"Found agent to publish: {agent.name} (id: {agent.id})\n")

    print_publishing_concept(agent)

    # --- Everything below this line is UNVERIFIED best-effort pseudocode ---
    try:
        publish_to_teams_and_copilot(agent)
    except Exception as ex:
        print(f"\n(Expected in this preview lab: {ex})")
        print("This confirms the call below is illustrative, not a proven SDK surface.")


def print_publishing_concept(agent):
    print("""Microsoft 365 / Work IQ publishing, conceptually:

  1. Package the Foundry agent as an M365 "declarative agent" or Teams app
     manifest, referencing the agent's Foundry project/agent id.
  2. Register that manifest in the Teams Admin Center (or via the
     Microsoft 365 Agents Toolkit) so it's installable in Teams and
     discoverable inside Microsoft 365 Copilot.
  3. Grant the agent **Work IQ** access - this lets it read the SIGNED-IN
     USER's own mail/files/chats at query time, scoped to that user's
     existing permissions (it does NOT get standing access to everyone's
     data - each request is evaluated against the calling user's rights).
  4. End users then talk to the same underlying Foundry agent from inside
     Teams/Copilot, with no separate bot-hosting infrastructure required.

This is fundamentally an ADMIN/PORTAL workflow (app registration, manifest,
tenant consent) layered on top of the agent you already built in code -
unlike Labs 6-18, there isn't a single Python call that "does" the
publishing step.
""")


def publish_to_teams_and_copilot(agent):
    # ⚠️ UNVERIFIED — illustrative only. The Microsoft 365 Agents SDK's
    # exact Python surface for this operation was not confirmed at the time
    # of writing; this is a best-effort sketch of what such a call would
    # look like, modeled on Azure SDKs' usual "create manifest, register,
    # deploy" resource pattern.
    from azure.ai.projects.models import PromptAgentDefinition  # noqa: F401 - illustrative import only

    manifest = {
        "agent_id": agent.id,
        "agent_name": agent.name,
        "target_surfaces": ["teams", "m365_copilot"],
        "work_iq_scopes": ["Mail.Read", "Files.Read", "Chat.Read"],  # user-consented, per-request scopes
    }
    print(f"\n(illustrative) Registering M365 publishing manifest: {manifest}")
    # A real implementation would call something like the Microsoft 365
    # Agents Toolkit CLI/SDK or Teams Admin Center Graph API here - neither
    # is wired up in this lab.
    raise NotImplementedError(
        "No verified SDK call exists in this lab for the actual publish step - "
        "see Microsoft Learn's Microsoft 365 Agents SDK / Foundry docs for the real API."
    )


if __name__ == "__main__":
    main()
