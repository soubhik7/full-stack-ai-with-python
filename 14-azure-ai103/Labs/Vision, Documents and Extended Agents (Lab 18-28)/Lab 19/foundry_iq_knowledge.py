# =============================================================================
# LAB 19 (NEW — added to fill an AI-103 syllabus gap, not an original course
# file).
#
# ⚠️⚠️⚠️ PREVIEW / BEST-EFFORT — UNVERIFIED CODE ⚠️⚠️⚠️
# Foundry IQ is named in this chapter's EXAM_NOTES.md (see "8 · Building
# agents with Foundry") as one of "two headline features AI-102 never had":
# a SHARED KNOWLEDGE PLATFORM where you register a knowledge source (docs,
# an index, a database, etc.) ONCE, and many different agents can ground
# their answers on it, all producing consistently CITED responses -
# "RAG-as-a-platform" instead of every agent wiring its own retrieval code.
#
# HOWEVER: no `.py` file anywhere in this repo actually calls a Foundry IQ
# API, and this lab's author could not find or verify a stable, public
# Python SDK class for it at the time of writing. The class/method names
# below (`KnowledgeSource`, `project_client.knowledge_sources.create(...)`,
# a `KnowledgeTool`) are this author's BEST GUESS at the shape, modeled on
# the real, working `MCPTool`/`FunctionTool`/`ConnectedAgentTool` pattern
# already proven elsewhere in this chapter (all three live in
# `azure.ai.projects.models`) - they are NOT confirmed to exist.
#
# Before running this: check the current Azure AI Foundry / `azure-ai-
# projects` SDK reference on Microsoft Learn for the real class names, and
# update the imports/calls below to match. Treat this file as a CONCEPT
# DIAGRAM you can execute, not a proven-working script.
#
# Prerequisites (once you've fixed up the real API shape): Azure AI Foundry
# project + `az login` + .env with PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME
# + a knowledge source already ingested (e.g. an Azure AI Search index).
# =============================================================================

import os
import sys
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

# --- Best-effort, unverified import — CONFIRM against current docs first ---
try:
    # Guessed class names, modeled on the real MCPTool/FunctionTool shape
    # that DOES exist in azure.ai.projects.models.
    from azure.ai.projects.models import PromptAgentDefinition, KnowledgeTool  # type: ignore
    KNOWLEDGE_TOOL_AVAILABLE = True
except ImportError:
    from azure.ai.projects.models import PromptAgentDefinition
    KNOWLEDGE_TOOL_AVAILABLE = False


def main():
    os.system('cls' if os.name == 'nt' else 'clear')

    if not KNOWLEDGE_TOOL_AVAILABLE:
        print("⚠️  `KnowledgeTool` could not be imported from azure.ai.projects.models.")
        print("This confirms the class name used below is not (yet) real in your installed")
        print("SDK version - check Microsoft Learn's current Foundry IQ documentation for")
        print("the actual class/method names, then update this script accordingly.\n")
        print("Showing the CONCEPT this lab is meant to teach instead of failing outright:\n")
        print_concept_explanation()
        return

    project_endpoint = config.project_endpoint
    model_deployment = config.model_deployment
    # A knowledge source you've already registered in the Foundry portal
    # (e.g. pointing at an ingested Azure AI Search index) - conceptually
    # the Foundry IQ equivalent of this chapter's AGENT_NAME env var.
    knowledge_source_name = config.knowledge_source_name

    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(),
    )

    # Register / reference a shared knowledge source
    # Register / reference a shared knowledge source
    #
    # The headline idea of Foundry IQ: this knowledge source is registered
    # ONCE, centrally, and can be referenced by MANY different agents below
    # (and by agents in other projects) - contrast with Lab 4's file-search
    # vector store or Lab 7/8's per-agent tools, which are each wired up
    # individually, agent by agent.
    knowledge_tool = KnowledgeTool(
        knowledge_source_name=knowledge_source_name,
        # `citations=True`-style setting is the other headline feature:
        # every answer grounded on this source comes back with consistent
        # source citations, regardless of which agent asked the question.
    )

    # Any number of agents can attach the SAME knowledge tool
    # Any number of agents can attach the SAME knowledge tool
    support_agent = project_client.agents.create_version(
        agent_name="support-agent-with-shared-knowledge",
        definition=PromptAgentDefinition(
            model=model_deployment,
            instructions="Answer support questions, grounding every answer in the shared knowledge source and citing it.",
            tools=[knowledge_tool],
        ),
    )
    sales_agent = project_client.agents.create_version(
        agent_name="sales-agent-with-shared-knowledge",
        definition=PromptAgentDefinition(
            model=model_deployment,
            instructions="Answer product questions, grounding every answer in the shared knowledge source and citing it.",
            tools=[knowledge_tool],
        ),
    )
    print(f"Two independent agents ({support_agent.name}, {sales_agent.name}) now ground on the SAME registered knowledge source.")

    project_client.agents.delete_version(agent_name=support_agent.name, agent_version=support_agent.version)
    project_client.agents.delete_version(agent_name=sales_agent.name, agent_version=sales_agent.version)


def print_concept_explanation():
    print("""
Foundry IQ, conceptually:

  1. Register a knowledge source ONCE (e.g. an Azure AI Search index,
     a set of documents) at the PROJECT level, not per-agent.
  2. Any number of agents attach that same knowledge source as a tool.
  3. Every one of those agents' answers comes back with consistent,
     CITED sources - because the citation logic lives in the shared
     platform, not duplicated in each agent's own retrieval code.

Compare this to the RAG patterns proven elsewhere in this chapter:
  - Lab 4 (file-search vector store) - scoped to ONE agent's files.
  - `02_foundry_agent_service/10_customer_rag_client.py` - "bring your own
    retrieval": the CLIENT code queries Azure AI Search directly and
    manually stuffs results into the prompt, agent by agent.
Foundry IQ's pitch is to replace both of those with one shared,
managed knowledge layer.
""")


if __name__ == "__main__":
    main()
