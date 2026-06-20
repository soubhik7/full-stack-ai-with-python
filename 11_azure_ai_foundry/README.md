# Chapter 11 — Azure AI Foundry & AI Agent Fundamentals

> **You will run 8 hands-on labs against a real Azure AI Foundry project — covering the platform's workflow, tools, a reusable toolbox, prompt vs. hosted agents, multi-agent orchestration, and knowledge-grounded RAG.**

Companion to `Azure_AI_Foundry_Study_Notes.pdf` (194 pages, 4 modules). The notes cover the concepts and portal UI; these labs are the same concepts done with code, run from your own machine against your own Foundry project.

---

## Prerequisites

- An Azure AI Foundry project with at least one chat model deployed (these labs were built and verified against a `gpt-4.1` deployment)
- Azure CLI signed in: `az login` — auth is Entra ID via `DefaultAzureCredential`, **no API keys**
- Two values in `.env` at the repo root:
  ```
  AZURE_AI_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
  AZURE_AI_MODEL_DEPLOYMENT=<your-deployment-name>
  ```
  Find your project endpoint with:
  ```bash
  az rest --method get --url "https://management.azure.com/subscriptions/<sub-id>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<account>/projects?api-version=2025-04-01-preview"
  ```
- `pip install -r requirements.txt` (installs `azure-ai-projects`, `azure-ai-agents`, `azure-identity`)

---

## Theme

All 8 labs share one running example — a fictional **Crystal Hotels** booking assistant — the same scenario the study notes use (Crystal Hotels, LuxeStay, Oakwood Hotel). Concepts build on each other instead of each lab inventing a new domain.

---

## Labs (00 → 07)

| Lab | Concept | Run |
|-----|---------|-----|
| `00_setup/` | Verify `.env` + auth work | `python verify_connection.py` |
| `01_workflow/` | Foundry's 5-component workflow, made programmatic | `python main.py` |
| `02_tools/` | Function tool + built-in code interpreter, each in isolation | `python function_tool.py`, `python code_interpreter_tool.py` |
| `03_toolbox/` | Bundling multiple tools into one reusable `ToolSet` | `python main.py` |
| `04_prompt_agent/` | Stateless prompt-only agent — no persisted Agent resource | `python main.py` |
| `05_hosted_agent/` | Persistent Agent Service — agent + thread with memory | `python main.py` |
| `06_connected_agents/` | Multi-agent orchestration via `ConnectedAgentTool` | `python main.py` |
| `07_knowledge_rag/` | Grounding answers in an uploaded doc via file search | `python main.py` |

Each lab is independent and self-contained (per this repo's convention) — run any one without having run the others first, aside from `00_setup` as a sanity check.

---

## Mapping to the study notes

| Lab | Study notes section |
|-----|---------------------|
| `01_workflow/` | §7.2 Five Essential Components, §7.3 Workflow Summary |
| `02_tools/`, `03_toolbox/` | §7.1 "a toolbox that simplifies the creation and deployment of AI agents" |
| `04_prompt_agent/`, `05_hosted_agent/` | §61 Agent Architecture Layers |
| `06_connected_agents/` | §61, Figure 61.2 — Sequential Multi-Agent Architecture |
| `07_knowledge_rag/` | §41–44 Knowledge Base Architecture & Integration Workflow |

---

## Cleanup

Every lab deletes the agents, threads, vector stores, and files it creates by the end of its own run (`05_hosted_agent/` is the one exception that explains, rather than performs, why you'd normally leave a hosted agent running in production). Nothing accumulates in your Foundry project from running these labs repeatedly.

---

## Previous Chapter

← [Chapter 10 — MCP](../10_mcp/)
