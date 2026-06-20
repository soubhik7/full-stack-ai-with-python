# Lab 01 — The Foundry Workflow

> **Pattern:** Model Catalog → Playground → Agents → Projects → Deployments, walked end-to-end in code.

---

## Why This Workflow?

Study notes §7.2–7.3 describe Azure AI Foundry as five components used in a fixed order — pick a model, test it, configure an agent, organize it under a project, ship it. That's normally a portal tour done by clicking through the Foundry UI. This lab is the same five steps done programmatically, so it's something you run once to see the whole platform end-to-end before going deep on any one piece in later labs.

```
[1] Model Catalog  → confirm which deployment you're using
[2] Playground     → one stateless chat call to test it
[3] Agents         → create_agent + thread + run, one exchange
[4] Projects       → list_agents() — what's registered in this project
[5] Deployments    → the production endpoint already in use
```

---

## Files

| File | What it shows |
|------|--------------|
| `main.py` | All five workflow steps, each as its own function |

---

## Run It

```bash
cd 11_azure_ai_foundry/01_workflow
python main.py
```

Requires: `AZURE_AI_PROJECT_ENDPOINT`, `AZURE_AI_MODEL_DEPLOYMENT` in `.env` (see [00_setup](../00_setup/)).

---

## Previous / Next

← [00 — Setup](../00_setup/) · [02 — Tools](../02_tools/) →
