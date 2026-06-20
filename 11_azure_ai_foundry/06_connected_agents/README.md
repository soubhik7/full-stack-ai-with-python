# Lab 06 — Connected Agents

> **Pattern:** Multi-agent orchestration — a parent agent delegating to focused sub-agents via `ConnectedAgentTool`.

---

## Why Split One Agent Into Several?

A single agent with a huge instruction prompt trying to handle availability, pricing, and conversation all at once is hard to maintain. Splitting into focused sub-agents — each with a narrow job and short instructions — and letting a parent agent delegate to them mirrors the real Crystal Hotels pipeline from the study notes (§61, Figure 61.2): booking validation → availability check → payment processing → confirmation. This lab implements the middle two steps.

This is a different, Foundry-specific flavor of multi-agent orchestration from the generic patterns in [08_ai_apps/17_multi_agent](../../08_ai_apps/17_multi_agent/) — here, "connected" is just a tool type wrapping an existing hosted agent.

---

## Files

| File | What it shows |
|------|--------------|
| `main.py` | Two sub-agents (`availability-checker`, `payment-calculator`) wrapped as tools on a `concierge` agent |

---

## Run It

```bash
cd 11_azure_ai_foundry/06_connected_agents
python main.py
```

---

## Key Concept: An Agent as a Tool

```python
availability_tool = ConnectedAgentTool(
    id=availability_agent.id,
    name="check_availability",
    description="Check whether a room type is available at Crystal Hotels.",
)
concierge = agents_client.create_agent(..., tools=availability_tool.definitions + payment_tool.definitions)
```

The `concierge` agent never sees the sub-agents' instructions — only the tool names and descriptions. Azure routes each call to the right sub-agent automatically.

---

## Previous / Next

← [05 — Hosted Agent](../05_hosted_agent/) · [07 — Knowledge RAG](../07_knowledge_rag/) →
