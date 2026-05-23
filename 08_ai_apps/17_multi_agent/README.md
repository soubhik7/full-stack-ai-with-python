# App 17 — Multi-Agent Systems

> **Pattern:** Multiple specialised AI agents that collaborate, delegate tasks, and produce results no single agent could achieve alone.

---

## Why Multi-Agent?

A single LLM call:
```
User → LLM → Answer
```

A multi-agent system:
```
User → Orchestrator → Research Agent   → findings
                    → Writing Agent    → draft
                    → Review Agent     → critique
                    → Orchestrator     → final answer
```

**Benefits:** Specialisation, parallelism, better quality through review loops, separation of concerns.

---

## Files

| File | What it shows |
|------|--------------|
| `main.py` | Three multi-agent patterns |

---

## Patterns Covered

| Pattern | Description |
|---------|-------------|
| **Orchestrator → Workers** | One agent delegates to specialised subagents |
| **Pipeline** | Output of Agent A feeds Agent B feeds Agent C |
| **Debate (Critic Loop)** | Writer → Critic → Writer (until quality passes) |

---

## Run It

```bash
cd 08_ai_apps/17_multi_agent
python main.py
```

Requires: `OPENAI_API_KEY` in `.env`

---

## Key Concept: Agent Specialisation

```python
# General agent (mediocre at everything)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": "You are a helpful AI."},
              {"role": "user", "content": "Research, write, and review an article."}]
)

# Specialised agents (excellent at their domain)
research = researcher_agent.run("Gather facts about MCP")
draft    = writer_agent.run(f"Write an article using: {research}")
review   = critic_agent.run(f"Critique and improve: {draft}")
```

---

## Previous App

← [16 — Structured Outputs](../16_structured_outputs/)

---

## Real-World Multi-Agent Frameworks

| Framework | Description |
|-----------|-------------|
| **LangGraph** | Graph-based agent orchestration (Chapter 08/05) |
| **AutoGen** | Microsoft's multi-agent conversation framework |
| **CrewAI** | Role-based crew with task delegation |
| **OpenAI Agents SDK** | Handoffs between specialised agents |
