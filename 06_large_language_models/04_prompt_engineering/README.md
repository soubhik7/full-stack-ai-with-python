# 04 — Advanced Prompt Engineering

> Go beyond zero-shot. Master the full prompt engineering toolkit used by professional AI engineers.

---

## Prerequisites

- [01 Transformers & HuggingFace](../01_transformers_and_huggingface/) *(know what an LLM is)*
- [Chapter 08 / 03 Prompts](../../08_ai_apps/03_prompts/) *(basic zero/few/CoT)*
- `OPENAI_API_KEY` in `.env`

---

## Files

| File | Technique | When to use it |
|------|-----------|---------------|
| `01_system_prompts.py` | System prompt design | Always — sets LLM behaviour |
| `02_advanced_few_shot.py` | Dynamic few-shot selection | When examples improve accuracy |
| `03_chain_of_thought.py` | CoT + Self-consistency | Complex reasoning tasks |
| `04_structured_output.py` | JSON mode + Pydantic | When you need machine-readable output |
| `05_prompt_chaining.py` | Sequential prompt chains | Multi-step pipelines |
| `06_meta_prompting.py` | LLM writes its own prompt | Automated prompt optimisation |

---

## Quick Reference: Which Technique for Which Task?

| Task | Technique |
|------|-----------|
| Simple Q&A | Zero-shot |
| Classification / extraction | Few-shot |
| Math / logic / step-by-step | Chain-of-Thought |
| Consistent persona | System prompt |
| Structured data (JSON) | Structured output |
| Multiple reasoning paths | Self-consistency |
| Multi-step pipelines | Prompt chaining |
| Unknown optimal prompt | Meta-prompting |

---

## Run Any Script

```bash
source venv/bin/activate
python 06_large_language_models/04_prompt_engineering/01_system_prompts.py
```

---

## Key Insight: The Prompt Engineering Stack

```
┌─────────────────────────────────────────────────────┐
│  System Prompt (who the LLM is + rules)             │
├─────────────────────────────────────────────────────┤
│  Few-shot Examples (what good output looks like)    │
├─────────────────────────────────────────────────────┤
│  User Message (the actual request)                  │
├─────────────────────────────────────────────────────┤
│  Output Format (JSON schema, Pydantic model)        │
└─────────────────────────────────────────────────────┘
```

Mastering all four layers = mastering prompt engineering.
