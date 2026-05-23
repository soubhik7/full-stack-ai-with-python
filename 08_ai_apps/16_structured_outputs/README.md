# App 16 — Structured Outputs (Production Patterns)

> **Pattern:** LLM output that your code can reliably parse and process — using Pydantic and OpenAI's structured output mode.

---

## The Problem

Raw LLM output is a string. Your code needs data:

```python
# Unreliable:
response = "The sentiment is POSITIVE with 85% confidence."
# How do you extract "POSITIVE" and 0.85 reliably?

# Reliable:
result = SentimentResult(sentiment="POSITIVE", confidence=0.85)
# Just use result.sentiment and result.confidence.
```

Structured outputs solve this with **type-safe, validated LLM responses**.

---

## Files

| File | What it shows |
|------|--------------|
| `main.py` | 6 production patterns for structured LLM output |

---

## Patterns Covered

| Pattern | Use case |
|---------|---------|
| Pydantic `.parse()` | Type-safe extraction with validation |
| JSON mode | Guaranteed valid JSON, flexible schema |
| Enum fields | Constrained categorical outputs |
| Nested models | Complex hierarchical data |
| List extraction | Extract arrays from unstructured text |
| Pipeline pattern | Chain structured outputs through a workflow |

---

## Run It

```bash
cd 08_ai_apps/16_structured_outputs
python main.py
```

Requires: `OPENAI_API_KEY` in `.env`

---

## Key API

```python
from pydantic import BaseModel

class MyOutput(BaseModel):
    field: str
    score: float
    tags: list[str]

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[...],
    response_format=MyOutput,  # ← Pydantic model here
)

result: MyOutput = response.choices[0].message.parsed
# result.field, result.score, result.tags are all typed ✓
```

---

## Previous App

← [15 — Streaming](../15_streaming/)

## Next App

→ [17 — Multi-Agent](../17_multi_agent/)
