# Lab 07 — Knowledge / RAG Grounding

> **Pattern:** Ground an agent's answers in an uploaded document via file search + a vector store, instead of trusting the model's prior "knowledge".

---

## Why Grounding Matters

A model asked about an exact policy detail (a cancellation window, a fee) will happily produce a plausible-sounding but wrong number. File search fixes this by giving the agent a real document to search before answering.

**A real failure surfaced while building this lab:** with softer instructions ("answer using the knowledge base"), the model skipped calling `file_search` entirely and answered from general assumption — confidently citing a 24-hour cancellation window when the actual policy document says 72 hours. The fix was instructions that leave no discretion: *"You have NO built-in knowledge of hotel policies. You MUST call file_search before answering."* Worth knowing if your own grounded agents seem to "make things up" despite having a knowledge base attached — check whether the tool is actually being called (inspect run steps) before assuming the retrieval itself is broken.

---

## Files

| File | What it shows |
|------|--------------|
| `data/hotel_policy.md` | The knowledge source — cancellation, pet, checkout, and loyalty policies |
| `main.py` | Upload → vector store → `FileSearchTool` → grounded answer |

---

## Run It

```bash
cd 11_azure_ai_foundry/07_knowledge_rag
python main.py
```

---

## Key Concept: Forcing Tool Use

```python
instructions=(
    "You have NO built-in knowledge of hotel policies. For every guest question, you MUST "
    "call the file_search tool to retrieve the relevant policy text before answering."
)
```

---

## Previous Lab

← [06 — Connected Agents](../06_connected_agents/)
