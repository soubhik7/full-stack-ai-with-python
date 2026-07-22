# 01 — Core Client Workflow

Three scripts against the real `hippocampai` library, each adapted from the project's own
`examples/` (which the upstream repo runs in CI). Run in order — each builds on the retrieval
concepts from the previous one:

| Script | Demonstrates |
|---|---|
| `01_basic_remember_recall.py` | Core `remember()` / `recall()` loop; memory types and importance |
| `02_conversation_extraction.py` | `extract_from_conversation()` — LLM-driven extraction instead of hand-written `remember()` calls |
| `03_hybrid_retrieval_breakdown.py` | The `breakdown` dict behind `recall()`'s ranking (sim / rerank / recency / importance) |

## Setup

From `15_hippocampus_ai/` (not the repo root — see the parent README for why this chapter has its
own venv):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip3 install -r ../requirements.txt
```

Create `15_hippocampus_ai/.env`:

```bash
QDRANT_URL=http://localhost:6333
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b-instruct
LLM_BASE_URL=http://localhost:11434
```

Or, to reuse this repo's existing Groq key instead of running Ollama locally:

```bash
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=...          # same value as the root .env
ALLOW_CLOUD=true
```

Start Qdrant (one-time, keep it running in the background):

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Run

```bash
python 01_basic_remember_recall.py
python 02_conversation_extraction.py
python 03_hybrid_retrieval_breakdown.py
```

Each script prints what it stores and what it recalls — no assertions, this is meant to be read
alongside the terminal output. Memories persist in Qdrant across runs (per `user_id`), so re-running
a script adds duplicates rather than starting fresh; delete the Qdrant container's volume (or the
`hippocampai_facts` / `hippocampai_prefs` collections) to reset.
