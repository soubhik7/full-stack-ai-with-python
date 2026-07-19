# 15 — HippocampAI: Long-Term Memory for AI Agents

Every chatbot in `08_ai_apps/` and `10_mcp/` forgets everything the moment a session ends. This
chapter studies [HippocampAI](https://github.com/rexdivakar/HippocampAI), an open-source
"memory engine" that gives an LLM agent persistent, queryable, long-term memory — modeled on how
the brain's hippocampus encodes, consolidates, and recalls experience.

This is a study of a real, external open-source project (Apache 2.0), not an original curriculum
build. Source: `/Users/soubhik/hippocamp-ai/HippocampAI` (`pip install hippocampai`, PyPI package
`hippocampai==0.5.1`).

| Sub-chapter | Content |
|-------------|---------|
| [`00_concepts/`](00_concepts/README.md) | The hippocampus analogy, memory types, architecture, hybrid retrieval scoring, sleep-phase consolidation |
| [`01_core_client/`](01_core_client/README.md) | Three runnable scripts against the real `hippocampai` library: remember/recall, conversation extraction, hybrid-retrieval score breakdown |

## Why an isolated environment

`hippocampai` pins `sentence-transformers>=2.2,<3.0`. This repo's root `requirements.txt` runs a
newer `sentence-transformers` (5.x) for chapters 04–09. Installing `hippocampai` into the shared
`venv/` silently downgrades that pin repo-wide and breaks other chapters. So, same as
`09_projects/04_made_with_ml`, this chapter gets its **own virtual environment** — do not add
`hippocampai` to the root `requirements.txt`.

```bash
cd 15_hippocampus_ai
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prerequisites to actually run the scripts

`hippocampai`'s core library talks directly to a vector store and an LLM — there's no in-memory
mode. You need:

1. **Qdrant** (vector store) running locally:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
2. **An LLM provider** for memory-type classification and fact extraction. Cheapest local option —
   Ollama, already used elsewhere in this repo (see `00_setup/README.md`):
   ```bash
   ollama serve
   ollama pull qwen2.5:7b-instruct
   ```
   Or set `LLM_PROVIDER=groq` / `openai` + the matching API key and `ALLOW_CLOUD=true` in `.env`
   (this repo's root `.env` already has `GROQ_API_KEY`).
3. A `.env` file inside `15_hippocampus_ai/` (see `01_core_client/README.md` for the minimal set
   of variables — `QDRANT_URL`, `LLM_PROVIDER`, `LLM_MODEL`).

Without Qdrant + an LLM reachable, `MemoryClient()` will raise a connection error on the first
`remember()` call — that's expected, not a bug in the scripts.

## Key idea in one paragraph

HippocampAI stores every fact/preference/goal/habit/event as a `Memory` object with an embedding,
an importance score, and a half-life. `remember()` classifies and embeds text into Qdrant;
`recall()` fuses vector similarity, BM25 keyword search, cross-encoder reranking, recency decay,
and importance into one ranked list — the same hybrid-retrieval pattern taught in
[`07_rag/`](../07_rag/), but wrapped around a persistent, per-user memory store instead of a
static document corpus. A background "sleep phase" job periodically replays and consolidates
recent memories the way hippocampal replay consolidates episodic memory into long-term memory
during sleep — see [`00_concepts/README.md`](00_concepts/README.md) for the full mapping.
