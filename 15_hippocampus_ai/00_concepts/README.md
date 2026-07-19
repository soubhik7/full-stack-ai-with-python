# The Hippocampus Concept

## 1. The biological metaphor

The hippocampus is the brain region that turns short-lived experience into durable memory. Three
things it does, and how HippocampAI names its software equivalent:

| Hippocampus | HippocampAI |
|---|---|
| **Encoding** — a new experience is converted into a neural pattern | `client.remember(text, user_id)` — text is classified into a memory type and embedded into a vector |
| **Retrieval** — a cue (smell, question, image) reactivates the stored pattern | `client.recall(query, user_id)` — a query is embedded and matched against stored memories |
| **Consolidation (sleep replay)** — during sleep, the hippocampus replays the day's experience and transfers stable, important patterns to the cortex for long-term storage, while noise is discarded | The **Sleep Phase** background job: nightly, it clusters the day's memories, has an LLM decide what's promotion-worthy vs. low-value, merges duplicates into synthesized long-term memories, and prunes the rest |

This is why the core API reads like cognition, not CRUD:

```python
from hippocampai import MemoryClient

client = MemoryClient()
client.remember("I prefer oat milk in my coffee", user_id="alice", type="preference")
results = client.recall("coffee preferences", user_id="alice")
```

`remember()` / `recall()` are not naming flourish — they map onto a real architectural split
between a **write path** (classify → embed → store) and a **read path** (embed query → hybrid
search → rank), same as encoding and retrieval are functionally distinct processes in the brain.

## 2. Memory types

HippocampAI classifies every stored memory into one of six types, each with its own decay
half-life (`docs/CONFIGURATION.md` in the source repo — half-lives are how fast an unused memory's
importance score decays):

| Type | Example | Default half-life |
|---|---|---|
| `fact` | "I work as a software engineer at TechCorp" | 30 days |
| `preference` | "I prefer oat milk in my coffee" | 90 days |
| `goal` | "I want to learn machine learning this year" | 90 days |
| `habit` | "I usually work remotely on Tuesdays" | 90 days |
| `event` | "I went hiking in Yosemite last weekend" | 14 days |
| `context` | conversation/session-level scratch memory | 30 days |

Facts and events decay fast (they go stale); preferences, goals, and habits decay slowly (they
describe the user, not a moment). If `type` isn't passed to `remember()`, an LLM call classifies
the text automatically.

## 3. Architecture — one codebase, two interfaces

```
                Application Layer (your code)
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
  MemoryClient (Python import)      FastAPI server (HTTP)
        │            "Mode 1"        "Mode 2"      │
        └─────────────────┬─────────────────┘
                          ▼
              MemoryManagementService   ← shared business logic
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   Pipeline layer    Retrieval layer    Storage layer
   (fact/entity       (hybrid search,    (Qdrant vectors,
    extraction)         reranking)        Redis cache)
```

Both the Python library and the REST API call the *same* `MemoryManagementService` — there's no
"the API is the real product, the library is a thin wrapper" split. This is the same
library-vs-server duality this repo already teaches in `10_mcp/03_mcp_client/` (an MCP server has
one implementation, multiple transports); HippocampAI applies it to a memory store instead of a
tool server.

## 4. Hybrid retrieval — how `recall()` actually ranks results

This is the same fusion-of-signals idea taught in `07_rag/`'s hybrid retrieval approach, applied
per-user instead of over a static document set. Every candidate memory gets a weighted score:

```
final_score = 0.55 · vector_similarity
            + 0.20 · cross_encoder_rerank
            + 0.15 · recency_decay
            + 0.10 · importance
          (+ 0.15 · knowledge_graph_score   — when graph retrieval is enabled)
          (+ 0.10 · relevance_feedback      — when a user has rated past recalls)
```

Pipeline for a `recall("coffee preferences", user_id="alice")` call:

1. Embed the query (`BAAI/bge-small-en-v1.5` by default) and pull the top ~200 candidates from
   Qdrant by cosine similarity.
2. Run BM25 keyword scoring over the same candidate set — catches exact-term matches
   (`"oat milk"`) that embeddings alone can under-rank.
3. Fuse vector rank and BM25 rank with Reciprocal Rank Fusion (RRF).
4. Re-score the top candidates with a cross-encoder (`ms-marco-MiniLM-L-6-v2`) for
   query-memory pairwise relevance — much more precise than embedding similarity alone.
5. Blend in recency decay (`exp(-Δt / half_life)`) and the memory's importance score.
6. Return the top `k`, each carrying a `breakdown` dict of the individual signal scores (see
   `01_core_client/03_hybrid_retrieval_breakdown.py`).

## 5. Consolidation — the "sleep phase"

`AUTO_CONSOLIDATION_ENABLED=true` schedules a nightly Celery job that mirrors hippocampal replay:

```
collect_recent_memories(user_id, lookback_hours=24)
        │  Postgres query for the last day's event/context memories
        ▼
cluster_memories(memories)
        │  group by session_id, then by semantic/temporal proximity
        ▼
llm_review_cluster(cluster)
        │  LLM reads a cluster and returns: promoted_facts, low_value_ids,
        │  updated_memories, synthetic_memories (a summary memory)
        ▼
apply_consolidation_decisions(review_result)
        │  delete low-value, promote/update important ones,
        │  write the synthesized summary memory
```

This is the direct computational analogue of what neuroscience calls "systems consolidation":
today's noisy, redundant episodic memories get replayed, compressed, and the stable signal gets
promoted into a durable form, while the redundant raw detail is discarded. Off by default in
production because it costs one LLM call per cluster per user per night.

## 6. Three API styles, one backend

HippocampAI exposes progressively more power behind the same storage:

```python
# Beginner — SimpleMemory: minimal surface for quick prototyping
from hippocampai import SimpleMemory as Memory
m = Memory()
m.add("I prefer dark mode", user_id="alice")

# Intermediate — SimpleSession: conversation-shaped wrapper
from hippocampai import SimpleSession as Session
session = Session(session_id="chat_123")
session.add_message("user", "Hello!")

# Advanced — MemoryClient: full API (102+ methods — patterns, conflicts, graph, temporal…)
from hippocampai import MemoryClient
client = MemoryClient()
patterns = client.detect_patterns(user_id="alice")
```

All three sit on the same `MemoryManagementService`; switching from `SimpleMemory` to
`MemoryClient` later doesn't require a data migration.

## 7. Local vs. remote — same code, different backend

```python
# Local mode — direct calls to Qdrant/Redis/LLM in-process (5-15ms)
client = MemoryClient(mode="local")

# Remote mode — same methods, HTTP calls to a running HippocampAI API server (20-50ms)
client = MemoryClient(mode="remote", api_url="http://localhost:8000")
```

Useful default: build and test against local mode (what `01_core_client/` uses), then point the
same call sites at a shared remote HippocampAI deployment for a multi-service agent system,
without touching call sites.

## Further reading (in the source repo)

- `docs/ARCHITECTURE.md` — full component and deployment diagrams
- `docs/SLEEP_PHASE_ARCHITECTURE.md` — consolidation job internals
- `docs/FEATURES.md` — knowledge graph, relevance feedback, procedural memory, bi-temporal facts
- `docs/API_REFERENCE.md` — all 102+ `MemoryClient` methods
