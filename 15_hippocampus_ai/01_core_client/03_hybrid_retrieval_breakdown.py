"""Inspecting the per-signal score breakdown behind recall()'s ranking.

Adapted from HippocampAI's own examples/03_hybrid_retrieval.py. Every RecallResult
carries a `breakdown` dict with the individual vector-similarity, cross-encoder-rerank,
recency, and importance scores that were fused into `result.score` — see
00_concepts/README.md#4-hybrid-retrieval-how-recall-actually-ranks-results for the
weighted formula this breakdown corresponds to.

Requires: Qdrant reachable at QDRANT_URL and an LLM reachable per LLM_PROVIDER/LLM_MODEL
in .env — see ../README.md#prerequisites-to-actually-run-the-scripts.
"""

from hippocampai import MemoryClient

print("=" * 60)
print("  HippocampAI - hybrid retrieval score breakdown")
print("=" * 60)

client = MemoryClient()
user_id = "carol"

memories_data = [
    ("I prefer vegetarian food and avoid meat", "preference", 8.0),
    ("I am allergic to peanuts", "fact", 10.0),
    ("I love Italian cuisine, especially pasta", "preference", 7.0),
    ("I went to a great sushi restaurant last week", "event", 6.0),
    ("I want to learn to cook French food", "goal", 7.5),
    ("I work as a chef at a restaurant", "fact", 8.0),
    ("I enjoy baking desserts on weekends", "habit", 6.5),
    ("I am lactose intolerant", "fact", 9.0),
]

print(f"\nEncoding {len(memories_data)} diverse memories for user_id={user_id!r}...")
for text, mem_type, importance in memories_data:
    client.remember(text=text, user_id=user_id, type=mem_type, importance=importance)
    print(f"  [{mem_type:<10}] {text}")

# BM25 is built per-user from stored memories; rebuild it after a batch of writes
# so keyword matching (not just vector similarity) is included in the fusion.
print("\nRebuilding BM25 index...")
client.retriever.rebuild_bm25(user_id)

queries = [
    "What food does Carol like?",
    "What are Carol's dietary restrictions?",
    "What does Carol do for work?",
]

for query in queries:
    print(f"\nQuery: {query!r}")
    results = client.recall(query=query, user_id=user_id, k=3)
    for i, result in enumerate(results, 1):
        b = result.breakdown
        print(f"  {i}. {result.memory.text}")
        print(
            f"     final={result.score:.3f}  "
            f"sim={b['sim']:.3f}  rerank={b['rerank']:.3f}  "
            f"recency={b['recency']:.3f}  importance={b['importance']:.3f}"
        )

print("\nDone. `sim` alone would often rank these differently — rerank and importance")
print("are what push the dietary-restriction fact above a merely food-related memory.")
