"""Core HippocampAI workflow: remember() to encode, recall() to retrieve.

Adapted from HippocampAI's own examples/01_basic_usage.py. Stores one memory of
each type (fact, preference, goal, event — see 00_concepts/README.md#2-memory-types)
for a single user, then issues three natural-language recall queries and shows how
recall() ranks the matching memory above the unrelated ones.

Requires: Qdrant reachable at QDRANT_URL and an LLM reachable per LLM_PROVIDER/LLM_MODEL
in .env — see ../README.md#prerequisites-to-actually-run-the-scripts.
"""

from hippocampai import MemoryClient

print("=" * 60)
print("  HippocampAI - remember() / recall()")
print("=" * 60)

client = MemoryClient()
user_id = "alice"

print(f"\nEncoding memories for user_id={user_id!r}...")

memory1 = client.remember(
    text="I prefer oat milk in my coffee",
    user_id=user_id,
    type="preference",
    importance=8.0,
)
print(f"  [preference] {memory1.text!r}  (id={memory1.id[:8]}...)")

memory2 = client.remember(
    text="I work as a software engineer at TechCorp",
    user_id=user_id,
    type="fact",
    importance=7.0,
)
print(f"  [fact]       {memory2.text!r}  (id={memory2.id[:8]}...)")

memory3 = client.remember(
    text="I want to learn machine learning this year",
    user_id=user_id,
    type="goal",
    importance=9.0,
)
print(f"  [goal]       {memory3.text!r}  (id={memory3.id[:8]}...)")

memory4 = client.remember(
    text="I went hiking in Yosemite last weekend",
    user_id=user_id,
    type="event",
    importance=6.0,
)
print(f"  [event]      {memory4.text!r}  (id={memory4.id[:8]}...)")

queries = [
    "How does Alice like her coffee?",
    "Where does Alice work?",
    "What are Alice's goals?",
]

for query in queries:
    print(f"\nQuery: {query!r}")
    results = client.recall(query=query, user_id=user_id, k=3)
    for i, result in enumerate(results, 1):
        print(
            f"  {i}. [{result.memory.type.value:<10}] {result.memory.text}"
            f"  (score={result.score:.3f}, importance={result.memory.importance}/10)"
        )

print("\nDone. Notice recall() returns the on-topic memory first for each query —")
print("that ranking is hybrid retrieval, broken down in 03_hybrid_retrieval_breakdown.py.")
