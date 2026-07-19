"""Extracting memories from raw conversation text instead of writing remember() calls by hand.

Adapted from HippocampAI's own examples/02_conversation_extraction.py.
extract_from_conversation() runs an LLM pass over a chat transcript, pulls out the
facts/preferences/goals worth remembering, classifies each, and stores them —
this is what a chat agent would call after every turn instead of deciding by hand
what's worth persisting.

Requires: Qdrant reachable at QDRANT_URL and an LLM reachable per LLM_PROVIDER/LLM_MODEL
in .env — see ../README.md#prerequisites-to-actually-run-the-scripts.
"""

from hippocampai import MemoryClient

print("=" * 60)
print("  HippocampAI - extract_from_conversation()")
print("=" * 60)

client = MemoryClient()
user_id = "bob"
session_id = "session_001"

conversations = [
    """
    User: I really enjoy drinking green tea in the morning.
    Assistant: That's great! Green tea is healthy.
    User: Yes, and I usually have it without sugar.
    """,
    """
    User: I work as a data scientist.
    Assistant: Interesting! What kind of projects do you work on?
    User: Mostly machine learning and predictive analytics.
    """,
    """
    User: I want to run a marathon next year.
    Assistant: That's an ambitious goal!
    User: I know, I need to start training soon.
    """,
]

print(f"\nExtracting memories from {len(conversations)} conversation turns for user_id={user_id!r}...")

for i, conversation in enumerate(conversations, 1):
    memories = client.extract_from_conversation(
        conversation=conversation, user_id=user_id, session_id=session_id
    )
    print(f"\nConversation {i} -> {len(memories)} memories extracted:")
    for mem in memories:
        print(f"  [{mem.type.value}] {mem.text}  (importance={mem.importance}/10)")

queries = [
    "What does Bob like to drink?",
    "What is Bob's profession?",
    "What are Bob's goals?",
]

print("\nRecalling what was extracted:")
for query in queries:
    print(f"\nQuery: {query!r}")
    results = client.recall(query=query, user_id=user_id, k=2)
    if results:
        for result in results:
            print(f"  -> {result.memory.text}  (score={result.score:.3f})")
    else:
        print("  -> no results")
