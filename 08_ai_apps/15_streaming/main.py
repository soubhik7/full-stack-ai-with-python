"""
App 15 — Streaming AI Responses
================================
Demonstrates all major streaming patterns with the OpenAI API.
Streaming is essential for chat apps — users see tokens as they arrive.

Patterns:
  1. Basic streaming  — simplest usage
  2. Collect while streaming — build the full string + print live
  3. Stream with callbacks  — pluggable event handlers
  4. Streaming to file      — write long content to disk live
  5. Parallel streaming     — generate multiple responses concurrently
  6. Streaming with early stop — halt when a condition is met

Run: python 08_ai_apps/15_streaming/main.py
Requires: OPENAI_API_KEY in .env
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()

client = OpenAI()
async_client = AsyncOpenAI()


# ══════════════════════════════════════════════════════════════════════════════
# 1. BASIC STREAMING
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("1. BASIC STREAMING — tokens appear immediately")
print("="*60)

print("\n🤖 Response (streaming):\n")

start = time.time()
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Write a 3-sentence explanation of what streaming is in the context of AI APIs."}
    ],
    stream=True,
    max_tokens=120,
)

for chunk in stream:
    token = chunk.choices[0].delta.content or ""
    print(token, end="", flush=True)

elapsed = time.time() - start
print(f"\n\n⏱️  Completed in {elapsed:.1f}s (vs ~5s wait without streaming)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. COLLECT WHILE STREAMING (get full string + live display)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. COLLECT WHILE STREAMING — accumulate the full response")
print("="*60)


def stream_and_collect(messages: list[dict], model: str = "gpt-4o-mini") -> str:
    """
    Stream a response while collecting all tokens into a string.

    Returns:
        The complete response as a string (available after streaming finishes).
    """
    full_text = []
    print("\n🤖 ", end="", flush=True)

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=200,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        full_text.append(token)
        print(token, end="", flush=True)

    print()  # newline after response
    return "".join(full_text)


result = stream_and_collect([
    {"role": "user", "content": "Name 3 key differences between RAG and fine-tuning."}
])
print(f"\n📊 Total characters collected: {len(result)}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. STREAMING WITH CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. CALLBACKS — pluggable event handlers during streaming")
print("="*60)


def stream_with_callbacks(
    messages: list[dict],
    on_token: Optional[Callable[[str], None]] = None,
    on_complete: Optional[Callable[[str, dict], None]] = None,
    model: str = "gpt-4o-mini",
    max_tokens: int = 200,
) -> str:
    """
    Stream with pluggable callbacks for each token and completion.

    Args:
        messages: Chat messages.
        on_token: Called for each token. e.g. lambda t: print(t, end="")
        on_complete: Called when done. Args: full_text, usage_stats.
        model: OpenAI model to use.
        max_tokens: Max tokens to generate.
    """
    tokens = []
    start_time = time.time()

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=max_tokens,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            tokens.append(token)
            if on_token:
                on_token(token)

    full_text = "".join(tokens)
    usage_stats = {
        "elapsed_ms": int((time.time() - start_time) * 1000),
        "char_count": len(full_text),
        "word_count": len(full_text.split()),
    }

    if on_complete:
        on_complete(full_text, usage_stats)

    return full_text


# Demo: different callbacks for different use cases
token_count = [0]

def counting_token_handler(token: str):
    token_count[0] += 1
    print(token, end="", flush=True)

def completion_logger(full_text: str, stats: dict):
    print(f"\n\n📊 Stats: {stats['word_count']} words in {stats['elapsed_ms']}ms")

print("\n🤖 Response (with counting callback):\n")
stream_with_callbacks(
    [{"role": "user", "content": "What are the main components of a transformer architecture?"}],
    on_token=counting_token_handler,
    on_complete=completion_logger,
)


# ══════════════════════════════════════════════════════════════════════════════
# 4. STREAMING TO FILE (for long content generation)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. STREAMING TO FILE — write long content live to disk")
print("="*60)

output_path = Path(__file__).parent / "streamed_output.txt"

print(f"\n📁 Writing to: {output_path.name}")
print("🤖 Content preview:\n")

with open(output_path, "w", encoding="utf-8") as f:
    f.write("# AI Learning Path: Key Topics\n\n")

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                "List 5 key topics in AI/ML that every developer should learn, "
                "with a 2-sentence description of each. Format as a numbered list."
            )
        }],
        stream=True,
        max_tokens=400,
    )

    char_count = 0
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        f.write(token)
        f.flush()  # write to disk immediately (not buffered)
        char_count += len(token)
        print(token, end="", flush=True)

print(f"\n\n✅ Wrote {char_count} characters to {output_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PARALLEL STREAMING (async — multiple requests simultaneously)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. PARALLEL STREAMING — generate multiple responses concurrently")
print("="*60)


async def async_stream(prompt: str, label: str) -> str:
    """Stream one response asynchronously."""
    tokens = []
    print(f"\n🔀 [{label}] starting...")

    stream = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=80,
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            tokens.append(chunk.choices[0].delta.content)

    result = "".join(tokens)
    print(f"  [{label}] done: {result[:60]}...")
    return result


async def parallel_stream_demo():
    prompts = [
        ("Explain BERT in one sentence.", "BERT"),
        ("Explain GPT in one sentence.", "GPT"),
        ("Explain RAG in one sentence.", "RAG"),
    ]

    start = time.time()
    results = await asyncio.gather(*[async_stream(p, l) for p, l in prompts])
    elapsed = time.time() - start

    print(f"\n⏱️  All 3 responses in {elapsed:.1f}s (would take ~3x longer sequentially)")
    return results


responses = asyncio.run(parallel_stream_demo())
print(f"\n📊 Collected {len(responses)} responses in parallel")


# ══════════════════════════════════════════════════════════════════════════════
# 6. STREAMING + EARLY STOP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("6. EARLY STOP — halt streaming when a condition is met")
print("="*60)


def stream_until(
    messages: list[dict],
    stop_phrase: str = "DONE",
    max_tokens: int = 500,
) -> str:
    """Stream tokens and stop when a specific phrase appears."""
    tokens = []
    print(f"\n🤖 Streaming until '{stop_phrase}' appears:\n")

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
        max_tokens=max_tokens,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        tokens.append(token)
        print(token, end="", flush=True)

        current = "".join(tokens)
        if stop_phrase.lower() in current.lower():
            print(f"\n\n⛔ Stop phrase '{stop_phrase}' detected — halting stream.")
            stream.close()
            break

    return "".join(tokens)


result = stream_until(
    messages=[{
        "role": "user",
        "content": (
            "List AI tasks that are easy: 1. Text classification. 2. Translation. "
            "3. Summarisation. DONE. Then list hard tasks (don't stop early)."
        )
    }],
    stop_phrase="DONE",
)
print(f"\n📊 Collected before stop: {len(result)} chars")

print("\n\n💡 Summary of Streaming Patterns:")
print("  Basic streaming         → client.chat.completions.create(stream=True)")
print("  Collect while streaming → accumulate chunks into a list, join at end")
print("  Callbacks               → pass on_token / on_complete for flexibility")
print("  Stream to file          → f.write(token) + f.flush() on each chunk")
print("  Parallel (async)        → asyncio.gather(*[stream1, stream2, ...])")
print("  Early stop              → stream.close() when condition is met")
