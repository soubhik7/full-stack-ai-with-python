"""
05_prompt_chaining.py — Sequential Prompt Chains
=================================================
Prompt chaining = breaking a complex task into smaller, focused prompts
where each output feeds the next.

Why chain?
  - Each LLM call is focused on one thing → higher quality
  - You can validate/transform between steps
  - Easier to debug (inspect intermediate results)
  - Can run steps in parallel when independent

Patterns:
  1. Linear chain — A → B → C (sequential)
  2. Fan-out → gather (parallel → merge)
  3. Conditional chain — branch based on output
  4. Iterative refinement — loop until quality threshold

Run: python 06_large_language_models/04_prompt_engineering/05_prompt_chaining.py
"""

import asyncio
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def llm(system: str, user: str, max_tokens: int = 500) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# 1. LINEAR CHAIN — Blog Post Generator
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("1. LINEAR CHAIN — Blog Post Generator")
print("   Topic → Outline → Draft → Polish")
print("="*60)

TOPIC = "Why MCP (Model Context Protocol) will change how we build AI apps"


def generate_blog_post(topic: str) -> dict:
    print(f"\n📌 Topic: {topic}")

    # Step 1: Generate outline
    print("  Step 1/4: Generating outline...")
    outline = llm(
        system="You are a technical blog strategist. Create focused outlines.",
        user=f"Create a 5-section blog post outline for: '{topic}'. List only section titles and one sentence each.",
        max_tokens=200,
    )

    # Step 2: Write introduction
    print("  Step 2/4: Writing introduction...")
    intro = llm(
        system="You write engaging technical blog introductions. Be compelling, not clickbaity.",
        user=f"Write a 3-paragraph introduction for a blog post titled '{topic}'.\nOutline: {outline}",
        max_tokens=300,
    )

    # Step 3: Write conclusion
    print("  Step 3/4: Writing conclusion...")
    conclusion = llm(
        system="You write memorable blog post conclusions with clear calls to action.",
        user=f"Write a 2-paragraph conclusion for a blog post titled '{topic}'.\nIntroduction written: {intro[:300]}...",
        max_tokens=200,
    )

    # Step 4: Generate SEO metadata
    print("  Step 4/4: Generating SEO metadata...")
    seo = llm(
        system="You are an SEO expert. Return only JSON with: title, meta_description, keywords (array).",
        user=f"Generate SEO metadata for a blog post titled: '{topic}'",
        max_tokens=150,
    )

    return {
        "topic": topic,
        "outline": outline,
        "introduction": intro,
        "conclusion": conclusion,
        "seo_metadata": seo,
    }


blog = generate_blog_post(TOPIC)
print(f"\n📝 Outline:\n{blog['outline']}")
print(f"\n📖 Introduction (first 300 chars):\n{blog['introduction'][:300]}...")
print(f"\n🔍 SEO Metadata:\n{blog['seo_metadata']}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONDITIONAL CHAIN — Route to different handlers
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. CONDITIONAL CHAIN — Route based on classification")
print("="*60)


def classify_and_route(user_message: str) -> str:
    """Route a user message to the appropriate specialized handler."""

    # Step 1: Classify intent
    intent = llm(
        system="Classify user messages into one word: COMPLAINT, QUESTION, PRAISE, or OTHER",
        user=user_message,
        max_tokens=10,
    ).strip().upper()

    print(f"  Intent: {intent}")

    # Step 2: Route to specialized handler
    if "COMPLAINT" in intent:
        return llm(
            system=(
                "You are an empathetic customer support specialist handling complaints. "
                "Always: apologise, acknowledge the issue, offer a concrete solution."
            ),
            user=user_message,
            max_tokens=200,
        )
    elif "QUESTION" in intent:
        return llm(
            system=(
                "You are a knowledgeable product expert. "
                "Give clear, accurate answers with relevant examples."
            ),
            user=user_message,
            max_tokens=200,
        )
    elif "PRAISE" in intent:
        return llm(
            system=(
                "You are a warm customer success manager. "
                "Thank the customer genuinely and suggest they leave a review."
            ),
            user=user_message,
            max_tokens=150,
        )
    else:
        return llm(
            system="You are a helpful general assistant.",
            user=user_message,
            max_tokens=200,
        )


test_messages = [
    "Your delivery was 5 days late and the item arrived damaged!",
    "Can this product work with both iOS and Android?",
    "I absolutely love your service, best decision I've made!",
]

for msg in test_messages:
    print(f"\n👤 User: {msg}")
    response = classify_and_route(msg)
    print(f"🤖 Response: {response[:200]}...")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ITERATIVE REFINEMENT — Keep improving until quality threshold
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. ITERATIVE REFINEMENT — Improve until quality passes")
print("="*60)


def score_writing(text: str, criteria: str) -> int:
    """Ask the LLM to score a piece of writing 1-10."""
    response = llm(
        system="You score writing quality. Return only a number 1-10. Nothing else.",
        user=f"Score this text (1-10) based on: {criteria}\n\nText: {text}",
        max_tokens=5,
    )
    try:
        return int("".join(c for c in response if c.isdigit())[:2])
    except ValueError:
        return 5


def refine_until_good(initial_text: str, quality_threshold: int = 7, max_rounds: int = 3) -> str:
    """Iteratively improve text until it scores above the threshold."""
    current = initial_text
    for round_num in range(1, max_rounds + 1):
        score = score_writing(current, "clarity, conciseness, and professional tone")
        print(f"  Round {round_num}: score = {score}/10")

        if score >= quality_threshold:
            print(f"  ✅ Quality threshold ({quality_threshold}) reached!")
            break

        # Improve the text
        current = llm(
            system="You are a professional editor. Improve writing for clarity, conciseness, and tone.",
            user=f"Improve this text (current score: {score}/10):\n\n{current}\n\nMake it clearer and more professional.",
            max_tokens=300,
        )
    return current


ROUGH_DRAFT = (
    "so basically the thing is that machine learning is like when computers "
    "learn stuff from data and stuff. its kinda like when you teach a dog tricks "
    "but instead its a computer and the tricks are like predictions or whatever. "
    "people use it for lots of things."
)

print(f"\n📝 Rough draft:\n{ROUGH_DRAFT}")
print("\n🔄 Refining...")
polished = refine_until_good(ROUGH_DRAFT)
print(f"\n✨ Final polished text:\n{polished}")


print("\n\n💡 Prompt Chaining Best Practices:")
print("  ✓ Make each step focused on ONE thing")
print("  ✓ Validate/transform output between steps")
print("  ✓ Add scores or classifiers to decide branches")
print("  ✓ Cap iterations to avoid infinite loops")
print("  ✓ Use parallel chains for independent sub-tasks")
