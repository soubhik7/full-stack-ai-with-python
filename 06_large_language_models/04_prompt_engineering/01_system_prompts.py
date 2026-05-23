"""
01_system_prompts.py — System Prompt Design
============================================
The system prompt is the most powerful lever in prompt engineering.
It defines:
  - Who the LLM is (persona)
  - What it can and cannot do (constraints)
  - How it should format its responses
  - What tone/style to use

This script demonstrates 5 system prompt patterns with the same user query.

Run: python 06_large_language_models/04_prompt_engineering/01_system_prompts.py
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

USER_QUERY = "How do neural networks learn?"


def ask(system: str, user: str, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  PATTERN: {label}")
    print(f"{'='*60}")
    print(f"📋 System: {system[:120]}...")
    print(f"👤 User: {user}")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=300,
    )
    print(f"\n🤖 Response:\n{response.choices[0].message.content}")


# ── Pattern 1: Basic (no system prompt) ──────────────────────────────────────
print("\n{'='*60}")
print("  PATTERN: No System Prompt (baseline)")
print("{'='*60}")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": USER_QUERY}],
    max_tokens=200,
)
print(f"👤 User: {USER_QUERY}")
print(f"\n🤖 Response:\n{resp.choices[0].message.content}")


# ── Pattern 2: Persona + Audience ─────────────────────────────────────────────
ask(
    system=(
        "You are an enthusiastic AI teacher explaining concepts to a 12-year-old. "
        "Use simple words, relatable analogies (like sports, games, cooking), "
        "and lots of encouragement. Keep responses under 150 words."
    ),
    user=USER_QUERY,
    label="Persona + Target Audience",
)


# ── Pattern 3: Expert + Format Constraint ────────────────────────────────────
ask(
    system=(
        "You are a senior machine learning researcher with 15+ years of experience. "
        "Respond in bullet points. "
        "Always include: (1) the intuition, (2) the math, (3) a practical implication. "
        "Use technical vocabulary without over-explaining basics."
    ),
    user=USER_QUERY,
    label="Expert Persona + Structured Format",
)


# ── Pattern 4: Constraint-Heavy ───────────────────────────────────────────────
ask(
    system=(
        "You are a concise AI assistant. Rules you MUST follow:\n"
        "1. Maximum 3 sentences in your answer.\n"
        "2. No jargon — use only words a 10-year-old knows.\n"
        "3. End every response with a relevant emoji.\n"
        "4. Never say 'I' — refer to yourself as 'This assistant'."
    ),
    user=USER_QUERY,
    label="Hard Constraints (rules-based)",
)


# ── Pattern 5: Role + Context + Output Format ────────────────────────────────
ask(
    system=(
        "You are an AI curriculum designer building a 5-day bootcamp on deep learning. "
        "Context: The student is a Python developer with no ML background. "
        "When answering questions, structure your response as:\n"
        "  DAY: Which bootcamp day this concept fits into (Day 1–5)\n"
        "  CONCEPT: 2-sentence plain-English explanation\n"
        "  CODE PREVIEW: One line of pseudocode illustrating the idea\n"
        "  NEXT STEP: What concept to learn after this one"
    ),
    user=USER_QUERY,
    label="Role + Context + Structured Output Template",
)


# ── Pattern 6: Guard-rails (refuse out-of-scope) ─────────────────────────────
ask(
    system=(
        "You are a customer support bot for 'LearnAI Academy', an online education platform. "
        "You ONLY answer questions about:\n"
        "  - Course content and curricula\n"
        "  - Pricing and enrollment\n"
        "  - Technical issues with the platform\n"
        "For ANY other topic, politely say: "
        "'That's outside my area of expertise. Please contact our general support team.'\n"
        "Never answer general knowledge, politics, coding help, or personal advice questions."
    ),
    user="How do neural networks learn, and also can you write me a poem?",
    label="Guard-rails (domain restriction)",
)


print("\n\n💡 Key Takeaways:")
print("  1. Persona → shapes tone and vocabulary")
print("  2. Audience → calibrates complexity")
print("  3. Format constraints → structure the output")
print("  4. Hard rules → enforce non-negotiable behaviours")
print("  5. Guard-rails → prevent off-topic responses")
print("\nNext: 02_advanced_few_shot.py — when and how to add examples")
