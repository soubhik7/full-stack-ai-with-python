"""
03_chain_of_thought.py — Chain-of-Thought & Self-Consistency
=============================================================
Chain-of-Thought (CoT) prompting forces the LLM to "think out loud"
before giving an answer — dramatically improving accuracy on reasoning tasks.

Topics covered:
  1. Standard CoT ("Let's think step by step")
  2. Zero-shot CoT (just add the magic phrase)
  3. Structured CoT (explicit reasoning format)
  4. Self-consistency (sample N answers, take the majority vote)
  5. Tree of Thoughts (explore multiple reasoning branches)

Run: python 06_large_language_models/04_prompt_engineering/03_chain_of_thought.py
"""

import json
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def ask(prompt: str, max_tokens: int = 400, temperature: float = 0.0) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ── 1. No CoT (baseline) ──────────────────────────────────────────────────────
print("="*60)
print("1. NO CoT — Direct answer (baseline)")
print("="*60)

HARD_PROBLEM = (
    "A farmer has chickens and cows. He counts 50 heads and 140 legs. "
    "How many chickens and how many cows does he have?"
)

direct = ask(f"Answer directly: {HARD_PROBLEM}")
print(f"❓ Problem: {HARD_PROBLEM}")
print(f"🤖 Direct answer: {direct}")


# ── 2. Zero-Shot CoT ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. ZERO-SHOT CoT — Magic phrase: 'Let's think step by step'")
print("="*60)

zero_shot_cot = f"{HARD_PROBLEM}\n\nLet's think step by step."
result = ask(zero_shot_cot)
print(f"🧠 With 'Let's think step by step':\n{result}")


# ── 3. Structured CoT ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. STRUCTURED CoT — Explicit reasoning format")
print("="*60)

STRUCTURED_SYSTEM = """Solve problems using this exact format:
UNDERSTAND: Restate the problem in your own words.
IDENTIFY: List all given information as variables.
PLAN: Describe your solving approach in 1-2 sentences.
SOLVE: Show every calculation step.
CHECK: Verify your answer makes sense.
ANSWER: State the final answer clearly."""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": STRUCTURED_SYSTEM},
        {"role": "user", "content": HARD_PROBLEM},
    ],
    max_tokens=500,
)
print(f"🧠 Structured CoT:\n{response.choices[0].message.content}")


# ── 4. Self-Consistency ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. SELF-CONSISTENCY — Sample 5 answers, take majority vote")
print("="*60)

LOGIC_PROBLEM = (
    "All programmers drink coffee. Sam drinks coffee. "
    "Is Sam definitely a programmer? Answer YES or NO and explain why."
)

print(f"❓ Logic problem: {LOGIC_PROBLEM}\n")

answers = []
for i in range(5):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"{LOGIC_PROBLEM}\n\nThink step by step, then give a one-word answer: YES or NO"},
        ],
        max_tokens=200,
        temperature=0.7,  # Higher temp → more diverse reasoning paths
    )
    text = response.choices[0].message.content
    # Extract YES or NO
    answer = "YES" if "YES" in text.upper()[:50] or text.upper().endswith("YES") else "NO"
    answers.append(answer)
    print(f"  Sample {i+1}: {answer}")

vote = Counter(answers).most_common(1)[0]
print(f"\n🗳️  Majority vote: {vote[0]} ({vote[1]}/5 samples agree)")
print("📌 Self-consistency reduces hallucination on ambiguous reasoning tasks.")


# ── 5. Step-Back Prompting ────────────────────────────────────────────────────
print("\n" + "="*60)
print("5. STEP-BACK PROMPTING — Abstract before solving")
print("="*60)

SPECIFIC_QUESTION = (
    "If I drop a 1 kg steel ball and a 1 kg feather simultaneously "
    "from the top of the Eiffel Tower, which hits the ground first?"
)

# Step 1: Get the principle
step_back = ask(
    f"What is the general physics principle that governs what happens "
    f"when objects of different shapes are dropped in an atmosphere?",
    max_tokens=150,
)
print(f"📚 Step-back (abstract principle):\n{step_back}\n")

# Step 2: Apply the principle
final = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What is the general physics principle that governs falling objects in an atmosphere?"},
        {"role": "assistant", "content": step_back},
        {"role": "user", "content": SPECIFIC_QUESTION},
    ],
    max_tokens=200,
)
print(f"❓ Specific question: {SPECIFIC_QUESTION}")
print(f"🤖 Answer (using principle):\n{final.choices[0].message.content}")


# ── 6. ReAct Pattern (Reason + Act) ─────────────────────────────────────────
print("\n" + "="*60)
print("6. ReAct PATTERN — Interleave Thought, Action, Observation")
print("="*60)

REACT_SYSTEM = """You solve problems using the ReAct pattern. Alternate between:
Thought: [what you're thinking]
Action: [what you would do or look up]
Observation: [what you found or calculated]
...repeat until you reach an answer...
Final Answer: [your conclusion]"""

react_problem = (
    "I need to plan a trip from Mumbai to Delhi. "
    "The train takes 16 hours and costs ₹1200. "
    "The flight takes 2 hours and costs ₹4500. "
    "My time is worth ₹500/hour. Which option is cheaper overall?"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user", "content": react_problem},
    ],
    max_tokens=400,
)
print(f"❓ Problem: {react_problem}\n")
print(f"🤖 ReAct reasoning:\n{response.choices[0].message.content}")


print("\n\n💡 Summary:")
print("  Zero-shot CoT   → Add 'think step by step' → free accuracy boost")
print("  Structured CoT  → Force a reasoning template → consistent output")
print("  Self-consistency→ Sample multiple answers, take majority → fewer errors")
print("  Step-back       → Abstract first, then apply → better generalisation")
print("  ReAct           → Thought → Action → Observation loop → agentic tasks")
