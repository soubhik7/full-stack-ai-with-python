"""
02_advanced_few_shot.py — Advanced Few-Shot Prompting
======================================================
Few-shot prompting = giving the LLM examples before your actual request.
This script covers:
  1. Basic few-shot (static examples)
  2. Format few-shot (show the output structure, not just the answer)
  3. Dynamic few-shot (select the most relevant examples from a pool)
  4. Contrastive few-shot (show what NOT to do alongside what to do)
  5. Chain-of-thought few-shot (show reasoning steps, not just answers)

Run: python 06_large_language_models/04_prompt_engineering/02_advanced_few_shot.py
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def chat(messages: list[dict], max_tokens: int = 300) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ── 1. Basic Few-Shot ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1. BASIC FEW-SHOT — Sentiment Classification")
print("="*60)

basic_prompt = """Classify the sentiment of customer reviews as POSITIVE, NEGATIVE, or NEUTRAL.

Review: "The product arrived on time and works perfectly!"
Sentiment: POSITIVE

Review: "Absolute waste of money. Broke after one day."
Sentiment: NEGATIVE

Review: "It's okay I guess. Does what it says."
Sentiment: NEUTRAL

Review: "I've been using this for three months and it still works great. Would buy again."
Sentiment:"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": basic_prompt}],
    max_tokens=10,
)
print(f"Input: 'I've been using this for three months...'")
print(f"Output: {response.choices[0].message.content.strip()}")


# ── 2. Format Few-Shot ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. FORMAT FEW-SHOT — Teach the output structure")
print("="*60)

format_examples = [
    {
        "role": "user",
        "content": "Extract key info from: 'John Smith, 28, Software Engineer at Google, based in San Francisco. Contact: john@email.com'"
    },
    {
        "role": "assistant",
        "content": '{"name": "John Smith", "age": 28, "job_title": "Software Engineer", "company": "Google", "city": "San Francisco", "email": "john@email.com"}'
    },
    {
        "role": "user",
        "content": "Extract key info from: 'Dr. Priya Patel, 45, Cardiologist at AIIMS Delhi. Phone: +91-9876543210'"
    },
    {
        "role": "assistant",
        "content": '{"name": "Dr. Priya Patel", "age": 45, "job_title": "Cardiologist", "company": "AIIMS Delhi", "city": "Delhi", "email": null, "phone": "+91-9876543210"}'
    },
    {
        "role": "user",
        "content": "Extract key info from: 'Maria Garcia, Senior Data Scientist at Netflix, LA. maria.garcia@netflix.com'"
    },
]

result = chat(format_examples)
print(f"Input: 'Maria Garcia, Senior Data Scientist at Netflix, LA...'")
print(f"Output:\n{result}")


# ── 3. Dynamic Few-Shot ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. DYNAMIC FEW-SHOT — Select relevant examples at runtime")
print("="*60)

# Example pool: different types of customer issues
EXAMPLE_POOL = {
    "shipping": [
        ("My order hasn't arrived after 2 weeks!", "I apologise for the delay. I've escalated your order #[ORDER_ID] to our logistics team. You'll receive an update within 24 hours."),
        ("The tracking shows 'delivered' but I got nothing.", "I'm sorry to hear that. This sometimes happens when a package is left with a neighbour or in a secure location. Could you check with neighbours? If not found, I'll arrange a replacement."),
    ],
    "refund": [
        ("I want my money back, this is terrible!", "I completely understand your frustration. I'd be happy to process a full refund. Could you share your order number so I can initiate it right away?"),
        ("How long does a refund take?", "Refunds typically take 5-7 business days to appear on your statement, depending on your bank. I'll process it now — you'll get a confirmation email shortly."),
    ],
    "technical": [
        ("The app keeps crashing!", "I apologise for the inconvenience. Please try: (1) Force-close the app, (2) Clear cache in Settings → Apps, (3) Reinstall if the issue persists. Does that help?"),
        ("I can't log in, it says 'invalid password'", "Let's get you back in. Use 'Forgot Password' on the login page — you'll receive a reset link in under 2 minutes. Check your spam folder too."),
    ],
}


def select_examples(query: str) -> list[tuple]:
    """Simple keyword-based example selector (in production: use embeddings)."""
    q = query.lower()
    if any(w in q for w in ["shipping", "arrived", "delivery", "tracking", "package"]):
        return EXAMPLE_POOL["shipping"]
    elif any(w in q for w in ["refund", "money", "return", "cancel"]):
        return EXAMPLE_POOL["refund"]
    else:
        return EXAMPLE_POOL["technical"]


def dynamic_few_shot_respond(user_query: str) -> str:
    examples = select_examples(user_query)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful customer support agent. Be empathetic, concise, and solution-focused."
        }
    ]
    # Add dynamically selected examples
    for user_ex, assistant_ex in examples:
        messages.append({"role": "user", "content": user_ex})
        messages.append({"role": "assistant", "content": assistant_ex})

    messages.append({"role": "user", "content": user_query})
    return chat(messages)


queries = [
    "My package was supposed to arrive 3 days ago and I can't find it anywhere.",
    "I bought something last week and want to return it, can I get a refund?",
    "The website is giving me a 500 error when I try to checkout.",
]

for q in queries:
    print(f"\n👤 Customer: {q}")
    print(f"🤖 Support: {dynamic_few_shot_respond(q)}")


# ── 4. Contrastive Few-Shot ───────────────────────────────────────────────────
print("\n" + "="*60)
print("4. CONTRASTIVE FEW-SHOT — Show good AND bad examples")
print("="*60)

contrastive_prompt = """Write professional email subject lines for marketing campaigns.

BAD: "BUY NOW!!!! SALE SALE SALE 50% OFF!!!!"
GOOD: "Limited time: 50% off your next order"
Why GOOD works: specific, creates urgency without being spammy

BAD: "Important message from our company"
GOOD: "Your exclusive member discount expires tonight"
Why GOOD works: personalised, specific benefit, creates urgency

BAD: "Newsletter #47"
GOOD: "5 recipes you can make in under 15 minutes"
Why GOOD works: specific number, clear value proposition, curiosity-inducing

Now write a subject line for: A back-to-school sale with 30% off backpacks and stationery."""

result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": contrastive_prompt}],
    max_tokens=80,
)
print(f"Prompt: Back-to-school sale, 30% off backpacks and stationery")
print(f"Subject line: {result.choices[0].message.content.strip()}")


# ── 5. CoT Few-Shot ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("5. CHAIN-OF-THOUGHT FEW-SHOT — Show reasoning steps")
print("="*60)

cot_few_shot = """Solve word problems step by step.

Problem: A chai shop sells 3 types of chai: Masala (₹25), Ginger (₹30), Kadak (₹20).
A customer buys 2 Masala and 3 Ginger chais. What's the total?
Solution:
  Step 1: Cost of Masala chais = 2 × ₹25 = ₹50
  Step 2: Cost of Ginger chais = 3 × ₹30 = ₹90
  Step 3: Total = ₹50 + ₹90 = ₹140
Answer: ₹140

Problem: A train travels at 80 km/h. It needs to cover 360 km.
If it starts at 9:00 AM, what time does it arrive?
Solution:
  Step 1: Time = Distance ÷ Speed = 360 ÷ 80 = 4.5 hours
  Step 2: 4.5 hours = 4 hours 30 minutes
  Step 3: Arrival = 9:00 AM + 4h 30m = 1:30 PM
Answer: 1:30 PM

Problem: A shop offers a 20% discount on a ₹1500 item, then charges 18% GST on the discounted price. What is the final price?
Solution:"""

result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": cot_few_shot}],
    max_tokens=150,
)
print(f"Problem: Discount + GST calculation")
print(f"Solution:\n{result.choices[0].message.content.strip()}")


print("\n\n💡 Key Takeaways:")
print("  1. Basic few-shot → format + label consistency")
print("  2. Format few-shot → teach output structure (great for JSON/XML)")
print("  3. Dynamic few-shot → pick examples by similarity (use embeddings in production)")
print("  4. Contrastive few-shot → show what to avoid (improves quality)")
print("  5. CoT few-shot → show reasoning steps (essential for math/logic)")
