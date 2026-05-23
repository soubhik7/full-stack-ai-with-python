"""
App 16 — Structured Outputs (Production Patterns)
===================================================
Six production patterns for getting reliable structured data from LLMs.
When you need LLM output your code can actually use.

Patterns:
  1. Basic Pydantic extraction
  2. Enum + constrained fields
  3. Nested models
  4. Bulk extraction (list of items)
  5. Conditional/optional fields
  6. Multi-step structured pipeline

Run: python 08_ai_apps/16_structured_outputs/main.py
Requires: OPENAI_API_KEY in .env
"""

import json
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

load_dotenv()
client = OpenAI()


# ══════════════════════════════════════════════════════════════════════════════
# 1. BASIC PYDANTIC EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("1. BASIC PYDANTIC EXTRACTION")
print("="*60)


class BookSummary(BaseModel):
    title: str
    author: str
    year_published: Optional[int] = None
    genre: str
    one_sentence_summary: str
    rating: float = Field(ge=1.0, le=10.0, description="1-10 quality score")
    recommended_for: list[str] = Field(description="Types of readers who'd enjoy this")


text = """
'The Pragmatic Programmer' by David Thomas and Andrew Hunt (published 1999)
is an essential read for software engineers. It covers career development,
coding practices, and professional skills. An absolute classic that shaped
modern software engineering culture. I'd give it a 9 out of 10.
"""

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract book information from text."},
        {"role": "user", "content": text},
    ],
    response_format=BookSummary,
)

book: BookSummary = response.choices[0].message.parsed
print(f"\n📚 Title: {book.title}")
print(f"   Author: {book.author}")
print(f"   Year: {book.year_published}")
print(f"   Genre: {book.genre}")
print(f"   Rating: {book.rating}/10")
print(f"   For: {', '.join(book.recommended_for)}")
print(f"   Summary: {book.one_sentence_summary}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ENUM FIELDS — Constrain the output to valid options
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. ENUM FIELDS — Constrained categorical outputs")
print("="*60)


class Sentiment(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketClassification(BaseModel):
    category: str = Field(description="Issue category: billing, technical, shipping, other")
    sentiment: Sentiment
    urgency: Urgency
    summary: str = Field(max_length=100, description="One-sentence summary of the issue")
    needs_human_agent: bool
    suggested_response_template: str = Field(description="Template ID to use: apology, info, escalate, resolve")


tickets = [
    "MY ORDER HAS BEEN STUCK IN PROCESSING FOR 2 WEEKS. THIS IS UNACCEPTABLE. I WANT A REFUND NOW!!!",
    "Hi, could you tell me what your return policy is? Thanks",
    "The app crashed and I lost all my data. Pretty frustrated but willing to wait for a fix.",
]

for ticket in tickets:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You classify customer support tickets."},
            {"role": "user", "content": f"Classify this ticket: {ticket}"},
        ],
        response_format=TicketClassification,
    )
    tc: TicketClassification = response.choices[0].message.parsed
    print(f"\n📧 Ticket: {ticket[:60]}...")
    print(f"   Sentiment: {tc.sentiment.value}  Urgency: {tc.urgency.value}  Human: {tc.needs_human_agent}")
    print(f"   Category: {tc.category}  Template: {tc.suggested_response_template}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. NESTED MODELS — Complex hierarchical extraction
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. NESTED MODELS — Hierarchical data extraction")
print("="*60)


class Ingredient(BaseModel):
    name: str
    amount: str
    unit: Optional[str] = None
    is_optional: bool = False


class RecipeStep(BaseModel):
    step_number: int
    instruction: str
    duration_minutes: Optional[int] = None
    tip: Optional[str] = None


class Recipe(BaseModel):
    name: str
    cuisine: str
    servings: int
    prep_time_minutes: int
    cook_time_minutes: int
    difficulty: str  # easy, medium, hard
    ingredients: list[Ingredient]
    steps: list[RecipeStep]
    tags: list[str]


recipe_text = """
Masala Chai — Classic Indian Spiced Tea (serves 2)
Prep: 5 minutes, Cook: 10 minutes. Difficulty: Easy.

Ingredients:
- 2 cups whole milk (you can substitute oat milk)
- 1 cup water
- 2 tsp CTC or Assam black tea leaves
- 1-inch fresh ginger, grated
- 4 green cardamom pods, crushed
- 1 stick cinnamon
- 2-3 tbsp sugar to taste

Instructions:
1. Bring water to boil with ginger and spices (5 min). Infuse the flavours.
2. Add tea leaves and simmer for 2 minutes until dark. Tip: don't over-boil.
3. Add milk, bring back to boil, simmer 3 minutes. Stir occasionally.
4. Strain into cups through a fine mesh strainer.

Tags: beverage, Indian, quick, vegetarian, spiced
"""

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract complete recipe information into structured format."},
        {"role": "user", "content": recipe_text},
    ],
    response_format=Recipe,
)

recipe: Recipe = response.choices[0].message.parsed
print(f"\n🍵 Recipe: {recipe.name} ({recipe.cuisine})")
print(f"   Serves {recipe.servings} | Prep {recipe.prep_time_minutes}m + Cook {recipe.cook_time_minutes}m")
print(f"   Difficulty: {recipe.difficulty}")
print(f"   Ingredients ({len(recipe.ingredients)}):")
for ing in recipe.ingredients:
    opt = " (optional)" if ing.is_optional else ""
    print(f"     - {ing.amount} {ing.unit or ''} {ing.name}{opt}")
print(f"   Steps: {len(recipe.steps)}")
for step in recipe.steps:
    print(f"     {step.step_number}. {step.instruction[:60]}...")


# ══════════════════════════════════════════════════════════════════════════════
# 4. BULK EXTRACTION — Extract a list of items
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. BULK EXTRACTION — Extract arrays from unstructured text")
print("="*60)


class CompanyMention(BaseModel):
    company_name: str
    ticker_symbol: Optional[str] = None
    sentiment: str  # positive, negative, neutral
    context: str = Field(description="Why the company is mentioned")


class NewsAnalysis(BaseModel):
    headline_summary: str
    companies_mentioned: list[CompanyMention]
    market_impact: str  # bullish, bearish, neutral
    key_numbers: list[str] = Field(description="Any financial figures mentioned")


news = """
Markets surged today as Apple (AAPL) beat earnings expectations by 15%,
reporting $89.5B in revenue. Microsoft (MSFT) also performed well, up 8%
on cloud growth. However, Meta (META) fell 12% after disappointing ad revenue,
dragging the Nasdaq down slightly. Tesla (TSLA) remained flat amid mixed
delivery numbers. Analysts remain cautious about NVIDIA's (NVDA) valuation
after its 200% run this year.
"""

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Analyse financial news and extract structured data."},
        {"role": "user", "content": news},
    ],
    response_format=NewsAnalysis,
)

analysis: NewsAnalysis = response.choices[0].message.parsed
print(f"\n📰 Summary: {analysis.headline_summary}")
print(f"   Market impact: {analysis.market_impact}")
print(f"   Key numbers: {', '.join(analysis.key_numbers)}")
print(f"\n   Companies mentioned:")
for co in analysis.companies_mentioned:
    ticker = f"({co.ticker_symbol})" if co.ticker_symbol else ""
    print(f"     {co.company_name} {ticker}: {co.sentiment} — {co.context[:60]}...")


# ══════════════════════════════════════════════════════════════════════════════
# 5. MULTI-STEP STRUCTURED PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. MULTI-STEP PIPELINE — Chain structured outputs")
print("="*60)


class IntentClassification(BaseModel):
    intent: str  # question, task, complaint, compliment, other
    entities: list[str] = Field(description="Key entities mentioned")
    requires_tool: bool
    tool_name: Optional[str] = None


class Response(BaseModel):
    text: str
    follow_up_questions: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


def process_user_message(message: str) -> dict:
    """Two-step pipeline: classify intent → generate appropriate response."""

    # Step 1: Classify
    intent_resp = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classify the user's intent and extract entities."},
            {"role": "user", "content": message},
        ],
        response_format=IntentClassification,
    )
    intent: IntentClassification = intent_resp.choices[0].message.parsed

    # Step 2: Generate response based on classification
    response_resp = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a helpful AI assistant. "
                    f"The user's intent is '{intent.intent}'. "
                    f"Key entities: {intent.entities}. "
                    f"Generate a helpful response."
                ),
            },
            {"role": "user", "content": message},
        ],
        response_format=Response,
    )
    final: Response = response_resp.choices[0].message.parsed

    return {
        "intent": intent.intent,
        "entities": intent.entities,
        "response": final.text,
        "follow_ups": final.follow_up_questions,
        "confidence": final.confidence,
    }


test_messages = [
    "What's the best way to learn Python for machine learning?",
    "I'm getting a ModuleNotFoundError when importing numpy.",
]

for msg in test_messages:
    result = process_user_message(msg)
    print(f"\n👤 User: {msg}")
    print(f"   Intent: {result['intent']}")
    print(f"   Entities: {result['entities']}")
    print(f"   Response: {result['response'][:120]}...")
    print(f"   Confidence: {result['confidence']:.0%}")
    print(f"   Follow-ups: {result['follow_ups'][0] if result['follow_ups'] else 'None'}")


print("\n\n💡 Summary:")
print("  client.beta.chat.completions.parse(response_format=Model) → typed output")
print("  Enum fields → LLM can only output valid enum values")
print("  Nested models → handles complex hierarchical structures")
print("  list[Model] inside a wrapper → extract arrays of structured items")
print("  Chain multiple parse() calls → build structured pipelines")
