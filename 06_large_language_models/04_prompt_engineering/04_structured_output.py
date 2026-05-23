"""
04_structured_output.py — Getting JSON from LLMs
=================================================
Getting reliable structured data from LLMs:
  1. JSON mode (OpenAI) — forces valid JSON
  2. Pydantic + .parse() — type-safe parsing with validation
  3. Structured output schemas — guarantee specific fields
  4. Multi-level nesting — complex nested JSON
  5. Function calling as structured output

These patterns are essential when LLM output feeds into your code.

Run: python 06_large_language_models/04_prompt_engineering/04_structured_output.py
"""

import json
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

load_dotenv()
client = OpenAI()


# ══════════════════════════════════════════════════════════════════════════════
# 1. JSON Mode — Forces the model to output valid JSON
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("1. JSON MODE — Always returns valid JSON")
print("="*60)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},   # ← key flag
    messages=[
        {
            "role": "system",
            "content": "You extract information and return it as JSON.",
        },
        {
            "role": "user",
            "content": (
                "Extract all people, their roles, and companies from: "
                "'Sundar Pichai is CEO of Google. Sam Altman leads OpenAI. "
                "Dario Amodei co-founded Anthropic.'"
            ),
        },
    ],
    max_tokens=300,
)

raw = response.choices[0].message.content
data = json.loads(raw)  # Always safe because response_format=json_object
print(f"Raw JSON:\n{raw}")
print(f"\nParsed people: {[p['name'] for p in data.get('people', [])]}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Pydantic Structured Output (OpenAI .parse())
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. PYDANTIC STRUCTURED OUTPUT — Type-safe with validation")
print("="*60)


class SentimentAnalysis(BaseModel):
    text: str = Field(description="The original text analysed")
    sentiment: str = Field(description="POSITIVE, NEGATIVE, or NEUTRAL")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    key_phrases: list[str] = Field(description="Phrases that drove the sentiment")
    suggested_response: Optional[str] = Field(
        default=None,
        description="If negative, a suggested customer service response"
    )

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v: str) -> str:
        allowed = {"POSITIVE", "NEGATIVE", "NEUTRAL"}
        if v.upper() not in allowed:
            raise ValueError(f"Sentiment must be one of {allowed}")
        return v.upper()


reviews = [
    "This product is absolutely amazing! Best purchase I've made this year.",
    "Complete garbage. Stopped working after 2 days. Want a refund immediately.",
    "It's okay. Does the job but nothing special about it.",
]

for review in reviews:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Analyse the sentiment of customer reviews."},
            {"role": "user", "content": f"Analyse: {review}"},
        ],
        response_format=SentimentAnalysis,
    )

    result: SentimentAnalysis = response.choices[0].message.parsed
    print(f"\n📝 Review: {review[:60]}...")
    print(f"   Sentiment: {result.sentiment} (confidence: {result.confidence:.0%})")
    print(f"   Key phrases: {result.key_phrases}")
    if result.suggested_response:
        print(f"   Suggested reply: {result.suggested_response[:80]}...")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Complex Nested Schema
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. NESTED SCHEMA — Extract structured data from unstructured text")
print("="*60)


class Address(BaseModel):
    street: Optional[str] = None
    city: str
    state: Optional[str] = None
    country: str
    pincode: Optional[str] = None


class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None


class PersonProfile(BaseModel):
    full_name: str
    age: Optional[int] = Field(default=None, ge=0, le=150)
    occupation: str
    company: Optional[str] = None
    address: Optional[Address] = None
    contact: Optional[ContactInfo] = None
    skills: list[str] = Field(default_factory=list)
    summary: str = Field(description="One-sentence professional summary")


text = """
Meet Dr. Anjali Sharma, 38, a Machine Learning researcher at IIT Delhi.
She specialises in NLP and Computer Vision. She completed her PhD from
Stanford and lives in New Delhi, India (110016).
You can reach her at anjali.sharma@iitd.ac.in or +91-9988776655.
Find her on LinkedIn: linkedin.com/in/anjalisharma
"""

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract structured person profile data from text."},
        {"role": "user", "content": f"Extract profile from:\n{text}"},
    ],
    response_format=PersonProfile,
)

profile: PersonProfile = response.choices[0].message.parsed
print(f"Name: {profile.full_name}")
print(f"Age: {profile.age}")
print(f"Occupation: {profile.occupation} @ {profile.company}")
print(f"City: {profile.address.city if profile.address else 'N/A'}")
print(f"Email: {profile.contact.email if profile.contact else 'N/A'}")
print(f"Skills: {', '.join(profile.skills)}")
print(f"Summary: {profile.summary}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Using LLM as a Data Transformation Pipeline
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. LLM AS DATA PIPELINE — Transform noisy text into clean records")
print("="*60)


class Product(BaseModel):
    name: str
    price_inr: float = Field(ge=0)
    category: str
    in_stock: bool
    rating: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    tags: list[str] = Field(default_factory=list)


class ProductCatalog(BaseModel):
    products: list[Product]
    total_count: int
    categories: list[str]


raw_catalog = """
- iPhone 15 Pro, Rs 1,29,990, Electronics, Available ✓, ⭐⭐⭐⭐½ (4.5 stars), [smartphone, apple, 5G]
- Levi's 501 Jeans, 4999 rupees, Clothing, Out of stock ✗, 4 stars, [denim, casual, men]
- The Alchemist (Book), ₹299, Books, Yes it's available, rating: 4.8/5, [fiction, bestseller, paulo-coelho]
- Sony WH-1000XM5 headphones, INR 24,990, Electronics, In stock, 4.7, [noise-cancelling, wireless, sony]
"""

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "Parse product catalog data into structured format. Normalise prices to INR numbers.",
        },
        {"role": "user", "content": f"Parse this catalog:\n{raw_catalog}"},
    ],
    response_format=ProductCatalog,
)

catalog: ProductCatalog = response.choices[0].message.parsed
print(f"Total products: {catalog.total_count}")
print(f"Categories: {catalog.categories}")
print("\nProducts:")
for p in catalog.products:
    status = "✓" if p.in_stock else "✗"
    rating_str = f"⭐{p.rating}" if p.rating else "No rating"
    print(f"  {status} {p.name} — ₹{p.price_inr:,.0f} [{p.category}] {rating_str}")


print("\n\n💡 Summary:")
print("  response_format={type:json_object} → guarantees valid JSON, any shape")
print("  .parse(response_format=PydanticModel) → validates fields + types automatically")
print("  Nested Pydantic models → handles complex hierarchical data")
print("  Pipeline pattern → clean messy real-world data reliably")
