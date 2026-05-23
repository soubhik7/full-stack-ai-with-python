"""
02_llm_as_judge.py — Using GPT-4 to Evaluate LLM Outputs
==========================================================
LLM-as-judge is the most practical evaluation method when:
  - You don't have reference answers
  - You need to measure subjective qualities (helpfulness, tone)
  - You're comparing two models head-to-head

This script demonstrates:
  1. Single-answer grading (1-10 with criteria)
  2. Pairwise comparison (A vs B)
  3. Multi-criteria rubric scoring
  4. RAG faithfulness evaluation
  5. Bias detection in judge evaluations

Run: python 06_large_language_models/05_llm_evaluation/02_llm_as_judge.py
Requires: OPENAI_API_KEY in .env
"""

import json
import statistics
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def llm_call(system: str, user: str, max_tokens: int = 500) -> str:
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
# 1. SINGLE-ANSWER GRADING
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("1. SINGLE-ANSWER GRADING — Rate on a 1-10 scale")
print("="*65)


class SingleGrade(BaseModel):
    score: int = Field(ge=1, le=10, description="Quality score 1-10")
    reasoning: str = Field(description="Why this score was given")
    strengths: list[str] = Field(description="What the answer does well")
    weaknesses: list[str] = Field(description="What could be improved")


def grade_answer(question: str, answer: str, rubric: str) -> SingleGrade:
    """Ask GPT-4 to grade an answer with detailed reasoning."""
    response = client.beta.chat.completions.parse(
        model="gpt-4o",  # use strongest model as judge
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator. Grade answers objectively based on the rubric. "
                    "Be strict but fair. A 10 is exceptional, 7 is good, 5 is average."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Answer to evaluate:\n{answer}\n\n"
                    f"Rubric: {rubric}"
                ),
            },
        ],
        response_format=SingleGrade,
        max_tokens=400,
    )
    return response.choices[0].message.parsed


QUESTION = "What is the difference between supervised and unsupervised learning?"
RUBRIC = "Rate on: (1) Technical accuracy, (2) Clarity for a beginner, (3) Use of examples"

ANSWERS = {
    "Vague answer": "Supervised learning uses labels and unsupervised doesn't.",
    "Good answer": (
        "In supervised learning, the model trains on labelled data — each example has "
        "an input and a correct output. The model learns to predict outputs for new inputs. "
        "Example: email spam detection (label = spam/not-spam).\n\n"
        "In unsupervised learning, there are no labels. The model finds patterns on its own. "
        "Example: customer segmentation — the algorithm groups similar customers without "
        "being told what the groups should be."
    ),
    "Over-complicated": (
        "Supervised learning optimises parameters θ to minimise L(f(x;θ), y) over a labelled "
        "dataset D={(xi,yi)}. Unsupervised learning seeks to model p(x) without access to labels, "
        "typically through density estimation, clustering via EM algorithms, or dimensionality "
        "reduction using PCA or autoencoders."
    ),
}

for answer_type, answer in ANSWERS.items():
    grade = grade_answer(QUESTION, answer, RUBRIC)
    print(f"\n📝 Answer type: {answer_type}")
    print(f"   Score: {grade.score}/10")
    print(f"   Reasoning: {grade.reasoning[:120]}...")
    print(f"   Strengths: {grade.strengths[0] if grade.strengths else 'None'}")
    print(f"   Weaknesses: {grade.weaknesses[0] if grade.weaknesses else 'None'}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PAIRWISE COMPARISON (A/B Testing)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("2. PAIRWISE COMPARISON — Which answer is better?")
print("="*65)


class PairwiseResult(BaseModel):
    winner: str = Field(description="'A', 'B', or 'TIE'")
    confidence: str = Field(description="'HIGH', 'MEDIUM', or 'LOW'")
    reasoning: str
    a_strengths: str
    b_strengths: str


def compare_answers(question: str, answer_a: str, answer_b: str) -> PairwiseResult:
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You compare two AI assistant answers and determine which is better. "
                    "Consider accuracy, helpfulness, and clarity. Be decisive."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Answer A:\n{answer_a}\n\n"
                    f"Answer B:\n{answer_b}\n\n"
                    "Which answer is better?"
                ),
            },
        ],
        response_format=PairwiseResult,
        max_tokens=400,
    )
    return response.choices[0].message.parsed


COMPARE_QUESTION = "How does attention work in transformers?"

ANSWER_A = (
    "Attention allows the model to focus on relevant parts of the input. "
    "It computes a weighted sum of values based on similarity between queries and keys."
)

ANSWER_B = (
    "Attention in transformers works like this:\n"
    "1. Each word creates three vectors: Query (what am I looking for?), "
    "Key (what do I contain?), Value (what should I return?).\n"
    "2. A 'relevance score' is computed between my Query and every other word's Key.\n"
    "3. These scores are softmaxed into weights.\n"
    "4. My output = weighted sum of all Values.\n\n"
    "Intuition: When processing 'The cat sat on it', attention helps 'it' "
    "attend strongly to 'cat', resolving the coreference."
)

result = compare_answers(COMPARE_QUESTION, ANSWER_A, ANSWER_B)
print(f"Question: {COMPARE_QUESTION[:70]}...")
print(f"\nWinner: {result.winner} (confidence: {result.confidence})")
print(f"Reasoning: {result.reasoning[:200]}...")
print(f"A strengths: {result.a_strengths[:100]}...")
print(f"B strengths: {result.b_strengths[:100]}...")


# ══════════════════════════════════════════════════════════════════════════════
# 3. MULTI-CRITERIA RUBRIC SCORING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("3. MULTI-CRITERIA RUBRIC — Evaluate each dimension separately")
print("="*65)


class CriterionScore(BaseModel):
    criterion: str
    score: int = Field(ge=1, le=5)
    justification: str


class RubricEvaluation(BaseModel):
    criteria_scores: list[CriterionScore]
    overall_score: float
    summary: str
    recommendation: str


CRITERIA = [
    "Accuracy: Is the information factually correct?",
    "Completeness: Does it cover all important aspects?",
    "Clarity: Is it easy to understand?",
    "Conciseness: Is it appropriately brief without missing key points?",
    "Actionability: Does it give concrete, usable information?",
]

EVAL_QUESTION = "How do I improve the performance of a slow Python script?"
EVAL_ANSWER = (
    "To speed up Python: use list comprehensions instead of loops, "
    "avoid global variables, and consider using NumPy for numerical operations. "
    "You can also try PyPy as a faster alternative Python runtime."
)

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": (
                "You evaluate AI answers on a detailed rubric. "
                "Score each criterion 1-5. Be precise and fair."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {EVAL_QUESTION}\n\n"
                f"Answer: {EVAL_ANSWER}\n\n"
                f"Evaluate on these criteria:\n"
                + "\n".join(f"- {c}" for c in CRITERIA)
            ),
        },
    ],
    response_format=RubricEvaluation,
    max_tokens=600,
)

eval_result: RubricEvaluation = response.choices[0].message.parsed
print(f"Answer: '{EVAL_ANSWER[:80]}...'\n")
for cs in eval_result.criteria_scores:
    stars = "★" * cs.score + "☆" * (5 - cs.score)
    print(f"  {cs.criterion.split(':')[0]:<15} {stars} ({cs.score}/5): {cs.justification[:60]}...")
print(f"\n  Overall: {eval_result.overall_score:.1f}/5.0")
print(f"  Summary: {eval_result.summary}")
print(f"  Recommendation: {eval_result.recommendation}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. RAG FAITHFULNESS EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("4. RAG FAITHFULNESS — Is the answer grounded in the context?")
print("="*65)


class FaithfulnessResult(BaseModel):
    is_faithful: bool
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    hallucinated_claims: list[str] = Field(description="Claims NOT supported by context")
    supported_claims: list[str] = Field(description="Claims properly grounded in context")
    explanation: str


def evaluate_rag_faithfulness(
    question: str,
    context: str,
    answer: str,
) -> FaithfulnessResult:
    """Check if a RAG answer is faithful to the retrieved context."""
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You evaluate whether an AI answer is faithful to the provided context. "
                    "Flag any claims in the answer that are NOT supported by the context (hallucinations). "
                    "A faithful answer ONLY uses information present in the context."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Context (retrieved documents):\n{context}\n\n"
                    f"Generated Answer:\n{answer}\n\n"
                    "Evaluate faithfulness."
                ),
            },
        ],
        response_format=FaithfulnessResult,
        max_tokens=500,
    )
    return response.choices[0].message.parsed


CONTEXT = """
MCP (Model Context Protocol) was released by Anthropic in November 2024.
It is an open-source protocol that standardises how AI applications connect
to external tools and data sources. MCP uses JSON-RPC 2.0 as its message format
and supports two transport types: stdio and SSE (Server-Sent Events).
The protocol defines three primitive types: Tools, Resources, and Prompts.
"""

FAITHFUL_ANSWER = (
    "MCP is an open-source protocol from Anthropic (released November 2024). "
    "It standardises AI tool connections using JSON-RPC 2.0, with stdio and SSE transports. "
    "It supports three primitives: Tools, Resources, and Prompts."
)

HALLUCINATED_ANSWER = (
    "MCP is Anthropic's protocol released in 2024. It uses WebSocket transport "
    "and supports four primitives: Tools, Resources, Prompts, and Actions. "
    "It was built in collaboration with OpenAI and Google DeepMind."
)

for label, answer in [("Faithful answer", FAITHFUL_ANSWER), ("Hallucinated answer", HALLUCINATED_ANSWER)]:
    result = evaluate_rag_faithfulness("What is MCP?", CONTEXT, answer)
    print(f"\n📋 {label}:")
    print(f"   Faithful: {'✅' if result.is_faithful else '❌'}")
    print(f"   Score: {result.faithfulness_score:.0%}")
    if result.hallucinated_claims:
        print(f"   ⚠️  Hallucinations: {result.hallucinated_claims}")
    print(f"   Explanation: {result.explanation[:120]}...")


print("\n\n💡 LLM-as-Judge Best Practices:")
print("  ✓ Use a stronger model as judge (GPT-4o > GPT-4o-mini)")
print("  ✓ Always ask for reasoning, not just a score")
print("  ✓ Use position-swapped pairwise eval to reduce order bias")
print("  ✓ Average multiple judge samples for high-stakes decisions")
print("  ✓ Calibrate the judge with known-good/bad examples")
