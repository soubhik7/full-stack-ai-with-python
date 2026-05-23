"""
03_eval_pipeline.py — Complete Evaluation Pipeline
====================================================
Build a reusable evaluation pipeline that:
  1. Loads a golden dataset (question-answer pairs)
  2. Generates model responses
  3. Scores them with multiple metrics
  4. Produces a summary report
  5. Compares two model versions (A/B test)

This is what you'd run in CI/CD to catch prompt regressions.

Run: python 06_large_language_models/05_llm_evaluation/03_eval_pipeline.py
Requires: OPENAI_API_KEY in .env
"""

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


# ── Data Types ────────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """One test case in the evaluation dataset."""
    id: str
    question: str
    reference_answer: str
    category: str = "general"
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating one sample."""
    sample_id: str
    question: str
    reference: str
    prediction: str
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalReport:
    """Aggregate report across all samples."""
    model_name: str
    total_samples: int
    avg_scores: dict[str, float]
    by_category: dict[str, dict[str, float]]
    results: list[EvalResult]
    latency_ms: float


# ── Golden Dataset ────────────────────────────────────────────────────────────

GOLDEN_DATASET: list[EvalSample] = [
    EvalSample(
        id="ml-001",
        question="What is gradient descent?",
        reference_answer=(
            "Gradient descent is an optimization algorithm that iteratively adjusts "
            "model parameters to minimize a loss function. It computes the gradient "
            "(partial derivatives) of the loss with respect to each parameter and "
            "moves in the opposite direction by a small step size called the learning rate."
        ),
        category="machine_learning",
    ),
    EvalSample(
        id="ml-002",
        question="What is overfitting and how do you prevent it?",
        reference_answer=(
            "Overfitting is when a model learns the training data too well, including "
            "noise, and fails to generalise to new data. Prevention techniques include: "
            "regularisation (L1/L2), dropout, early stopping, cross-validation, "
            "increasing training data, and reducing model complexity."
        ),
        category="machine_learning",
    ),
    EvalSample(
        id="dl-001",
        question="What is a convolutional neural network (CNN) used for?",
        reference_answer=(
            "CNNs are designed for processing grid-like data (images, video, audio spectrograms). "
            "They use convolutional layers to detect local patterns (edges, textures, shapes) "
            "regardless of their position in the input. Commonly used for image classification, "
            "object detection, image segmentation, and face recognition."
        ),
        category="deep_learning",
    ),
    EvalSample(
        id="dl-002",
        question="Explain the vanishing gradient problem.",
        reference_answer=(
            "The vanishing gradient problem occurs in deep networks during backpropagation: "
            "gradients become extremely small as they propagate backward through many layers, "
            "causing early layers to learn very slowly or stop learning. "
            "Solutions: ReLU activation (instead of sigmoid/tanh), batch normalisation, "
            "residual connections (ResNet), and LSTM/GRU for recurrent networks."
        ),
        category="deep_learning",
    ),
    EvalSample(
        id="llm-001",
        question="What is the difference between GPT and BERT?",
        reference_answer=(
            "GPT (decoder-only) is trained to predict the next token, making it ideal for "
            "text generation tasks. BERT (encoder-only) is trained with masked language "
            "modelling and next sentence prediction, making it ideal for classification, "
            "NER, and question answering. GPT generates left-to-right; BERT reads "
            "bidirectionally for understanding."
        ),
        category="llms",
    ),
    EvalSample(
        id="llm-002",
        question="What is RAG (Retrieval-Augmented Generation)?",
        reference_answer=(
            "RAG combines retrieval and generation: given a query, it first retrieves "
            "relevant documents from a knowledge base (using vector similarity search), "
            "then feeds both the query and retrieved context to an LLM to generate "
            "a grounded answer. RAG reduces hallucination and keeps knowledge current "
            "without fine-tuning the model."
        ),
        category="llms",
    ),
]


# ── Model Under Test ──────────────────────────────────────────────────────────

def generate_with_model(
    question: str,
    system_prompt: str,
    model: str = "gpt-4o-mini",
) -> tuple[str, float]:
    """Generate a response and measure latency."""
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        max_tokens=300,
        temperature=0.0,
    )
    latency_ms = (time.time() - start) * 1000
    return response.choices[0].message.content.strip(), latency_ms


# ── Scorers ───────────────────────────────────────────────────────────────────

def rouge_l_score(reference: str, hypothesis: str) -> float:
    """Simple ROUGE-L F1 score without external dependencies."""
    def lcs(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref or not hyp:
        return 0.0
    l = lcs(ref, hyp)
    p = l / len(hyp)
    r = l / len(ref)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def llm_helpfulness_score(question: str, answer: str) -> float:
    """LLM-as-judge score for helpfulness (0-1)."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You rate answers 1-10 on helpfulness, accuracy, and clarity. "
                    "Return only the number."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\nAnswer: {answer}",
            },
        ],
        max_tokens=5,
        temperature=0.0,
    )
    try:
        score = float(response.choices[0].message.content.strip())
        return score / 10.0
    except ValueError:
        return 0.5


def length_penalty_score(answer: str, reference: str) -> float:
    """Penalise answers that are too short or absurdly long."""
    ref_len = len(reference.split())
    ans_len = len(answer.split())

    # Ideal range: 50% to 200% of reference length
    ratio = ans_len / ref_len if ref_len > 0 else 0
    if 0.5 <= ratio <= 2.0:
        return 1.0
    elif ratio < 0.5:
        return ratio / 0.5  # penalise too short
    else:
        return max(0.0, 1.0 - (ratio - 2.0) / 4.0)  # penalise too long


# ── Evaluation Pipeline ───────────────────────────────────────────────────────

def run_evaluation(
    dataset: list[EvalSample],
    system_prompt: str,
    model: str = "gpt-4o-mini",
    scorers: Optional[dict[str, Callable]] = None,
) -> EvalReport:
    """
    Run the full evaluation pipeline.

    Args:
        dataset: List of test cases.
        system_prompt: The system prompt to test.
        model: OpenAI model to evaluate.
        scorers: Dict of scorer_name → scorer_function(question, reference, prediction).

    Returns:
        EvalReport with aggregate and per-sample results.
    """
    if scorers is None:
        scorers = {
            "rouge_l": lambda q, ref, pred: rouge_l_score(ref, pred),
            "llm_helpfulness": lambda q, ref, pred: llm_helpfulness_score(q, pred),
            "length_quality": lambda q, ref, pred: length_penalty_score(pred, ref),
        }

    results: list[EvalResult] = []
    total_latency = 0.0

    print(f"\n🧪 Evaluating model: {model} on {len(dataset)} samples...")

    for sample in dataset:
        print(f"  [{sample.id}] {sample.question[:50]}...", end=" ", flush=True)

        # Generate prediction
        prediction, latency_ms = generate_with_model(sample.question, system_prompt, model)
        total_latency += latency_ms

        # Score
        scores = {}
        for scorer_name, scorer_fn in scorers.items():
            try:
                scores[scorer_name] = scorer_fn(sample.question, sample.reference_answer, prediction)
            except Exception as e:
                scores[scorer_name] = 0.0

        composite = sum(scores.values()) / len(scores) if scores else 0.0
        scores["composite"] = composite

        results.append(EvalResult(
            sample_id=sample.id,
            question=sample.question,
            reference=sample.reference_answer,
            prediction=prediction,
            scores=scores,
            metadata={"category": sample.category, "latency_ms": latency_ms},
        ))
        print(f"✅ composite={composite:.2f}")

    # Aggregate scores
    all_score_names = list(scorers.keys()) + ["composite"]
    avg_scores = {
        name: sum(r.scores.get(name, 0) for r in results) / len(results)
        for name in all_score_names
    }

    # By category
    categories = set(s.category for s in dataset)
    by_category = {}
    for cat in categories:
        cat_results = [r for r in results if r.metadata.get("category") == cat]
        by_category[cat] = {
            name: sum(r.scores.get(name, 0) for r in cat_results) / len(cat_results)
            for name in all_score_names
        }

    return EvalReport(
        model_name=model,
        total_samples=len(dataset),
        avg_scores=avg_scores,
        by_category=by_category,
        results=results,
        latency_ms=total_latency / len(dataset),
    )


def print_report(report: EvalReport):
    """Print a formatted evaluation report."""
    print(f"\n{'='*65}")
    print(f"  EVALUATION REPORT — {report.model_name}")
    print(f"{'='*65}")
    print(f"  Samples evaluated: {report.total_samples}")
    print(f"  Avg latency: {report.latency_ms:.0f} ms/sample")

    print(f"\n  AVERAGE SCORES:")
    for metric, score in report.avg_scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"    {metric:<20} {bar} {score:.3f}")

    print(f"\n  BY CATEGORY:")
    for cat, scores in report.by_category.items():
        composite = scores.get("composite", 0)
        print(f"    {cat:<25} composite={composite:.3f}")

    print(f"\n  WORST PERFORMING SAMPLES:")
    sorted_results = sorted(report.results, key=lambda r: r.scores.get("composite", 0))
    for r in sorted_results[:2]:
        print(f"    [{r.sample_id}] composite={r.scores.get('composite', 0):.3f}")
        print(f"      Q: {r.question[:60]}...")
        print(f"      Generated: {r.prediction[:80]}...")


# ── Run the Pipeline ──────────────────────────────────────────────────────────

# System prompt v1: baseline
PROMPT_V1 = "You are a helpful AI assistant that answers machine learning questions."

# System prompt v2: improved
PROMPT_V2 = (
    "You are an expert ML educator. When answering questions:\n"
    "1. Give a clear definition first\n"
    "2. Explain the intuition\n"
    "3. Mention 2-3 practical applications or techniques\n"
    "Keep responses between 80-150 words. Use plain language."
)

print("Running A/B evaluation: Baseline vs Improved Prompt")
print("This will make API calls and may take 1-2 minutes...\n")

# Evaluate both prompts on the same dataset
report_v1 = run_evaluation(GOLDEN_DATASET, PROMPT_V1, model="gpt-4o-mini")
report_v2 = run_evaluation(GOLDEN_DATASET, PROMPT_V2, model="gpt-4o-mini")

print_report(report_v1)
print_report(report_v2)

# Compare
print(f"\n{'='*65}")
print("  A/B COMPARISON: Baseline vs Improved Prompt")
print(f"{'='*65}")
for metric in report_v1.avg_scores:
    v1_score = report_v1.avg_scores[metric]
    v2_score = report_v2.avg_scores[metric]
    delta = v2_score - v1_score
    direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
    print(f"  {metric:<20} {v1_score:.3f} → {v2_score:.3f}  {direction} {abs(delta):.3f}")

composite_v1 = report_v1.avg_scores.get("composite", 0)
composite_v2 = report_v2.avg_scores.get("composite", 0)
winner = "V2 (Improved)" if composite_v2 > composite_v1 else "V1 (Baseline)"
print(f"\n  🏆 Winner: {winner} (composite: {max(composite_v1, composite_v2):.3f})")
