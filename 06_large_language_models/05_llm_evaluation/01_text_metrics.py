"""
01_text_metrics.py — Classic NLP Evaluation Metrics
====================================================
Covers BLEU, ROUGE, and BERTScore — the three most common
reference-based metrics for evaluating text generation.

Understanding when each is useful and what its scores mean.

Install first:
    pip3 install rouge-score bert-score nltk

Run: python 06_large_language_models/05_llm_evaluation/01_text_metrics.py
"""

import math
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

# ── BLEU from scratch (for understanding) ────────────────────────────────────

def ngrams(tokens: list[str], n: int) -> Counter:
    """Get n-gram counts from a token list."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    Calculate BLEU score (0–1, higher = better).

    BLEU measures n-gram precision: what fraction of n-grams in the
    hypothesis appear in the reference.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens:
        return 0.0

    # Brevity penalty (penalises short translations)
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else math.exp(
        1 - len(ref_tokens) / len(hyp_tokens)
    )

    # Calculate precision for each n-gram order
    log_score = 0.0
    for n in range(1, min(max_n + 1, len(hyp_tokens) + 1)):
        ref_ngrams = ngrams(ref_tokens, n)
        hyp_ngrams = ngrams(hyp_tokens, n)

        # Clipped precision
        matches = sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())

        if total == 0 or matches == 0:
            return 0.0

        log_score += math.log(matches / total)

    return bp * math.exp(log_score / max_n)


# ── ROUGE ─────────────────────────────────────────────────────────────────────

def rouge_1(reference: str, hypothesis: str) -> dict[str, float]:
    """
    ROUGE-1: Unigram overlap between reference and hypothesis.
    Measures recall — are the important words in the hypothesis?
    """
    ref_tokens = set(reference.lower().split())
    hyp_tokens = set(hypothesis.lower().split())

    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = ref_tokens & hyp_tokens

    precision = len(overlap) / len(hyp_tokens)
    recall = len(overlap) / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def lcs_length(a: list, b: list) -> int:
    """Length of longest common subsequence."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(reference: str, hypothesis: str) -> dict[str, float]:
    """
    ROUGE-L: Longest Common Subsequence (LCS) overlap.
    Better than ROUGE-1 because it considers word order.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = lcs_length(ref_tokens, hyp_tokens)

    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


# ══════════════════════════════════════════════════════════════════════════════
# DEMO: Evaluate different summaries of the same reference
# ══════════════════════════════════════════════════════════════════════════════

REFERENCE = (
    "Machine learning is a type of artificial intelligence that allows computers "
    "to learn from data and improve their performance without being explicitly programmed."
)

CANDIDATES = {
    "Perfect match (same text)": REFERENCE,
    "Good paraphrase": (
        "Machine learning is an AI technique where computers learn from data "
        "to improve performance without explicit programming."
    ),
    "Partial overlap": (
        "Machine learning uses data to help computers get better at tasks."
    ),
    "Completely different": (
        "The weather in Mumbai today is hot and humid with a chance of rain."
    ),
    "Too verbose": (
        "Machine learning, which is widely considered to be one of the most "
        "important and transformative fields in the history of artificial intelligence, "
        "is a type of computational approach where algorithms enable computers to "
        "automatically learn from data and continuously improve their performance "
        "on specific tasks without requiring humans to explicitly program every single "
        "rule or step in the process."
    ),
}

print("="*70)
print("REFERENCE TEXT:")
print(f'"{REFERENCE}"')
print("="*70)

print(f"\n{'Candidate':<25} {'BLEU':>6} {'R1-F1':>6} {'RL-F1':>6}")
print("-"*50)

for name, candidate in CANDIDATES.items():
    bleu = bleu_score(REFERENCE, candidate)
    r1 = rouge_1(REFERENCE, candidate)
    rl = rouge_l(REFERENCE, candidate)
    print(f"  {name:<23} {bleu:>6.3f} {r1['f1']:>6.3f} {rl['f1']:>6.3f}")


# ── Using rouge-score library (more robust) ───────────────────────────────────
print("\n" + "="*70)
print("ROUGE SCORES (using rouge-score library):")
print("="*70)

try:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    for name, candidate in CANDIDATES.items():
        scores = scorer.score(REFERENCE, candidate)
        r1_f = scores["rouge1"].fmeasure
        r2_f = scores["rouge2"].fmeasure
        rl_f = scores["rougeL"].fmeasure
        print(f"  {name:<30} R1={r1_f:.3f}  R2={r2_f:.3f}  RL={rl_f:.3f}")

except ImportError:
    print("  (rouge-score not installed — run: pip3 install rouge-score)")


# ── BERTScore ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BERTSCORE — Semantic similarity using BERT embeddings:")
print("(Better than ROUGE — captures meaning, not just word overlap)")
print("="*70)

try:
    from bert_score import score as bert_score_fn

    refs = [REFERENCE] * len(CANDIDATES)
    hyps = list(CANDIDATES.values())

    P, R, F1 = bert_score_fn(hyps, refs, lang="en", verbose=False)

    for i, (name, _) in enumerate(CANDIDATES.items()):
        print(f"  {name:<30} BERTScore-F1 = {F1[i].item():.3f}")

except ImportError:
    print("  (bert-score not installed — run: pip3 install bert-score)")
    print("  BERTScore would capture semantic similarity even for paraphrases.")


# ── Interpretation Guide ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("INTERPRETATION GUIDE:")
print("="*70)
print("""
  BLEU:
    > 0.6  → Excellent (near identical)
    0.4-0.6 → Good
    0.2-0.4 → Acceptable
    < 0.2  → Poor

  ROUGE-1 F1:
    > 0.5  → Good summarisation
    0.3-0.5 → Acceptable
    < 0.3  → Poor coverage

  BERTScore F1:
    > 0.9  → Near-perfect semantic match
    0.8-0.9 → Good semantic similarity
    0.7-0.8 → Partial overlap
    < 0.7  → Semantically different

  KEY INSIGHT: BERTScore > ROUGE > BLEU for capturing meaning.
  Use BLEU for exact-match tasks (translation), ROUGE for summarisation,
  BERTScore when paraphrasing is acceptable.
""")
