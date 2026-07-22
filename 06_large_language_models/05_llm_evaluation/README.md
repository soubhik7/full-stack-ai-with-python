# 05 — LLM Evaluation

> You can't improve what you can't measure. Learn to evaluate LLM output systematically.

---

## Why Evaluation Matters

You've built a chatbot. Users say "it's sometimes wrong." How do you know:
- Which prompts perform best?
- Whether your RAG system actually helped?
- If fine-tuning improved things?

**Answer: systematic evaluation.**

---

## Files

| File | What it covers |
|------|---------------|
| `01_text_metrics.py` | BLEU, ROUGE, BERTScore — classic NLP metrics |
| `02_llm_as_judge.py` | Use GPT-4 to evaluate GPT-4 output |
| `03_eval_pipeline.py` | Build a complete evaluation pipeline with golden datasets |

---

## Evaluation Techniques

### 1. Reference-Based Metrics (Classic NLP)

Compare model output to a **ground truth** answer.

| Metric | What it measures | Best for |
|--------|-----------------|---------|
| **BLEU** | N-gram overlap | Machine translation |
| **ROUGE-1** | Unigram recall | Summarisation |
| **ROUGE-L** | Longest common subsequence | Summarisation |
| **BERTScore** | Semantic similarity via embeddings | General text quality |

**Limitation:** Need reference answers. Bad at measuring fluency.

### 2. LLM-as-Judge

Use a powerful LLM (GPT-4, Claude) to evaluate another LLM's output.

```
System: "You are an expert evaluator. Rate this response 1-10 on helpfulness..."
User: "Question: X\nModel answer: Y\nReference: Z\nRate it."
```

**Pros:** Flexible, no need for exact reference, catches nuance  
**Cons:** Expensive, the judge LLM can be biased, may prefer verbose answers

### 3. Human Evaluation (Gold Standard)

Real humans rate outputs on specific dimensions:
- Helpfulness (1-5)
- Accuracy (1-5)  
- Coherence (1-5)
- Safety (pass/fail)

**When to use:** For final validation before production.

---

## Evaluation Dimensions

| Dimension | Question | Measurement |
|-----------|---------|-------------|
| **Correctness** | Is the answer factually right? | Reference match / LLM judge |
| **Helpfulness** | Does it address the user's need? | LLM judge |
| **Faithfulness** | (RAG) Is it grounded in retrieved docs? | LLM judge |
| **Conciseness** | Is it appropriately brief? | Length + LLM judge |
| **Safety** | Does it avoid harmful content? | Classifier |
| **Groundedness** | Does it avoid hallucination? | LLM judge |

---

## Run the Scripts

```bash
source venv/bin/activate
pip3 install rouge-score bert-score   # first time only
python 06_large_language_models/05_llm_evaluation/01_text_metrics.py
python 06_large_language_models/05_llm_evaluation/02_llm_as_judge.py
python 06_large_language_models/05_llm_evaluation/03_eval_pipeline.py
```

---

## Key Insight: The Evaluation Triangle

```
         Human Eval
        (accurate, slow)
            /\
           /  \
          /    \
         /      \
  Metrics ——————— LLM Judge
(cheap, fast)   (cheap, flexible)
```

Use all three in combination:
- **Metrics** → cheap regression testing in CI/CD
- **LLM judge** → nuanced quality during development
- **Human eval** → final validation before production
