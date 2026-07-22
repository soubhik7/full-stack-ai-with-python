# Chapter 06 — Large Language Models

> **You will go from using HuggingFace pipelines, to building a GPT-style model from scratch in PyTorch, to fine-tuning it on custom data.**

---

## Prerequisites

- [Chapter 04 — Deep Learning](../04_deep_learning/README.md) *(transformers build on deep learning)*
- [Chapter 05 — NLP](../05_nlp/README.md) *(tokenisation, text representation)*
- Packages: `transformers`, `torch`, `datasets`, `sentence-transformers`, `huggingface-hub`

---

## Learning Path

```
01_transformers_and_huggingface/   ← Use pre-trained models (pipelines)
02_llm_from_scratch/               ← Build a GPT from first principles
03_llm_fine_tuning/                ← Adapt a model to your own data
04_prompt_engineering/             ← Master system prompts, CoT, structured output
05_llm_evaluation/                 ← BLEU/ROUGE/BERTScore, LLM-as-judge, eval pipelines
```

---

## Sub-chapter Breakdown

### `01_transformers_and_huggingface/` — Pre-trained Models

The fastest way to use state-of-the-art models — no training required.

| Notebook | What you learn |
|----------|---------------|
| `text_pipelines.ipynb` | Sentiment analysis, summarisation, translation, Q&A |
| `image_pipelines.ipynb` | Image classification, object detection, image captioning |
| `image_data.ipynb` | Loading and preprocessing image datasets |
| `pytorch_and_tensorflow.ipynb` | Same pipeline in both backends — compare APIs |
| `fine_tuning.ipynb` | Fine-tune a BERT model on a classification task |

**Key HuggingFace concepts:**
- `pipeline()` — one-line inference
- `AutoTokenizer` / `AutoModel` — architecture-agnostic loading
- `Trainer` API — training loop abstraction
- `datasets` library — standardised data loading

---

### `02_llm_from_scratch/` — Build a GPT in PyTorch

Understand every component of a transformer LLM before using a library.

| Notebook | Content |
|----------|---------|
| `core_model.ipynb` | Tokeniser, embedding, positional encoding, attention head |
| `model.ipynb` | Multi-head attention, feed-forward, layer norm, full block |
| `model2.ipynb` | Stacked blocks, training loop, text generation |

📄 **Reference PDF:** `llm_from_scratch_pytorch.pdf` — full written guide.

**Architecture components covered:**
- Byte-pair encoding tokeniser
- Token + positional embeddings
- Scaled dot-product attention
- Multi-head self-attention
- Transformer block (attention → FFN → residual → norm)
- Causal language modelling objective
- Greedy & temperature sampling

---

### `03_llm_fine_tuning/` — Custom Model Training

A three-stage pipeline for training a domain-specific model.

| Script | Stage | What it does |
|--------|-------|-------------|
| `01_collect_and_clean.py` | Data | Scrape, clean, and format training text |
| `02_train.py` | Training | Fine-tune on cleaned data with HuggingFace `Trainer` |
| `03_evaluate.py` | Evaluation | Perplexity, generation quality metrics |
| `example_script.py` | Demo | End-to-end example you can run immediately |

---

## Transformer Architecture — Quick Reference

```
Input tokens
  → Token Embedding  +  Positional Encoding
  → [ Transformer Block ] × N
      ├─ Multi-Head Self-Attention
      ├─ Add & Norm
      ├─ Feed-Forward Network
      └─ Add & Norm
  → Language Model Head (Linear → Softmax)
  → Next-token probabilities
```

---

---

### `04_prompt_engineering/` — Advanced Prompt Engineering *(NEW)*

Master the full prompt engineering toolkit used by professional AI engineers.

| File | Technique |
|------|-----------|
| `01_system_prompts.py` | Persona, constraints, guard-rails |
| `02_advanced_few_shot.py` | Static, format, dynamic, contrastive, CoT few-shot |
| `03_chain_of_thought.py` | Zero-shot CoT, self-consistency, step-back, ReAct |
| `04_structured_output.py` | JSON mode, Pydantic `.parse()`, nested schemas |
| `05_prompt_chaining.py` | Linear chains, conditional routing, iterative refinement |

---

### `05_llm_evaluation/` — LLM Evaluation *(NEW)*

Measure and compare LLM output quality systematically.

| File | What it covers |
|------|---------------|
| `01_text_metrics.py` | BLEU, ROUGE-1, ROUGE-L, BERTScore — from scratch + libraries |
| `02_llm_as_judge.py` | GPT-4 as evaluator: grading, pairwise, rubric, RAG faithfulness |
| `03_eval_pipeline.py` | Full A/B pipeline: golden dataset → scores → report → comparison |

```bash
pip3 install rouge-score bert-score   # first-time only
python 06_large_language_models/05_llm_evaluation/03_eval_pipeline.py
```

---

## Next Step

Head to **[Chapter 07 — RAG](../07_rag/README.md)** →
