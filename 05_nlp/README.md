# Chapter 05 — Natural Language Processing

> **You will learn to take raw, messy text and transform it into structured representations that machine learning models can use.**

---

## Prerequisites

- [Chapter 03 — Classical Machine Learning](../03_machine_learning/README.md)
- Packages: `nltk`, `spacy`, `re`, `string`

---

## Learning Path

```
01_preprocessing/   ← Clean and normalise raw text
02_tagging/         ← Label words with linguistic roles
03_practice/        ← Combine preprocessing + tagging end-to-end
```

---

## Sub-chapter Breakdown

### `01_preprocessing/` — Cleaning Text

Every NLP pipeline starts with cleaning. Work through these five notebooks **in order** — each step feeds into the next.

| # | Notebook | Technique | What it does |
|---|----------|-----------|-------------|
| 01 | `01_nlp_cleaning.ipynb` | Text cleaning | Lowercase, remove punctuation, HTML tags, special chars |
| 02 | `02_nlp_tokenization.ipynb` | Tokenisation | Split text into words and sentences |
| 03 | `03_nlp_stemming.ipynb` | Stemming | Reduce words to root form (Porter, Lancaster) |
| 04 | `04_nlp_lemmatization.ipynb` | Lemmatisation | Context-aware root form (WordNet) |
| 05 | `05_nlp_ngrams.ipynb` | N-grams | Unigram / bigram / trigram language models |

**Key insight:** Stemming is fast but lossy ("running" → "run", "studies" → "studi"). Lemmatisation is slower but accurate ("studies" → "study").

---

### `02_tagging/` — Linguistic Annotations

Attach grammatical and semantic labels to tokens.

| Notebook | Task | Library |
|----------|------|---------|
| `01_pos_tagging.ipynb` | Part-of-Speech tagging | NLTK / spaCy |
| `02_named_entity_recognition.ipynb` | Named Entity Recognition (NER) | spaCy `en_core_web_sm` |

**POS tags:** NN (noun), VB (verb), JJ (adjective), RB (adverb), …
**NER labels:** PERSON, ORG, GPE, DATE, MONEY, …

---

### `03_practice/` — End-to-End

Apply the full pipeline on real datasets.

| Notebook | What it covers |
|----------|---------------|
| `data_pre_processing.ipynb` | Full preprocessing pipeline on a news dataset |
| `data_tagging_process.ipynb` | POS + NER on the same dataset |

---

## The Full NLP Pipeline (Mental Model)

```
Raw text
  → Clean  (lowercase, remove noise)
  → Tokenise  (split into words)
  → Normalise  (stem or lemmatise)
  → Tag  (POS, NER)
  → Features  (TF-IDF, word vectors, …)   ← Chapter 06 onwards
  → Model
```

---

## Quick NLTK Setup

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## Quick spaCy Setup

```bash
python -m spacy download en_core_web_sm
```

---

## Next Step

Head to **[Chapter 06 — Large Language Models](../06_large_language_models/README.md)** →
