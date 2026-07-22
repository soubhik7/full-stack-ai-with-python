# Chapter 09 — Capstone Projects

> **Apply everything you've learned. Each project is a complete, self-contained ML solution — from raw data to a working system.**

---

## Prerequisites

- Complete chapters 01–08 (or the chapters relevant to each project)

---

## Projects

### `01_house_price_ml/` — House Price Prediction
**Domain:** Regression & Clustering · **Chapters used:** 03

Four progressively refined approaches to the same problem.

| Sub-project | Algorithm | What it adds |
|-------------|-----------|-------------|
| `01_linear_regression/` | Linear regression | Baseline model |
| `02_nonlinear_regression/` | Polynomial regression | Captures non-linear price curves |
| `03_kmeans/` | K-Means clustering | Segment neighbourhoods |
| `04_hierarchical/` | Hierarchical clustering | Dendrogram-based segmentation |

**Key file:** `ml_tutorial.ipynb` — full annotated walkthrough.

---

### `02_sentiment_analysis/` — Amazon Alexa Reviews
**Domain:** NLP Classification · **Chapters used:** 03, 05

End-to-end sentiment classifier on real customer reviews.

| File | Role |
|------|------|
| `sentiment_model_training.ipynb` | EDA → feature engineering → model training → evaluation |
| `app.py` | Flask/FastAPI serving endpoint |
| `custom_tokenizer_function.py` | Domain-specific tokeniser |

**Techniques:** TF-IDF, sklearn pipeline, custom tokeniser, `Pipeline` with `ColumnTransformer`.

---

### `03_gmail_assistant/` — AI Gmail Summariser
**Domain:** Text Summarisation · **Chapters used:** 06

Two flavours of a Gmail summarisation system.

| Sub-project | Approach |
|-------------|---------|
| `builtin_model/` | Pre-trained HuggingFace summarisation model |
| `custom_model/` | Fine-tuned custom summarisation model |

**Key files:**
- `auth.py` — Gmail OAuth2 authentication
- `summarize.py` — Summarisation pipeline
- `email_summary_training.ipynb` — Training the custom model

---

### `04_made_with_ml/` — End-to-End MLOps
**Domain:** MLOps · **Chapters used:** 03, 04, 06

Production ML system following the [Made With ML](https://madewithml.com) curriculum.

**Stack:** Python · Ray · MLflow · Anyscale · pytest · pre-commit

```bash
cd 09_projects/04_made_with_ml
pip3 install -r requirements.txt   # project-local deps
make train                         # train with Ray
make evaluate
make serve
```

This project has its own `pyproject.toml`, `Makefile`, and internal Python package structure — treat it as a standalone repo.

---

### `05_great_learning_certification/` — AIML Certification Projects
**Domain:** End-to-end across all domains · **Chapters used:** 02–06

Eight graded projects from the Great Learning AIML certification covering the full spectrum.

| # | Project | Domain |
|---|---------|--------|
| 01 | `applied_statistics_project.ipynb` | Exploratory data analysis, hypothesis testing |
| 02 | `supervised_learning_classification_project.ipynb` | Classification (sklearn) |
| 03 | `ensemble_techniques_project.ipynb` | Random forests, gradient boosting, stacking |
| 04 | `unsupervised_learning_pca_project.ipynb` | PCA + K-Means |
| 05 | `featurization_model_selection_project.ipynb` | Feature engineering, cross-validation |
| 06 | `neural_network_project.ipynb` | Deep neural network with Keras |
| 07 | `face_detection_project.ipynb` + `face_recognition_project_aws.ipynb` | Computer vision, OpenCV, AWS Rekognition |
| 08 | `nlp_project_sarcasm_detection.ipynb` + `nlp_project_sequential.ipynb` | NLP classification |

---

## How to Approach a Project

1. **Read the notebook top to bottom** before running any cell — understand the goal
2. **Run the EDA cells** — understand the data before modelling
3. **Try changing one thing** — swap an algorithm, change a hyperparameter
4. **Write your own conclusions** — what would you do differently in production?

---

## You've completed the curriculum 🎉

Go back to **[reference/](../reference/README.md)** for a comprehensive AI/ML reference, or start building your own project using this repo as a template.
