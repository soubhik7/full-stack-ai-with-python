# Chapter 03 — Classical Machine Learning

> **You will build intuition for the four core ML task types — regression, classification, clustering, and recommendation — using scikit-learn on real datasets.**

---

## Prerequisites

- [Chapter 01 — Python Fundamentals](../01_python/README.md)
- [Chapter 02 — Math for ML](../02_math_for_ml/README.md) *(linear algebra & statistics sections)*
- Packages: `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`

---

## Learning Path

Work through the sub-chapters **in order**. Each one introduces a new task type.

```
01_regression/         ← Predict a continuous number
02_classification/     ← Predict a category
03_clustering/         ← Find hidden groups (no labels)
04_recommendation/     ← Personalise suggestions
```

---

## Sub-chapter Breakdown

### `01_regression/` — Predicting Continuous Values

All four notebooks use **CO₂ emissions data** — same dataset, increasing model complexity.

| Notebook | Algorithm | Key Concept |
|----------|-----------|-------------|
| `regression_simple_linear_co2.ipynb` | Simple Linear Regression | One feature → one output |
| `regression_multiple_linear_co2.ipynb` | Multiple Linear Regression | Many features → one output |
| `regression_polynomial_co2.ipynb` | Polynomial Regression | Non-linear curves with linear algebra |
| `regression_nonlinear.ipynb` | Non-linear Regression | Sigmoid / logistic curve fitting |

---

### `02_classification/` — Predicting Categories

Four algorithms on different datasets, then MNIST to stress-test performance.

| Notebook | Algorithm | Dataset |
|----------|-----------|---------|
| `classification_logistic_regression_churn.ipynb` | Logistic Regression | Telecom churn |
| `classification_knn_customer_categories.ipynb` | K-Nearest Neighbours | Customer segmentation |
| `classification_decision_trees_drug.ipynb` | Decision Tree | Drug prescription |
| `classification_svm_cancer.ipynb` | Support Vector Machine | Breast cancer diagnosis |
| `01_mnist_beginner_explanation.ipynb` | Multi-class classification | MNIST digits — conceptual |
| `02_mnist_performance_evaluation.ipynb` | Evaluation metrics | MNIST — precision, recall, F1, confusion matrix |

---

### `03_clustering/` — Unsupervised Grouping

Three algorithms, each suited to a different cluster shape.

| Notebook | Algorithm | Dataset | Best for |
|----------|-----------|---------|---------|
| `clustering_kmeans_customer_segmentation.ipynb` | K-Means | Customer data | Spherical clusters, known k |
| `clustering_hierarchical_cars.ipynb` | Hierarchical (Ward) | Cars | Dendrogram, unknown k |
| `clustering_dbscan_weather.ipynb` | DBSCAN | Weather stations | Arbitrary shapes, noise handling |

---

### `04_recommendation/` — Personalised Suggestions

Two complementary recommendation strategies on a movie dataset.

| Notebook | Algorithm | What it needs |
|----------|-----------|---------------|
| `recsys_content_based_movies.ipynb` | Content-Based Filtering | Item features (genre, cast, …) |
| `recsys_collaborative_filtering_movies.ipynb` | Collaborative Filtering | User-item interaction matrix |

---

## Data

Training datasets live in `data/training_data/`:

| File | Used in |
|------|---------|
| `FuelConsumption.csv` | Regression notebooks |
| `china_gdp.csv` | Nonlinear regression |
| `drug200.csv` | Decision tree classification |

---

## Key scikit-learn Concepts

By the end of this chapter you will know how to:

- Split data with `train_test_split`
- Fit a model with `.fit(X_train, y_train)`
- Predict with `.predict(X_test)`
- Evaluate with `mean_squared_error`, `r2_score`, `accuracy_score`, `classification_report`
- Cross-validate with `cross_val_score`
- Visualise with `confusion_matrix`, dendrograms, elbow curves

---

## Next Step

Head to **[Chapter 04 — Deep Learning](../04_deep_learning/README.md)** →
