# Chapter 04 — Deep Learning

> **You will build neural networks from a single artificial neuron all the way to deep multi-layer architectures — first by hand, then with PyBrain, sklearn, TensorFlow, and PyTorch.**

---

## Prerequisites

- [Chapter 03 — Classical Machine Learning](../03_machine_learning/README.md)
- Packages: `numpy`, `matplotlib`, `scikit-learn`, `tensorflow`, `torch`

---

## Learning Path

```
01_perceptron/               ← One neuron, manually
02_multilayer_perceptron/    ← Stack neurons into layers
03_neural_network_libraries/ ← Same networks with real frameworks
04_nn_from_scratch/          ← Backprop coded from scratch
05_coursera_assignments/     ← Andrew Ng's graded assignments
```

---

## Sub-chapter Breakdown

### `01_perceptron/` — The Building Block

Understand a single artificial neuron before touching any library.

| Notebook | Concept |
|----------|---------|
| `01_perceptron_basics.ipynb` | What is a perceptron? Weights, bias, activation |
| `02_perceptron_advanced.ipynb` | Learning rule, weight updates |
| `03_perceptron_and_operator.ipynb` | Learn AND from data |
| `04_perceptron_or_operator.ipynb` | Learn OR from data |
| `05_perceptron_xor_operator.ipynb` | Why a single neuron can't learn XOR |
| `06_homework_salary_increase.ipynb` | Regression with a perceptron |

📄 **Additional reading** in `additional_reading/`: Rosenblatt 1957 paper + McCulloch & Pitts.

---

### `02_multilayer_perceptron/` — Stacking Neurons

Add hidden layers to solve non-linear problems.

| Notebook | Dataset | What changes |
|----------|---------|-------------|
| `01_multilayer_perceptron.ipynb` | XOR / toy | Two-layer network solves XOR |
| `02_multilayer_perceptron_iris.ipynb` | Iris | 4-class classification |
| `03_multilayer_perceptron_credit.ipynb` | Credit | Real-world tabular data |
| `03_multilayer_perceptron_credit_updated.ipynb` | Credit | Improved preprocessing |
| `04_three_layer_neural_network.ipynb` | Mixed | Two hidden layers |

---

### `03_neural_network_libraries/` — Framework Comparison

Same problems, three different frameworks — see what each one offers.

| Notebooks | Framework | Style |
|-----------|-----------|-------|
| `01_pybrain_basics.ipynb` + `02_pybrain_advanced.ipynb` | PyBrain | Legacy, educational |
| `04_sklearn_basics.ipynb` + `05_sklearn_advanced.ipynb` | scikit-learn | `MLPClassifier` |
| `07_tensorflow_basics.ipynb` | TensorFlow/Keras | `Sequential` API |
| `09_pytorch_breast_cancer.ipynb` + `10_pytorch_diabetes.ipynb` | PyTorch | Manual `nn.Module` |
| `03_homework_pybrain_iris.ipynb` · `06_homework_sklearn_wine.ipynb` · `08_homework_tensorflow_fashion_mnist.ipynb` | — | Homework solutions |

---

### `04_nn_from_scratch/` — No Library, No Shortcuts

Implement forward pass, backpropagation, and gradient descent entirely in NumPy.

| File | Content |
|------|---------|
| `nn_from_scratch.ipynb` | Step-by-step notebook implementation |
| `nn_from_scratch.py` | Clean Python module |
| `ann_classification.py` | Binary classification example |
| `simple_classification.py` | Minimal working example |

---

### `05_coursera_assignments/` — Andrew Ng Deep Learning Specialisation

Graded assignments from Course 1 (Neural Networks and Deep Learning) and Course 2 (Improving Deep Neural Networks).

| Assignment | Topic |
|------------|-------|
| `W2A1/python_basics_numpy.ipynb` | NumPy broadcasting, vectorisation |
| `W2A2/logistic_regression_neural_network.ipynb` | Logistic regression as a 1-layer NN |
| `W3A1/planar_data_classification.ipynb` | 1-hidden-layer network |
| `W4A1/deep_neural_network_step_by_step.ipynb` | L-layer forward/backprop |
| `W4A2/deep_neural_network_application.ipynb` | Cat vs non-cat image classifier |
| `W5A1/initialization.ipynb` | Weight initialisation strategies |
| `W5A2/regularization.ipynb` | L2 regularisation, dropout |
| `W5A3/gradient_checking.ipynb` | Numerical gradient checking |

> ⚠️ The `*.py` files inside each W*/ folder are utility helpers — **do not rename or move them** or the notebooks will break.

**Standalone notes** at chapter root:
- `00_params_vs_hyperparams.ipynb` — parameters vs hyperparameters
- `01_train_dev_test_sets.ipynb` — data splits
- `02_bias_variance.ipynb` — bias/variance tradeoff
- `03_regularization.ipynb` — regularisation theory
- `neural_networks_case_study.ipynb` — spiral dataset end-to-end

---

## Next Step

Head to **[Chapter 05 — NLP](../05_nlp/README.md)** →
