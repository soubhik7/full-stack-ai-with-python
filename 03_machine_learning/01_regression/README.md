# 03.01 — Regression

> Predict a continuous output value from input features.

## Notebooks (run in order)

| # | Notebook | Algorithm | Complexity |
|---|----------|-----------|-----------|
| 1 | `regression_simple_linear_co2.ipynb` | Simple Linear Regression | One feature |
| 2 | `regression_multiple_linear_co2.ipynb` | Multiple Linear Regression | Many features |
| 3 | `regression_polynomial_co2.ipynb` | Polynomial Regression | Non-linear with `PolynomialFeatures` |
| 4 | `regression_nonlinear.ipynb` | Non-linear Regression | Sigmoid curve fitting with `scipy.optimize` |

All four use the **FuelConsumption.csv** dataset — CO₂ emissions from cars.

## Key sklearn APIs

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
```

← Back to [Chapter 03](../README.md) · Next: [02_classification →](../02_classification/README.md)
