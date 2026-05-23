# house_price_classifier.py

# ===============================
# IMPORT REQUIRED LIBRARIES
# ===============================

import numpy as np                  # Used for numerical operations and random number generation
import pandas as pd                 # Used for structured data handling (DataFrames)
import matplotlib.pyplot as plt     # Used for data visualization (plots and graphs)

from sklearn.model_selection import train_test_split   # Splits dataset into training and testing sets
from sklearn.linear_model import LogisticRegression    # Logistic Regression classification model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Classification evaluation metrics

import joblib                       # Used to save and load trained ML models


# ===============================
# STEP 1: CREATE SYNTHETIC DATASET
# ===============================

def create_dataset():
    np.random.seed(42)  # Ensures reproducibility (same random data every run)

    # Generate 100 random house sizes between 800 and 4000 sqft
    house_sizes = np.random.randint(800, 4000, 100)

    # Generate house prices using formula:
    # Price = 100 * size + random noise
    # Noise simulates real-world market variation
    prices = 100 * house_sizes + np.random.normal(0, 50000, 100)

    # Create a structured DataFrame
    df = pd.DataFrame({
        'Size_sqft': house_sizes,   # Feature (input variable)
        'Price': prices             # Continuous output
    })

    return df


# ===============================
# STEP 2: LOAD AND EXPLORE DATA
# ===============================

df = create_dataset()

print("Dataset Preview:")
print(df.head())

print(f"\nDataset Shape: {df.shape}")
# Shows (number_of_rows, number_of_columns)


# ===============================
# STEP 3: CONVERT TO CLASSIFICATION PROBLEM
# ===============================

# Logistic Regression works only for classification.
# So we convert continuous Price into binary category.

# Use median price as threshold
median_price = df['Price'].median()

# If price > median → 1 (Expensive)
# If price <= median → 0 (Not Expensive)
df['Expensive'] = (df['Price'] > median_price).astype(int)

print(f"\nMedian price threshold: {median_price:.2f}")


# ===============================
# STEP 4: PREPARE DATA FOR ML
# ===============================

# X = Feature(s) → Input variable(s)
X = df[['Size_sqft']]   # Must remain 2D

# y = Target → Classification label (0 or 1)
y = df['Expensive']


# ===============================
# SPLIT DATA INTO TRAIN & TEST
# ===============================

# 80% Training
# 20% Testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# ===============================
# STEP 5: CREATE & TRAIN MODEL
# ===============================

# Initialize Logistic Regression model
model = LogisticRegression()

# Train the model
# Instead of fitting a straight line,
# Logistic Regression learns a probability curve (sigmoid function)
model.fit(X_train, y_train)

print("\nModel trained successfully.")


# ===============================
# STEP 6: MAKE PREDICTIONS
# ===============================

# Predict class labels (0 or 1)
y_pred = model.predict(X_test)


# ===============================
# STEP 7: EVALUATE MODEL
# ===============================

# Accuracy → Percentage of correct predictions
accuracy = accuracy_score(y_test, y_pred)

# Confusion Matrix →
# [ [True Negatives, False Positives],
#   [False Negatives, True Positives] ]
cm = confusion_matrix(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.2f}")

print("\nConfusion Matrix:")
print(cm)

# Detailed metrics:
# Precision → How many predicted positives were correct?
# Recall → How many actual positives were detected?
# F1-score → Balance between precision and recall
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ===============================
# STEP 8: VISUALIZE CLASSIFICATION
# ===============================

plt.figure(figsize=(10, 6))

# Scatter plot of test data
plt.scatter(X_test, y_test, alpha=0.7)

plt.xlabel('House Size (sqft)')
plt.ylabel('Class (0 = Not Expensive, 1 = Expensive)')
plt.title('Logistic Regression Classification')

plt.grid(True, alpha=0.3)
plt.show()


# ===============================
# STEP 9: PREDICT NEW SAMPLE
# ===============================

# Input must be 2D → [[value]]
new_house_size = [[2500]]

# Predict class
predicted_class = model.predict(new_house_size)[0]

# Predict probability of being Expensive (class 1)
predicted_probability = model.predict_proba(new_house_size)[0][1]

print(f"\nFor 2500 sqft house:")
print(f"Probability of being Expensive: {predicted_probability:.2f}")
print(f"Predicted Class: {predicted_class}")


# ===============================
# STEP 10: SAVE THE MODEL
# ===============================

joblib.dump(model, 'house_price_classifier.pkl')

print("Model saved as 'house_price_classifier.pkl'")
