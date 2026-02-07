# house_price_predictor.py

# ===============================
# IMPORT REQUIRED LIBRARIES
# ===============================

import numpy as np                  # Used for numerical operations and random number generation
import pandas as pd                 # Used for handling structured data (DataFrames)
import matplotlib.pyplot as plt     # Used for data visualization (plots and graphs)

from sklearn.model_selection import train_test_split   # Used to split dataset into training and testing sets
from sklearn.linear_model import LinearRegression      # Linear Regression model
from sklearn.metrics import mean_squared_error, r2_score  # Model evaluation metrics


# ===============================
# STEP 1: CREATE SYNTHETIC DATASET
# ===============================

def create_dataset():
    np.random.seed(42)  # Ensures the same random data is generated every time (reproducibility)

    # Generate 100 random house sizes between 800 and 4000 square feet
    house_sizes = np.random.randint(800, 4000, 100)

    # Generate house prices using formula:
    # Price = 100 * size + random noise
    # Noise simulates real-world price variation
    prices = 100 * house_sizes + np.random.normal(0, 50000, 100)
    
    # Create a pandas DataFrame (table structure)
    df = pd.DataFrame({
        'Size_sqft': house_sizes,  # Feature (input variable)
        'Price': prices            # Target (output variable)
    })

    return df


# ===============================
# STEP 2: LOAD AND EXPLORE DATA
# ===============================

df = create_dataset()  # Create the dataset

print("Dataset Preview:")
print(df.head())       # Display first 5 rows

print(f"\nDataset Shape: {df.shape}")  
# Shows (number_of_rows, number_of_columns)


# ===============================
# STEP 3: VISUALIZE THE DATA
# ===============================

plt.figure(figsize=(10, 6))

# Scatter plot: Each dot represents one house
plt.scatter(df['Size_sqft'], df['Price'], alpha=0.7)

plt.xlabel('House Size (sqft)')   # X-axis label
plt.ylabel('Price ($)')           # Y-axis label
plt.title('House Prices vs Size') # Plot title

plt.grid(True, alpha=0.3)          # Adds light grid

plt.savefig('data_visualization.png')  # Save plot as image
plt.show()                         # Display plot


# ===============================
# STEP 4: PREPARE DATA FOR ML
# ===============================

# X = Feature(s) → input variable(s)
X = df[['Size_sqft']]  # Must be 2D (DataFrame format)

# y = Target → output variable
y = df['Price']


# ===============================
# SPLIT DATA INTO TRAIN & TEST
# ===============================

# 80% Training data
# 20% Testing data
# random_state ensures reproducibility

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# ===============================
# STEP 5: CREATE & TRAIN MODEL
# ===============================

# Initialize Linear Regression model
model = LinearRegression()

# Train the model using training data
# The model learns the best-fit line:
# Price = (coefficient * Size) + intercept
model.fit(X_train, y_train)


# ===============================
# STEP 6: MAKE PREDICTIONS
# ===============================

# Predict house prices for test data
y_pred = model.predict(X_test)


# ===============================
# STEP 7: EVALUATE MODEL
# ===============================

# Mean Squared Error (MSE)
# Measures average squared difference between actual and predicted values
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error (RMSE)
# Square root of MSE (same unit as price)
rmse = np.sqrt(mse)

# R-squared (R²)
# Measures how well model explains variance (0 to 1)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Display learned model equation
print(f"\nModel Equation: Price = {model.coef_[0]:.2f} * Size + {model.intercept_:.2f}")


# ===============================
# STEP 8: VISUALIZE PREDICTIONS
# ===============================

plt.figure(figsize=(10, 6))

# Actual values (blue dots)
plt.scatter(X_test, y_test, alpha=0.7, label='Actual Prices')

# Predicted regression line (red line)
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')

plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($)')
plt.title('Linear Regression: Actual vs Predicted Prices')

plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('predictions.png')
plt.show()


# ===============================
# STEP 9: PREDICT NEW HOUSE PRICE
# ===============================

# New input must be 2D array → [[value]]
new_house_size = [[2500]]  # 2500 sqft house

# Predict price
predicted_price = model.predict(new_house_size)[0]

print(f"\nPredicted price for 2500 sqft house: ${predicted_price:,.2f}")


# ===============================
# STEP 10: SAVE THE MODEL
# ===============================

import joblib  # Used for saving and loading ML models

# Save trained model to file
joblib.dump(model, 'house_price_model.pkl')

print("Model saved as 'house_price_model.pkl'")