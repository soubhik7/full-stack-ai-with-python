# house_price_predictor.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create synthetic dataset
def create_dataset():
    np.random.seed(42)
    # House sizes (in square feet)
    house_sizes = np.random.randint(800, 4000, 100)
    # Prices = $100 * size + random noise
    prices = 100 * house_sizes + np.random.normal(0, 50000, 100)
    
    df = pd.DataFrame({
        'Size_sqft': house_sizes,
        'Price': prices
    })
    return df

# Step 2: Load and explore data
df = create_dataset()
print("Dataset Preview:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")

# Step 3: Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(df['Size_sqft'], df['Price'], alpha=0.7)
plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($)')
plt.title('House Prices vs Size')
plt.grid(True, alpha=0.3)
plt.savefig('data_visualization.png')
plt.show()

# Step 4: Prepare data for ML
X = df[['Size_sqft']]  # Features
y = df['Price']        # Target variable

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Step 5: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")
print(f"\nModel Equation: Price = {model.coef_[0]:.2f} * Size + {model.intercept_:.2f}")

# Step 8: Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.7, label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($)')
plt.title('Linear Regression: Actual vs Predicted Prices')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('predictions.png')
plt.show()

# Step 9: Make a prediction for a new house
new_house_size = [[2500]]  # 2500 sqft house
predicted_price = model.predict(new_house_size)[0]
print(f"\nPredicted price for 2500 sqft house: ${predicted_price:,.2f}")

# Step 10: Save the model
import joblib
joblib.dump(model, 'house_price_model.pkl')
print("Model saved as 'house_price_model.pkl'")