# house_price_kmeans.py

# ===============================
# IMPORT REQUIRED LIBRARIES
# ===============================

import numpy as np                  # Numerical operations
import pandas as pd                 # Structured data handling
import matplotlib.pyplot as plt     # Visualization

from sklearn.cluster import KMeans  # K-Means clustering algorithm
from sklearn.preprocessing import StandardScaler  # Feature scaling
from sklearn.metrics import silhouette_score      # Cluster quality metric


# ===============================
# STEP 1: CREATE SYNTHETIC DATASET
# ===============================

def create_dataset():
    np.random.seed(42)  # Reproducibility

    # Generate 100 house sizes (800–4000 sqft)
    house_sizes = np.random.randint(800, 4000, 100)

    # Generate prices with noise
    prices = 100 * house_sizes + np.random.normal(0, 50000, 100)

    df = pd.DataFrame({
        'Size_sqft': house_sizes,
        'Price': prices
    })

    return df


# ===============================
# STEP 2: LOAD DATA
# ===============================

df = create_dataset()

print("Dataset Preview:")
print(df.head())

print(f"\nDataset Shape: {df.shape}")


# ===============================
# STEP 3: PREPARE FEATURES
# ===============================

# K-Means works with numerical features only.
# We use both Size and Price for clustering.

X = df[['Size_sqft', 'Price']]

# Feature scaling is IMPORTANT for K-Means
# Because it is distance-based (Euclidean distance).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ===============================
# STEP 4: APPLY K-MEANS
# ===============================

# Define number of clusters (k=2)
kmeans = KMeans(n_clusters=2, random_state=42)

# Fit the model
kmeans.fit(X_scaled)

# Get cluster labels for each house
df['Cluster'] = kmeans.labels_

print("\nCluster Centers (scaled values):")
print(kmeans.cluster_centers_)


# ===============================
# STEP 5: EVALUATE CLUSTER QUALITY
# ===============================

# Silhouette Score measures how well clusters are separated
sil_score = silhouette_score(X_scaled, df['Cluster'])

print(f"\nSilhouette Score: {sil_score:.4f}")
# Closer to 1 → better clustering
# Around 0 → overlapping clusters


# ===============================
# STEP 6: VISUALIZE CLUSTERS
# ===============================

plt.figure(figsize=(10, 6))

# Scatter plot with color by cluster
plt.scatter(df['Size_sqft'], df['Price'],
            c=df['Cluster'],
            cmap='viridis',
            alpha=0.7)

plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($)')
plt.title('K-Means Clustering of Houses (k=2)')
plt.grid(True, alpha=0.3)

plt.show()


# ===============================
# STEP 7: ELBOW METHOD (Optional)
# ===============================

# Used to determine optimal number of clusters

inertia_values = []

for k in range(1, 8):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia_values.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 8), inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True, alpha=0.3)
plt.show()
