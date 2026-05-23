# house_price_hierarchical.py

# ===============================
# IMPORT REQUIRED LIBRARIES
# ===============================

import numpy as np                      # Numerical operations
import pandas as pd                     # Data handling
import matplotlib.pyplot as plt         # Visualization

from sklearn.preprocessing import StandardScaler          # Feature scaling
from sklearn.cluster import AgglomerativeClustering      # Hierarchical clustering
from sklearn.metrics import silhouette_score             # Cluster quality metric

from scipy.cluster.hierarchy import dendrogram, linkage  # Dendrogram creation


# ===============================
# STEP 1: CREATE SYNTHETIC DATASET
# ===============================

def create_dataset():
    np.random.seed(42)  # Reproducibility

    # Generate house sizes (800–4000 sqft)
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

# Use both Size and Price for clustering
X = df[['Size_sqft', 'Price']]

# Scale features (very important for distance-based algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ===============================
# STEP 4: CREATE DENDROGRAM
# ===============================

# Linkage matrix (Ward minimizes variance within clusters)
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.grid(True, alpha=0.3)
plt.show()


# ===============================
# STEP 5: APPLY AGGLOMERATIVE CLUSTERING
# ===============================

# Define number of clusters (e.g., 2)
hierarchical_model = AgglomerativeClustering(
    n_clusters=2,
    linkage='ward'
)

# Assign cluster labels
df['Cluster'] = hierarchical_model.fit_predict(X_scaled)

print("\nCluster distribution:")
print(df['Cluster'].value_counts())


# ===============================
# STEP 6: EVALUATE CLUSTER QUALITY
# ===============================

sil_score = silhouette_score(X_scaled, df['Cluster'])

print(f"\nSilhouette Score: {sil_score:.4f}")
# Closer to 1 → better separation


# ===============================
# STEP 7: VISUALIZE CLUSTERS
# ===============================

plt.figure(figsize=(10, 6))

plt.scatter(df['Size_sqft'],
            df['Price'],
            c=df['Cluster'],
            cmap='viridis',
            alpha=0.7)

plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($)')
plt.title('Hierarchical Clustering (Agglomerative, k=2)')
plt.grid(True, alpha=0.3)

plt.show()
