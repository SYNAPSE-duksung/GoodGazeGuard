import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

# ============================================
# Load
# ============================================
feature_df = pd.read_csv("feature.csv")

print(feature_df.head())
print(feature_df.describe())
print(feature_df.isnull().sum())

# ============================================
# 사용할 Feature
# ============================================
feature_cols = [
    "movement_mean",
    "movement_cv",
    "movement_skew",

    "dispersion",

    "center_distance_std",

    "velocity_mean",
    "velocity_std",

    "fixation_mean_duration",
    "fixation_count",

    "hull_area"
]

features = feature_df[feature_cols].copy()

# ============================================
# Log Transform
# ============================================
for col in [
    "movement_mean",
    "dispersion",
    "velocity_mean",
    "velocity_std",
    "hull_area"
]:
    features[col] = np.log1p(features[col])

features = features.fillna(0)

# ============================================
# Standardization
# ============================================
scaler = StandardScaler()
X = scaler.fit_transform(features)

# ============================================
# KMeans
# ============================================
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

feature_df["cluster"] = kmeans.fit_predict(X)

score = silhouette_score(X, feature_df["cluster"])

print(f"\nSilhouette Score : {score:.3f}")

feature_df.to_csv(
    "feature_cluster.csv",
    index=False
)

# ============================================
# Cluster 정보
# ============================================
print("\nCluster Counts")
print(feature_df["cluster"].value_counts())

print("\nSequence Length")
print(
    pd.crosstab(
        feature_df["sequence_length"],
        feature_df["cluster"]
    )
)

print("\nTask + Sequence")
print(
    pd.crosstab(
        [feature_df["task"], feature_df["sequence_length"]],
        feature_df["cluster"],
        normalize="index"
    ).round(3)
)

print("\nTask")
print(
    pd.crosstab(
        feature_df["task"],
        feature_df["cluster"]
    )
)

# ============================================
# Cluster 평균
# ============================================
cluster_mean = feature_df.groupby("cluster")[feature_cols].mean()

print("\nCluster Mean")
print(cluster_mean)
# for k in range(2, 7):

#     km = KMeans(
#         n_clusters=k,
#         random_state=42,
#         n_init=10
#     )

#     label = km.fit_predict(X)

#     score = silhouette_score(X, label)

#     print(f"k={k}: {score:.3f}")

# ============================================
# PCA
# ============================================
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

print("\nExplained Variance")
print(pca.explained_variance_ratio_)
print("Total :", pca.explained_variance_ratio_.sum())

loading = pd.DataFrame(
    pca.components_.T,
    index=feature_cols,
    columns=["PC1", "PC2"]
)

print("\nPCA Loadings")
print(loading)

print("\nAbsolute Loadings")
print(
    loading.abs().sort_values(
        "PC1",
        ascending=False
    )
)

# ============================================
# PCA Plot
# ============================================
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=feature_df["cluster"],
    cmap="viridis",
    s=20
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KMeans Clustering")

plt.colorbar(label="Cluster")

plt.tight_layout()
plt.show()