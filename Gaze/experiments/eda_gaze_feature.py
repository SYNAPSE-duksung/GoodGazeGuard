import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# ==========================
# Load feature
# ==========================
feature_df = pd.read_csv("feature.csv")

print("=" * 50)
print("Feature Preview")
print("=" * 50)
print(feature_df.head())

print("\nShape :", feature_df.shape)

print("\nMissing Values")
print(feature_df.isna().sum())

# ==========================
# Correlation
# ==========================
corr = feature_df.corr(numeric_only=True)

print("\nCorrelation with movement_mean")
print(
    corr["movement_mean"]
    .sort_values(ascending=False)
)

# ==========================
# Heatmap
# ==========================
plt.figure(figsize=(15, 12))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation")
plt.tight_layout()
plt.show()

# ==========================
# Feature Distribution
# ==========================
feature_df.hist(
    figsize=(18, 15),
    bins=30
)

plt.tight_layout()
plt.show()

# ==========================
# Boxplots
# ==========================

features = [
    "movement_mean",
    "gaze_dispersion",
    "gaze_velocity_mean",
    "fixation_mean_duration"
]
for feature in features:
    plt.figure(figsize=(6,4))

    sns.boxplot(
        data=feature_df,
        x="task",
        y=feature,
        showfliers=False
    )

    plt.title(feature)

    plt.show()

# ==========================
# ANOVA
# ==========================

print("\n" + "=" * 50)
print("ANOVA")
print("=" * 50)

for feature in features:

    low = feature_df.loc[
        feature_df["sequence_length"] == 5,
        feature
    ]

    mid = feature_df.loc[
        feature_df["sequence_length"] == 9,
        feature
    ]

    high = feature_df.loc[
        feature_df["sequence_length"] == 13,
        feature
    ]

    F, p = f_oneway(low, mid, high)

    print(
        f"{feature:<25}"
        f"F = {F:10.4f}    "
        f"p = {p:.6e}"
    )