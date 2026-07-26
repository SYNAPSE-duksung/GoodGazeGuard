"""
1. train_feature.csv 로드
2. Correlation Matrix 계산
3. Highly Correlated Feature 출력
4. VIF 계산
5. 제거 후보 출력
"""
import os

import pandas as pd
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "2train_feature.csv")

# Metadata Columns
META_COLUMNS = [
    "subject",
    "split",
    "trial_id",
    "task",
    "sequence_length",
]

# Load Train Feature (Train만 이용)
df = pd.read_csv(TRAIN_PATH)

# Feature 추출
feature_cols = [
    col for col in df.columns
    if col not in META_COLUMNS
]
X = df[feature_cols].astype(float)

print("=" * 50)
print("Feature Selection Report")
print("=" * 50)

print(f"\n✔ Number of Features : {len(feature_cols)}")

# Correlation -> 모든 feature 간 상관계수 계산
corr = X.corr().abs()

upper = corr.where(
    np.triu(np.ones(corr.shape), k=1).astype(bool)
)

print("\n✔ Highly Correlated Features (>|0.95|)\n")

high_corr_pairs = []

for col in upper.columns:
    for row in upper.index:
        value = upper.loc[row, col]

        if pd.notna(value) and value > 0.95:
            high_corr_pairs.append((row, col, value))

if len(high_corr_pairs) == 0:
    print("None")

else:
    for f1, f2, value in sorted(high_corr_pairs,
                                key=lambda x: x[2],
                                reverse=True):
        print(f"{f1:30s} <-> {f2:30s} : {value:.3f}")

# VIF (다중공선성 확인)
print("\n✔ VIF\n")

if X.isnull().values.any():
    raise ValueError("NaN values exist in feature columns.")

vif_df = pd.DataFrame()
vif_df["Feature"] = feature_cols

vif_df["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(len(feature_cols))
]

vif_df = vif_df.sort_values(by="VIF", ascending=False)

# Feature별 VIF Dictionary
vif_dict = dict(zip(vif_df["Feature"], vif_df["VIF"]))

for _, row in vif_df.iterrows():
    print(f"{row['Feature']:30s} : {row['VIF']:.2f}")

# Correlated Feature Groups
print("\n✔ Correlated Feature Groups\n")

groups = []
group_info = []

visited = set()

for f1, f2, _ in high_corr_pairs:

    if f1 in visited or f2 in visited:
        continue

    group = {f1, f2}

    changed = True

    while changed:
        changed = False

        for a, b, _ in high_corr_pairs:

            if a in group and b not in group:
                group.add(b)
                changed = True

            elif b in group and a not in group:
                group.add(a)
                changed = True

    visited.update(group)

    groups.append(sorted(group))

if len(groups) == 0:
    print("None")

else:

    for i, group in enumerate(groups, start=1):
        # VIF가 가장 작은 feature를 대표 변수로 선택
        valid_group = [f for f in group if np.isfinite(vif_dict[f])]

        if len(valid_group) == 0:
            representative = sorted(group)[0]
        else:
            representative = min(valid_group, key=lambda x: vif_dict[x])

        remove = sorted(
            [f for f in group if f != representative]
        )

        group_info.append({
            "representative": representative,
            "remove": remove
        })
        print(f"\nGroup {i}")

        print(f"Representative : {representative}")
        print(f"VIF : {vif_dict[representative]:.2f}")

        if len(remove) == 0:
            print("Remove : None")

        else:
            print("Remove")

            for feat in remove:
                print(f"   - {feat} (VIF={vif_dict[feat]:.2f})")

# Candidate Features
# 대표변수를 제외한 나머지를 제거 후보로 추가
print("\n✔ Candidate Features to Remove\n")

# Correlation으로 제거할 feature
remove_features = set()

for info in group_info:
    remove_features.update(info["remove"])

print("\n✔ Features Removed by Correlation\n")

if len(remove_features) == 0:
    print("None")
else:
    for feat in sorted(remove_features):
        print(feat)

print("\n✔ Recalculate VIF After Correlation Removal\n")

remain_features = [
    f for f in feature_cols
    if f not in remove_features
]

X_new = X[remain_features]

vif_after = pd.DataFrame()

vif_after["Feature"] = remain_features

vif_after["VIF"] = [
    variance_inflation_factor(X_new.values, i)
    for i in range(len(remain_features))
]

vif_after = vif_after.sort_values(
    by="VIF",
    ascending=False
)

for _, row in vif_after.iterrows():
    print(f"{row['Feature']:30s} : {row['VIF']:.2f}")


print("\n✔ Final Candidate Features\n")

candidate = vif_after[
    vif_after["VIF"] > 10
]["Feature"].tolist()

if len(candidate) == 0:
    print("None")
else:
    for feat in candidate:
        print(f"{feat:30s} (VIF={vif_after.loc[vif_after['Feature']==feat,'VIF'].iloc[0]:.2f})")