"""
1. feature.csv 로드
2. train / valid / test 분리
3. StandardScaler 적용 (train만 fit)
4. train/valid/test feature 저장
5. scaler 저장
"""
import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler

# Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_PATH = os.path.join(DATA_DIR, "feature_selected.csv")

TRAIN_PATH = os.path.join(DATA_DIR, "2train_feature.csv")
VALID_PATH = os.path.join(DATA_DIR, "2valid_feature.csv")
TEST_PATH = os.path.join(DATA_DIR, "2test_feature.csv")

SCALER_PATH = os.path.join(DATA_DIR, "2scaler.pkl")

# Metadata Columns
# 학습에 사용하면 안되는 메타데이터 목록 미리 정의
META_COLUMNS = [
    "subject",
    "split",
    "trial_id",
    "task",
    "sequence_length",
]

# Load feature.csv
df = pd.read_csv(INPUT_PATH)
print(f"Total samples : {len(df)}")

# Split
train_df = df[df["split"] == "train"].copy()
valid_df = df[df["split"] == "valid"].copy()
test_df = df[df["split"] == "test"].copy()

print("Train :", train_df.shape)
print("Valid :", valid_df.shape)
print("Test  :", test_df.shape)


# Feature Columns
missing = [col for col in META_COLUMNS if col not in df.columns]
if missing:
    raise ValueError(f"Missing metadata columns: {missing}")

feature_cols = [
    col for col in df.columns
    if col not in META_COLUMNS
]

print(f"\nNumber of features : {len(feature_cols)}")
print(df[feature_cols].dtypes)

train_df[feature_cols] = train_df[feature_cols].astype(float)
valid_df[feature_cols] = valid_df[feature_cols].astype(float)
test_df[feature_cols] = test_df[feature_cols].astype(float)

# StandardScaler
scaler = StandardScaler()

# train의 경우 평균과 표준편차 계산 (valid와 test의 경우는 transform만 진행 -> Data Leakage 방지)
train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
valid_df.loc[:, feature_cols] = scaler.transform(valid_df[feature_cols])
test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

# Save CSV
train_df.to_csv(TRAIN_PATH, index=False)
valid_df.to_csv(VALID_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

# scaler 저장 (현재 학습한 평균과 표준편차 저장)
joblib.dump(scaler, SCALER_PATH)

print("\nSaved files")
print(TRAIN_PATH)
print(VALID_PATH)
print(TEST_PATH)
print(SCALER_PATH)

print("\nDone.")