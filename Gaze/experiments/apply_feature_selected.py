"""
feature.csv에서 선택한 feature를 제거하여
feature_selected.csv 생성
"""

import os
import pandas as pd

# Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

FEATURE_PATH = os.path.join(DATA_DIR, "feature.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "feature_selected.csv")

# 제거할 Feature
REMOVE_FEATURES = [
    # Velocity와 중복
    "acceleration_mean",
    "acceleration_std",
    "acceleration_max",

    # 거의 동일한 의미
    "movement_skew",
    "num_samples",

    # Spatial 중복
    "dispersion_x",
    "dispersion_y",

    # Movement 통계 중복
    "movement_p95",
    "movement_p99",
    "movement_iqr",
]

# Load
df = pd.read_csv(FEATURE_PATH)

print(f"Original Features : {df.shape[1]}")

# 삭제
df = df.drop(columns=REMOVE_FEATURES)

print(f"Selected Features : {df.shape[1]}")

# 저장
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved : {OUTPUT_PATH}")