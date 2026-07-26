import os
import random
import pandas as pd

# ==========================
# Configuration
# ==========================
BASE_PATH = r"D:\OpenNeuro"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

OUTPUT_FILE = "subject_split.csv"


# ==========================
# Find valid subjects
# ==========================
subjects = sorted([
    d for d in os.listdir(BASE_PATH)
    if d.startswith("sub-")
    and os.path.isdir(os.path.join(BASE_PATH, d))
])

valid_subjects = []

for subject in subjects:

    pupil_file = os.path.join(
        BASE_PATH,
        subject,
        "pupil",
        f"{subject}_task-memory_pupil.tsv"
    )

    # pupil 데이터가 존재하는 참가자만 사용
    if os.path.exists(pupil_file):
        valid_subjects.append(subject)

print(f"Total subjects : {len(subjects)}")
print(f"Valid subjects : {len(valid_subjects)}")


# ==========================
# Shuffle
# ==========================
shuffled_subjects = valid_subjects.copy()

random.seed(RANDOM_SEED)
random.shuffle(shuffled_subjects)


# ==========================
# Train / Valid / Test Split
# ==========================
n = len(shuffled_subjects)

train_end = int(n * TRAIN_RATIO)
valid_end = train_end + int(n * VALID_RATIO)

train_subjects = shuffled_subjects[:train_end]
valid_subjects = shuffled_subjects[train_end:valid_end]
test_subjects = shuffled_subjects[valid_end:]

print(f"Train : {len(train_subjects)}")
print(f"Valid : {len(valid_subjects)}")
print(f"Test  : {len(test_subjects)}")


# ==========================
# Save CSV
# ==========================
rows = []

for subject in train_subjects:
    rows.append({
        "subject_id": subject,
        "split": "train"
    })

for subject in valid_subjects:
    rows.append({
        "subject_id": subject,
        "split": "valid"
    })

for subject in test_subjects:
    rows.append({
        "subject_id": subject,
        "split": "test"
    })

split_df = pd.DataFrame(rows)

# 보기 좋게 subject 순으로 정렬
split_df = split_df.sort_values("subject_id").reset_index(drop=True)

split_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved -> {OUTPUT_FILE}")

print("\nSplit Summary")
print(split_df["split"].value_counts())

print("\nFirst 10 rows")
print(split_df.head(10))