import pandas as pd
from config import OUTPUT_DIR
from extract_features import extract_features

df = pd.read_csv(OUTPUT_DIR / "trial_raw.csv")

features = extract_features(df)

print(features)
print()
print("shape:", features.shape)
print()
print(features.columns.tolist())
