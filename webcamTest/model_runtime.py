"""Safe loading and prediction helpers for the Branch1 LightGBM model."""

from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from extract_features import FEATURE_ORDER


LABELS = ["Low", "Medium", "High"]
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "branch1" / "dataset" / "branch1_lightgbm_personalized.txt"
CACHE_PATH = BASE_DIR / ".model_cache" / "branch1_lightgbm_personalized.lf.txt"


def normalized_model_path(model_path=MODEL_PATH, cache_path=CACHE_PATH):
    """Return an LF-only copy, protecting LightGBM models from CRLF conversion."""
    source = Path(model_path)
    if not source.exists():
        raise FileNotFoundError(f"Branch1 model was not found: {source}")

    source_bytes = source.read_bytes()
    normalized_bytes = source_bytes.replace(b"\r\n", b"\n")
    cache = Path(cache_path)
    if not cache.exists() or cache.read_bytes() != normalized_bytes:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(normalized_bytes)
    return cache


@lru_cache(maxsize=1)
def load_model():
    model = lgb.Booster(model_file=str(normalized_model_path()))
    model_features = model.feature_name()
    if model_features != FEATURE_ORDER:
        raise RuntimeError(
            "Branch1 model feature schema does not match webcam FEATURE_ORDER. "
            f"model={model_features}, webcam={FEATURE_ORDER}"
        )
    return model


def predict_features(features: pd.DataFrame):
    if list(features.columns) != FEATURE_ORDER or len(features) != 1:
        raise ValueError("Prediction requires one row with the Branch1 41-feature schema.")
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = features.columns[~np.isfinite(values[0])].tolist()
        raise ValueError(f"Prediction is blocked because features are non-finite: {bad}")

    probabilities = load_model().predict(features)[0]
    label_index = int(np.argmax(probabilities))
    return LABELS[label_index], probabilities
