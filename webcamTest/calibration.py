import json

import numpy as np
import pandas as pd

from config import (
    DIGIT_SAMPLE_TOLERANCE_SECONDS,
    GAZE_CALIBRATION_PATH,
    OUTPUT_DIR,
    PUPIL_CALIBRATION_DIGIT_TARGET,
    PUPIL_REFERENCE_PATH,
)


FEATURE_REFERENCE_PATH = OUTPUT_DIR / "gaze_blink_reference.json"
PERSONALIZED_FEATURES = [
    "movement_mean", "movement_std", "movement_max", "movement_min", "movement_median",
    "movement_p95", "movement_p99", "movement_iqr", "movement_cv", "movement_skew", "movement_kurtosis",
    "gaze_dispersion", "dispersion_x", "dispersion_y",
    "center_distance_mean", "center_distance_std", "center_distance_max",
    "gaze_velocity_mean", "gaze_velocity_std", "gaze_velocity_max",
    "acceleration_mean", "acceleration_std", "acceleration_max",
    "fixation_mean_duration", "fixation_max_duration", "hull_area",
    "mean_ibi_ours", "std_ibi_ours", "blink_entropy_trial_ours",
    "blink_ratio", "blink_duration_mean", "blink_duration_std", "blink_duration_max", "blink_duration_min",
]

# 필수 칼럼이 있는지 확인
def get_digit_pupil_series(df):
    required = {"timestamp", "digit_index", "digit_shown_at", "rel_pupil"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Calibration raw is missing columns: {sorted(missing)}")
    # digit_index가 1 이상인 샘플만 남기고, digit onset 시점과 가장 가까운 rel_pupil 값을 선택
    valid = np.isfinite(df["timestamp"]) & np.isfinite(df["digit_index"]) & np.isfinite(df["digit_shown_at"]) & np.isfinite(df["rel_pupil"])
    samples = df.loc[valid, ["timestamp", "digit_index", "digit_shown_at", "rel_pupil"]]
    rows = []
    # digit_index별로 그룹화하여, digit onset 시점과 가장 가까운 rel_pupil 값을 선택
    for digit_index, group in samples.groupby("digit_index", sort=True):
        if digit_index < 1:
            continue
        # digit onset 시점과 가장 가까운 rel_pupil 값을 선택
        onset = group["digit_shown_at"].iloc[0]
        candidates = group[(group["timestamp"] >= onset) & (group["timestamp"] <= onset + DIGIT_SAMPLE_TOLERANCE_SECONDS)]
        if not candidates.empty:
            nearest = candidates.iloc[np.argmin(np.abs(candidates["timestamp"] - onset))]
            rows.append({"digit_index": int(digit_index), "rel_pupil": float(nearest["rel_pupil"])})
    return pd.DataFrame(rows, columns=["digit_index", "rel_pupil"])


def get_digit_pupil_samples(df):
    return get_digit_pupil_series(df)["rel_pupil"].tolist()


def load_pupil_reference(path=PUPIL_REFERENCE_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Pupil reference file was not found: {path}")
    reference = json.loads(path.read_text(encoding="utf-8"))
    mean, std = float(reference["mean"]), float(reference["std"])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Pupil reference must contain a finite mean and positive std.")
    return {"mean": mean, "std": std}


def zscore_digit_pupil(values, reference):
    return (np.asarray(values, dtype=float) - reference["mean"]) / reference["std"]


def save_pupil_reference(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < PUPIL_CALIBRATION_DIGIT_TARGET:
        raise ValueError("Not enough calibration digit samples.")
    std = float(np.std(values, ddof=1))
    if std <= 0:
        raise ValueError("Calibration pupil standard deviation is zero.")
    reference = {"mean": float(np.mean(values)), "std": std, "n_digit_samples": int(len(values)), "calibration_target": PUPIL_CALIBRATION_DIGIT_TARGET}
    PUPIL_REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PUPIL_REFERENCE_PATH.write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")
    return reference


class PupilCalibrator:
    def __init__(self):
        self.active, self.digit_values, self.raw_trials = False, [], []

    def start(self):
        self.active, self.digit_values, self.raw_trials = True, [], []

    def add_trial(self, raw_df):
        if not self.active:
            return None
        if raw_df.empty or ("quality_accepted" in raw_df.columns and not bool(raw_df["quality_accepted"].iloc[0])):
            return {"trial_samples": 0, "total_samples": len(self.digit_values), "reference": None,
                    "skipped": "Trial quality was not accepted."}
        self.raw_trials.append(raw_df.copy())
        samples = get_digit_pupil_samples(raw_df)
        self.digit_values.extend(samples)
        reference = None
        if len(self.digit_values) >= PUPIL_CALIBRATION_DIGIT_TARGET:
            reference = save_pupil_reference(self.digit_values)
            self.active = False
        return {"trial_samples": len(samples), "total_samples": len(self.digit_values), "reference": reference}


def fit_feature_reference(feature_table):
    reference = {}
    for column in PERSONALIZED_FEATURES:
        values = pd.to_numeric(feature_table[column], errors="coerce").dropna()
        std = values.std()
        reference[column] = None if len(values) < 2 or not np.isfinite(std) or std == 0 else {"mean": float(values.mean()), "std": float(std)}
    return reference


def save_feature_reference(reference, path=FEATURE_REFERENCE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")


def load_feature_reference(path=FEATURE_REFERENCE_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Gaze/blink reference file was not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def personalize_features(feature_table, reference):
    result = feature_table.copy()
    for column in PERSONALIZED_FEATURES:
        stats = reference.get(column)
        result[column] = np.nan if stats is None else (result[column] - stats["mean"]) / stats["std"]
    return result


class ScreenGazeCalibrator:
    """Fit an affine webcam-gaze -> normalized-screen mapping from five targets."""

    targets = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)]

    def __init__(self):
        self.active = False
        self.target_index = 0
        self.samples = []
        self.pairs = []

    def start(self):
        self.active, self.target_index, self.samples, self.pairs = True, 0, [], []

    @property
    def target(self):
        return self.targets[self.target_index] if self.active else None

    def add_sample(self, gaze_x, gaze_y):
        if self.active and np.isfinite(gaze_x) and np.isfinite(gaze_y):
            self.samples.append((float(gaze_x), float(gaze_y)))

    def capture_target(self):
        if not self.active:
            raise ValueError("Screen calibration is not active.")
        if len(self.samples) < 10:
            raise ValueError("Hold your gaze on the target until at least 10 face samples are collected.")
        self.pairs.append((np.median(self.samples, axis=0), self.target))
        self.samples = []
        self.target_index += 1
        if self.target_index < len(self.targets):
            return False, self.target

        raw = np.asarray([pair[0] for pair in self.pairs], dtype=float)
        target = np.asarray([pair[1] for pair in self.pairs], dtype=float)
        design = np.column_stack([raw, np.ones(len(raw))])
        coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        payload = {"coefficients": coefficients.tolist(), "targets": self.targets}
        GAZE_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        GAZE_CALIBRATION_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.active = False
        return True, None


def load_screen_gaze_mapping(path=GAZE_CALIBRATION_PATH):
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    coefficients = np.asarray(payload["coefficients"], dtype=float)
    if coefficients.shape != (3, 2) or not np.isfinite(coefficients).all():
        raise ValueError("Screen gaze calibration file is invalid.")
    return coefficients


def apply_screen_gaze_mapping(gaze_x, gaze_y, coefficients):
    if coefficients is None:
        return gaze_x, gaze_y
    mapped = np.asarray([gaze_x, gaze_y, 1.0]) @ coefficients
    return float(mapped[0]), float(mapped[1])
