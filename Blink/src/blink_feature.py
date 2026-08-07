"""
blink_feature.py

blink_load.py(raw blink 플래그 기반 로더)로 trial 단위 blink feature를 계산해서,
train_branch1_lightgbm.py의 load_and_merge()/load_blink()가 pupil/gaze 결과와
바로 병합할 수 있는 형태로 저장함.

train_branch1_lightgbm.py 주석에 나오는 "raw blink 플래그에서 직접 재계산한 버전
(61초 윈도우 제약 없이 계산, 커버리지 91~98%)"이 이 스크립트의 결과물이고,
컬럼명도 그 파이프라인이 기대하는 이름(subject_id, trial_index, n_blinks_ours,
blink_rate_per_min_ours, mean_ibi_ours, std_ibi_ours, blink_entropy_trial_ours,
trial_duration_sec)에 맞춰서 저장함.

주의
------------------------------------------------
- task / sequence_length / difficulty(=condition을 그대로 드러내는 값들)는
  일부러 저장하지 않음. clean_blink_dataset.py와 같은 이유: pupil 쪽에 이미
  label/condition 정보가 있어서 중복이고, 잘못 병합하면 condition이 우회
  노출될 위험이 있음.
- n_blinks_ours / blink_rate_per_min_ours / trial_duration_sec는 trial
  길이에 비례해서 커지는 값이라 train_branch1_lightgbm.py의
  DURATION_LEAKING_BLINK_COLS에 이미 등록되어 있고, --clean-signal 모드에서는
  자동으로 제외됨. mean_ibi_ours / std_ibi_ours / blink_entropy_trial_ours /
  blink_ratio / blink_duration_* 만 실제로 모델에 들어감.
- gaze_feature.py와 동일하게 blink 샘플이 5개 미만인 trial은 품질이 낮다고
  보고 제외함(num_samples는 진단용으로만 쓰고 export 컬럼에는 포함하지 않음).

사용법
------------------------------------------------
    python blink_feature.py
    (subject_split.csv가 있는 위치에서 실행 -> output/blink/blink_features_ours.csv 생성)

이후 pupil/gaze와 합쳐서 학습하려면:
    python train_branch1_lightgbm.py \
        --pupil output/timeseries/pupil_trial_dataset_wide.csv \
        --gaze data/branch/feature_all_gaze.csv \
        --blink output/blink/blink_features_ours.csv \
        --clean-signal
"""

import os
import numpy as np
import pandas as pd

from blink_load import load_blink_data
from event_parser import load_events
from trial_builder import parse_label, make_trials

OUT_PATH = os.path.join("output", "blink", "blink_features_ours.csv")

MIN_SAMPLES = 5  # gaze_feature.py와 동일 기준: 이보다 샘플이 적으면 trial 품질 낮다고 보고 제외


def shannon_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def compute_blink_entropy(ibi_values, n_bins=5):
    """
    trial 내 IBI(초 단위) 값들을 n_bins개 구간으로 나눠 히스토그램을 만들고
    그 분포의 Shannon entropy를 계산.
    -> blink 간격이 얼마나 규칙적/불규칙적인지 하나의 값으로 요약
       (인지 부하가 높을수록 blink 타이밍이 불규칙해진다는 가정하에 넣는 feature)
    """
    if len(ibi_values) < 2:
        return 0.0
    counts, _ = np.histogram(ibi_values, bins=n_bins)
    return shannon_entropy(counts)


def extract_trial_blink_features(blink_trial, trial_duration):
    """
    한 trial 구간의 blink 샘플들로부터 branch1 파이프라인이 기대하는
    이름의 feature dict를 계산. 샘플이 아예 없으면 (None, 0) 반환.
    """

    num_samples = len(blink_trial)

    if num_samples == 0:
        return None, num_samples

    blink_flag = blink_trial["blink"].fillna(0).astype(int).values
    timestamps = blink_trial["gaze_timestamp"].values

    # ==========================
    # 연속된 blink==1 구간을 하나의 blink event로 묶기
    # ==========================
    blink_events = []
    in_blink = False
    blink_start_ts = blink_end_ts = None

    for ts, flag in zip(timestamps, blink_flag):
        if flag == 1 and not in_blink:
            in_blink = True
            blink_start_ts = ts
            blink_end_ts = ts
        elif flag == 1 and in_blink:
            blink_end_ts = ts
        elif flag == 0 and in_blink:
            blink_events.append((blink_start_ts, blink_end_ts))
            in_blink = False

    if in_blink:
        blink_events.append((blink_start_ts, blink_end_ts))

    n_blinks = len(blink_events)
    durations = np.array([e - s for s, e in blink_events])

    blink_duration_mean = durations.mean() if len(durations) else 0
    blink_duration_std = durations.std() if len(durations) > 1 else 0
    blink_duration_max = durations.max() if len(durations) else 0
    blink_duration_min = durations.min() if len(durations) else 0

    blink_rate_per_min = (n_blinks / trial_duration) * 60 if trial_duration > 0 else 0

    time_in_blink = durations.sum()
    blink_ratio = time_in_blink / trial_duration if trial_duration > 0 else 0

    blink_starts = np.array([s for s, e in blink_events])

    if len(blink_starts) > 1:
        ibi = np.diff(blink_starts)
        mean_ibi = ibi.mean()
        std_ibi = ibi.std()
    else:
        ibi = np.array([])
        mean_ibi = 0
        std_ibi = 0

    blink_entropy_trial = compute_blink_entropy(ibi)

    row = {
        # branch1 파이프라인이 기대하는 이름 (DURATION_LEAKING_BLINK_COLS에 등록된 3개 포함)
        "n_blinks_ours": n_blinks,
        "blink_rate_per_min_ours": blink_rate_per_min,
        "mean_ibi_ours": mean_ibi,
        "std_ibi_ours": std_ibi,
        "blink_entropy_trial_ours": blink_entropy_trial,
        "trial_duration_sec": trial_duration,

        # 추가로 넣는 보너스 feature (trial 길이에 비례하지 않는 값이라 누수 아님)
        "blink_ratio": blink_ratio,
        "blink_duration_mean": blink_duration_mean,
        "blink_duration_std": blink_duration_std,
        "blink_duration_max": blink_duration_max,
        "blink_duration_min": blink_duration_min,
    }

    return row, num_samples


def main():

    split_df = pd.read_csv("subject_split.csv")
    subjects = split_df["subject_id"].tolist()

    all_rows = []

    for subject in subjects:

        print(f"Processing {subject}...")

        blink_df = load_blink_data(subject)
        events = load_events(subject)
        trials = make_trials(events)

        for trial_id, trial in enumerate(trials):

            start = trial.iloc[0]["timestamp"]

            if trial_id < len(trials) - 1:
                end = trials[trial_id + 1].iloc[0]["timestamp"]
            else:
                end = trial.iloc[-1]["timestamp"] + 2

            trial_duration = end - start

            blink_trial = blink_df[
                (blink_df["gaze_timestamp"] >= start) &
                (blink_df["gaze_timestamp"] < end)
            ].copy()

            row, num_samples = extract_trial_blink_features(blink_trial, trial_duration)

            if num_samples < MIN_SAMPLES:
                info = parse_label(trial.iloc[0]["label"])
                print(
                    subject, info["task"], info["seq"],
                    "blink samples:", num_samples, "start:", start, "end:", end,
                    "-> 제외됨" if num_samples < MIN_SAMPLES else ""
                )

            # 샘플이 아예 없거나(row is None) 너무 적으면(< MIN_SAMPLES) 이 trial은 건너뜀
            if row is None or num_samples < MIN_SAMPLES:
                continue

            row["subject_id"] = subject
            row["trial_index"] = trial_id

            all_rows.append(row)

    out_df = pd.DataFrame(all_rows)

    # subject_id, trial_index를 맨 앞으로
    cols = ["subject_id", "trial_index"] + [
        c for c in out_df.columns if c not in ("subject_id", "trial_index")
    ]
    out_df = out_df[cols]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"\n저장: {OUT_PATH} ({len(out_df)}행, {len(out_df.columns)}컬럼)")
    print(out_df.columns.tolist())
    print(out_df.head())


if __name__ == "__main__":
    main()