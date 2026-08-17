import numpy as np
import pandas as pd

from config import MIN_BLINK_DURATION_SECONDS, MIN_VALID_FRAME_RATIO, MIN_VALID_FRAMES

from calibration import (
    get_digit_pupil_series,
    load_pupil_reference,
    zscore_digit_pupil,
)
from calibration import personalize_features

FEATURE_ORDER = [
    "pupil_mean",
    "pupil_std",
    "pupil_min",
    "pupil_max",
    "pupil_first",
    "pupil_last",
    "pupil_slope",

    "movement_mean",
    "movement_std",
    "movement_max",
    "movement_min",
    "movement_median",
    "movement_p95",
    "movement_p99",
    "movement_iqr",
    "movement_cv",
    "movement_skew",
    "movement_kurtosis",

    "gaze_dispersion",
    "dispersion_x",
    "dispersion_y",

    "center_distance_mean",
    "center_distance_std",
    "center_distance_max",

    "gaze_velocity_mean",
    "gaze_velocity_std",
    "gaze_velocity_max",

    "acceleration_mean",
    "acceleration_std",
    "acceleration_max",

    "fixation_mean_duration",
    "fixation_max_duration",
    "hull_area",

    "mean_ibi_ours",
    "std_ibi_ours",
    "blink_entropy_trial_ours",

    "blink_ratio",
    "blink_duration_mean",
    "blink_duration_std",
    "blink_duration_max",
    "blink_duration_min",
]


def safe_slope(values, positions=None):
    """시간에 따른 선형 변화량"""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0

    x = np.arange(len(values)) if positions is None else np.asarray(positions, dtype=float)
    return np.polyfit(x, values, 1)[0]

def extract_features(df, pupil_reference=None, feature_reference=None):
    """
    trial_raw.csv-> 41개 feature
    """

    if "quality_accepted" in df.columns and not bool(df["quality_accepted"].iloc[0]):
        raise ValueError("Trial quality is below the configured minimum; feature extraction is blocked.")
    if len(df) < MIN_VALID_FRAMES:
        raise ValueError(f"At least {MIN_VALID_FRAMES} valid frames are required for feature extraction.")

    # -------------------------
    # 기본 데이터
    # -------------------------
    gaze_x = df["gaze_x"].to_numpy(dtype=float)
    gaze_y = df["gaze_y"].to_numpy(dtype=float)
    pupil = df["rel_pupil"].to_numpy(dtype=float)
    blink_ratio = df["blink_ratio"].to_numpy(dtype=float)
    blink_flag = df["blink_flag"].to_numpy(dtype=int)

    # NaN 제거
    valid = (
        np.isfinite(gaze_x)
        & np.isfinite(gaze_y)
        & np.isfinite(pupil)
        & np.isfinite(blink_ratio)
        & np.isfinite(blink_flag)
    )

    gaze_x = gaze_x[valid]
    gaze_y = gaze_y[valid]
    pupil = pupil[valid]
    blink_ratio = blink_ratio[valid]
    blink_flag = blink_flag[valid]

    # -------------------------
    # 시간
    # -------------------------
    timestamp = df["timestamp"].to_numpy(dtype=float)[valid]

    # webcam raw에는 digit onset event 이후의 frame마다 digit 정보가 저장된다.
    # 원 학습 데이터와 같이 각 onset에 가장 가까운 pupil sample 하나만 사용한다.
    pupil_positions = None
    if {"digit_index", "digit_shown_at"}.issubset(df.columns):
        digit_pupil = get_digit_pupil_series(df)
        if digit_pupil.empty:
            raise ValueError("No valid pupil sample was recorded within one second after a digit onset.")
        if pupil_reference is None:
            pupil_reference = load_pupil_reference()
        pupil_for_features = zscore_digit_pupil(
            digit_pupil["rel_pupil"].to_numpy(),
            pupil_reference,
        )
        pupil_positions = digit_pupil["digit_index"].to_numpy(dtype=float) - 1
    elif False and {"digit_index", "digit_shown_at"}.issubset(df.columns):
        digit_index = df["digit_index"].to_numpy(dtype=float)[valid]
        digit_shown_at = df["digit_shown_at"].to_numpy(dtype=float)[valid]
        digit_pupil = []

        for index in np.unique(digit_index[np.isfinite(digit_index)]):
            if index < 1:
                continue

            in_digit = digit_index == index
            onset_values = digit_shown_at[in_digit]
            onset_values = onset_values[np.isfinite(onset_values)]
            if len(onset_values) == 0:
                continue

            onset = onset_values[0]
            offsets = timestamp[in_digit] - onset
            within_tolerance = (offsets >= 0.0) & (offsets <= 1.0)
            if not np.any(within_tolerance):
                continue

            candidate_offsets = offsets[within_tolerance]
            candidate_pupil = pupil[in_digit][within_tolerance]
            digit_pupil.append(candidate_pupil[np.argmin(np.abs(candidate_offsets))])

        pupil_for_features = np.asarray(digit_pupil, dtype=float)
        if len(pupil_for_features) == 0:
            raise ValueError(
                "digit event 뒤 1초 안의 pupil sample이 없습니다. "
                "각 digit 표시 직후 mark_digit()을 호출했는지 확인하세요."
            )
    else:
        # digit event가 없는 과거 raw CSV도 확인할 수 있도록 유지한다.
        # 단, 이 경로는 학습의 digit 단위 pupil 입력과 동일하지 않다.
        pupil_for_features = pupil

    if len(timestamp) > 1:
        dt = np.diff(timestamp)
    else:
        dt = np.array([1.0])

    # ==================================================
    # PUPIL
    # train_branch1_lightgbm.py의 clean-signal과 동일한 7개
    # ==================================================
    pupil_features = {
        "pupil_mean": np.mean(pupil_for_features),
        "pupil_std": np.std(pupil_for_features),
        "pupil_min": np.min(pupil_for_features),
        "pupil_max": np.max(pupil_for_features),
        "pupil_first": pupil_for_features[0],
        "pupil_last": pupil_for_features[-1],
        "pupil_slope": safe_slope(pupil_for_features, pupil_positions),
    }

    # ==================================================
    # GAZE MOVEMENT
    # ==================================================
    dx = np.diff(gaze_x)
    dy = np.diff(gaze_y)
    movement = np.sqrt(dx ** 2 + dy ** 2)

    if len(movement) == 0:
        movement = np.array([0.0])

    # -------------------------
    # movement statistics
    # -------------------------
    movement_mean = np.mean(movement)
    movement_std = pd.Series(movement).std() if len(movement) > 1 else 0.0
    movement_features = {
        "movement_mean": movement_mean,
        "movement_std": movement_std,
        "movement_max": np.max(movement),
        "movement_min": np.min(movement),
        "movement_median": np.median(movement),
        "movement_p95": np.percentile(movement, 95),
        "movement_p99": np.percentile(movement, 99),
        "movement_iqr": np.percentile(movement, 75)
                            - np.percentile(movement, 25),
        "movement_cv": (
            movement_std / movement_mean
            if movement_mean > 1e-6 else 0.0
        ),
        "movement_skew": (
            pd.Series(movement).skew()
            if len(movement) > 2 else 0.0
        ),
        "movement_kurtosis": (
            pd.Series(movement).kurtosis()
            if len(movement) > 3 else 0.0
        ),
    }

    # ==================================================
    # GAZE DISPERSION
    # ==================================================
    dispersion_x = pd.Series(gaze_x).std() if len(gaze_x) > 1 else 0.0
    dispersion_y = pd.Series(gaze_y).std() if len(gaze_y) > 1 else 0.0

    gaze_dispersion = np.sqrt(
        dispersion_x ** 2 +
        dispersion_y ** 2
    )

    center_distance = np.sqrt(
        (gaze_x - 0.5) ** 2 +
        (gaze_y - 0.5) ** 2
    )

    gaze_features = {
        "gaze_dispersion": gaze_dispersion,
        "dispersion_x": dispersion_x,
        "dispersion_y": dispersion_y,

        "center_distance_mean": np.mean(center_distance),
        "center_distance_std": pd.Series(center_distance).std() if len(center_distance) > 1 else 0.0,
        "center_distance_max": np.max(center_distance),
    }

    # ==================================================
    # GAZE VELOCITY
    # ==================================================
    if len(dx) > 0:
        valid_dt = dt > 0
        velocity_x = dx[valid_dt] / dt[valid_dt]
        velocity_y = dy[valid_dt] / dt[valid_dt]
        gaze_velocity = np.sqrt(velocity_x ** 2 + velocity_y ** 2)
    else:
        gaze_velocity = np.array([0.0])

    gaze_velocity_features = {
        "gaze_velocity_mean": np.mean(gaze_velocity) if len(gaze_velocity) else 0.0,
        "gaze_velocity_std": pd.Series(gaze_velocity).std() if len(gaze_velocity) > 1 else 0.0,
        "gaze_velocity_max": np.max(gaze_velocity) if len(gaze_velocity) else 0.0,
    }

    # ==================================================
    # ACCELERATION
    # ==================================================
    if len(gaze_velocity) > 1 and len(dt) > 1:
        dv = np.diff(gaze_velocity)
        acceleration_dt = dt[valid_dt][1:]
        acceleration = dv / acceleration_dt
        acceleration = acceleration[np.isfinite(acceleration)]
    else:
        acceleration = np.array([0.0])

    acceleration_features = {
        "acceleration_mean": np.mean(acceleration) if len(acceleration) else 0.0,
        "acceleration_std": pd.Series(acceleration).std() if len(acceleration) > 1 else 0.0,
        "acceleration_max": np.max(acceleration) if len(acceleration) else 0.0,
    }

    # ==================================================
    # FIXATION
    # 기존 gaze feature와 동일한 방식
    # ==================================================
    fix_thr = np.quantile(movement, 0.10)

    fixation_mask = movement < fix_thr

    fix_lengths = []
    count = 0

    for x in fixation_mask:
        if x:
            count += 1
        elif count > 0:
            fix_lengths.append(count)
            count = 0

    if count > 0:
        fix_lengths.append(count)

    fixation_mean_duration = (
        np.mean(fix_lengths)
        if len(fix_lengths)
        else 0
    )

    fixation_max_duration = (
        np.max(fix_lengths)
        if len(fix_lengths)
        else 0
    )

    fixation_features = {
        "fixation_mean_duration": fixation_mean_duration,
        "fixation_max_duration": fixation_max_duration,
    }

    # ==================================================
    # HULL AREA
    # ==================================================
    try:
        from scipy.spatial import ConvexHull
        points = np.column_stack(
            [gaze_x, gaze_y]
        )

        if len(points) >= 3:
            hull = ConvexHull(points)
            hull_area = hull.volume
        else:
            hull_area = 0.0
    except Exception:
        hull_area = 0.0

    # ==================================================
    # BLINK
    # ==================================================
    # 학습 코드와 동일하게 blink_flag == 1인 연속 구간을
    # 하나의 blink event로 묶음
    blink_events = []
    in_blink = False
    blink_start_ts = None
    blink_end_ts = None

    for ts, flag in zip(timestamp, blink_flag):
        if flag == 1 and not in_blink:
            # blink 시작
            in_blink = True
            blink_start_ts = ts
            blink_end_ts = ts
        elif flag == 1 and in_blink:
            # blink 지속
            blink_end_ts = ts
        elif flag == 0 and in_blink:
            # blink 종료
            blink_events.append(
                (blink_start_ts, blink_end_ts)
            )

            in_blink = False
            blink_start_ts = None
            blink_end_ts = None

    # 마지막 샘플이 blink인 경우
    if in_blink:
        blink_events.append(
            (blink_start_ts, blink_end_ts)
        )

    blink_events = [
        (start, end)
        for start, end in blink_events
        if end - start >= MIN_BLINK_DURATION_SECONDS
    ]

    # --------------------------------------------------
    # trial duration
    # --------------------------------------------------
    if len(timestamp) > 1:
        trial_duration = timestamp[-1] - timestamp[0]
    else:
        trial_duration = 0.0

    # --------------------------------------------------
    # blink duration
    # --------------------------------------------------
    n_blinks = len(blink_events)
    durations = np.array(
        [end - start for start, end in blink_events],
        dtype=float
    )

    blink_duration_mean = (
        durations.mean()
        if len(durations)
        else 0.0
    )

    blink_duration_std = (
        durations.std()
        if len(durations) > 1
        else 0.0
    )

    blink_duration_max = (
        durations.max()
        if len(durations)
        else 0.0
    )

    blink_duration_min = (
        durations.min()
        if len(durations)
        else 0.0
    )

    # --------------------------------------------------
    # blink ratio
    # --------------------------------------------------
    time_in_blink = durations.sum()
    blink_ratio_feature = (
        time_in_blink / trial_duration
        if trial_duration > 0
        else 0.0
    )

    # --------------------------------------------------
    # IBI
    # --------------------------------------------------
    blink_starts = np.array(
        [start for start, end in blink_events],
        dtype=float
    )
    if len(blink_starts) > 1:
        ibi = np.diff(blink_starts)
        mean_ibi = ibi.mean()
        std_ibi = ibi.std()
    else:
        ibi = np.array([])
        mean_ibi = 0.0
        std_ibi = 0.0

    # --------------------------------------------------
    # blink entropy
    # --------------------------------------------------
    def shannon_entropy(counts):
        counts = np.asarray(counts, dtype=float)
        counts = counts[counts > 0]
        if counts.sum() == 0:
            return 0.0

        p = counts / counts.sum()
        return float(-(p * np.log2(p)).sum())

    def compute_blink_entropy(ibi_values, n_bins=5):
        if len(ibi_values) < 2:
            return 0.0
        counts, _ = np.histogram(
            ibi_values,
            bins=n_bins
        )
        return shannon_entropy(counts)

    blink_entropy_trial = compute_blink_entropy(ibi)
    blink_features = {
        "mean_ibi_ours": mean_ibi,
        "std_ibi_ours": std_ibi,
        "blink_entropy_trial_ours":blink_entropy_trial,
        "blink_ratio": blink_ratio_feature,
        "blink_duration_mean": blink_duration_mean,
        "blink_duration_std": blink_duration_std,
        "blink_duration_max":blink_duration_max,
        "blink_duration_min": blink_duration_min,
    }
    # ==================================================
    # 합치기
    # ==================================================
    features = {}
    features.update(pupil_features)
    features.update(movement_features)
    features.update(gaze_features)
    features.update(gaze_velocity_features)
    features.update(acceleration_features)
    features.update(fixation_features)
    features["hull_area"] = hull_area
    features.update(blink_features)

    # -------------------------
    # 정확히 41개 + 순서 확인
    # -------------------------
    result = pd.DataFrame(
        [[features[name] for name in FEATURE_ORDER]],
        columns=FEATURE_ORDER
    )

    if list(result.columns) != FEATURE_ORDER:
        raise RuntimeError("Feature order does not match the Branch1 model input order.")
    return personalize_features(result, feature_reference) if feature_reference is not None else result
