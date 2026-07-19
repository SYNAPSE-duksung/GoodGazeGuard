from load_data import load_gaze_data
from event_parser import load_events
from scipy.spatial import ConvexHull, QhullError
import pandas as pd
import numpy as np

subjects = [
    "sub-013",
    "sub-014",
    "sub-015",
    "sub-016",
    "sub-018",
    "sub-019",
    "sub-020",
    "sub-021",
    "sub-022"
]

def parse_label(label):

    label = str(int(label))

    if label.startswith("50"):   # Listening
        return {
            "task": "listening",
            "digit": int(label[2:4]),
            "seq": int(label[4:6])
        }

    elif label.startswith("60"):   # Memory
        return {
            "task": "memory",
            "digit": int(label[2:4]),
            "seq": int(label[4:6]),
            "correct": int(label[6])
        }

def make_trials(events):
    """
    Event들을 Trial 단위(5/9/13 digit)로 묶기
    """
    trials = []

    i = 0
    while i < len(events):
        info = parse_label(events.iloc[i]["label"])
        seq = info["seq"]
        trials.append(events.iloc[i:i+seq])

        i += seq

    return trials


all_features = []

for subject in subjects:

    print(f"Processing {subject}...")

    gaze_df = load_gaze_data(subject)
    events = load_events(subject)

    trials = make_trials(events)

    features = []

    for trial in trials:

        start = trial.iloc[0]["timestamp"]
        end = trial.iloc[-1]["timestamp"] + 2

        info = parse_label(trial.iloc[0]["label"])

        gaze_trial = gaze_df[
            (gaze_df["gaze_timestamp"] >= start) &
            (gaze_df["gaze_timestamp"] <= end)
        ].copy()

        # Trial 내부에서 movement 계산
        gaze_trial["dx"] = gaze_trial["gaze_norm_pos_x"].diff()
        gaze_trial["dy"] = gaze_trial["gaze_norm_pos_y"].diff()

        gaze_trial["movement"] = (
            gaze_trial["dx"]**2 + gaze_trial["dy"]**2
        )**0.5

        if len(gaze_trial) < 5:
            print(
                subject,
                info["task"],
                info["seq"],
                "gaze samples:", len(gaze_trial),
                "start:", start,
                "end:", end
            )

        movement = gaze_trial["movement"].dropna()     

        if len(movement) == 0:
            features.append({
                "subject": subject,
                "task": info["task"],
                "sequence_length": info["seq"],
                "label": int(trial.iloc[0]["label"]),

                "movement_mean": 0,
                "movement_std": 0,
                "movement_max": 0,
                "movement_min": 0,
                "movement_median": 0,
                "movement_p95": 0,
                "movement_p99": 0,
                "movement_iqr": 0,
                "movement_cv": 0,
                "movement_skew": 0,
                "movement_kurtosis": 0,
                "scanpath_length": 0,
                "num_samples": 0,
                "dispersion": 0,
                "center_distance_std": 0,
                "velocity_mean": 0,
                "velocity_std": 0,
                "fixation_mean_duration": 0,
                "fixation_count": 0,
                "hull_area": 0,
            })

            continue

        points = gaze_trial[
            ["gaze_norm_pos_x","gaze_norm_pos_y"]
        ].dropna().values

        try:
            if len(points) >= 3:
                hull_area = ConvexHull(points).volume
            else:
                hull_area = 0
        except QhullError:
            hull_area = 0   

        # ==========================
        # 1. Gaze Dispersion
        # ==========================
        dispersion_x = gaze_trial["gaze_norm_pos_x"].std()
        dispersion_y = gaze_trial["gaze_norm_pos_y"].std()

        dispersion = np.sqrt(
            dispersion_x**2 +
            dispersion_y**2
        )

        # ==========================
        # 2. Center Distance
        # ==========================
        center_distance = np.sqrt(
            (gaze_trial["gaze_norm_pos_x"] - 0.5) ** 2 +
            (gaze_trial["gaze_norm_pos_y"] - 0.5) ** 2
        )

        center_distance_mean = center_distance.mean()
        center_distance_std = center_distance.std()
        center_distance_max = center_distance.max()

        # ==========================
        # 3. Velocity
        # ==========================
        dt = gaze_trial["gaze_timestamp"].diff()

        velocity = (
            movement /
            gaze_trial["gaze_timestamp"].diff().loc[movement.index]
        )

        velocity = velocity.replace(
            [np.inf,-np.inf],
            np.nan
        ).dropna()

        velocity_mean = velocity.mean() if len(velocity) else 0
        velocity_std = velocity.std() if len(velocity) > 1 else 0
        velocity_max = velocity.max() if len(velocity) else 0

        # ==========================
        # 4. Acceleration
        # ==========================
        velocity_dt = gaze_trial["gaze_timestamp"].diff().loc[
            velocity.index
        ]

        acceleration = (
            velocity.diff() / velocity_dt
        )

        acceleration_mean = acceleration.mean() if len(acceleration) else 0
        acceleration_std = acceleration.std() if len(acceleration) > 1 else 0
        acceleration_max = acceleration.max() if len(acceleration) else 0

        # ==========================
        # 5. Fixation Duration
        # ==========================
        fix_thr = movement.quantile(0.10)

        fix_mask = movement < fix_thr

        fix_lengths = []

        count = 0

        for x in fix_mask:
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

        fixation_count = len(fix_lengths)
        
        # IQR
        q1 = movement.quantile(0.25)
        q3 = movement.quantile(0.75)
        iqr = q3 - q1

        # Coefficient of Variation
        movement_cv = (
            movement.std() / movement.mean()
            if movement.mean() > 1e-6 else 0
        )

        num_samples=len(gaze_trial)

        features.append({
            "subject": subject,
            "task": info["task"],
            "sequence_length": info["seq"],
            "label": int(trial.iloc[0]["label"]),

            "movement_mean": movement.mean(),
            "movement_std": movement.std() if len(movement) > 1 else 0,

            "movement_max": movement.max(),
            "movement_min": movement.min(),
            "movement_median": movement.median(),

            "movement_p95": movement.quantile(0.95),
            "movement_p99": movement.quantile(0.99),

            "movement_iqr": iqr,
            "movement_cv": movement_cv,

            "movement_skew": movement.skew() if len(movement) > 2 else 0,
            "movement_kurtosis": movement.kurt() if len(movement) > 3 else 0,
            "scanpath_length": movement.sum(),
            "num_samples": num_samples,

            # Dispersion
            "dispersion": dispersion,
            "dispersion_x": dispersion_x,
            "dispersion_y": dispersion_y,

            # Center distance
            "center_distance_mean": center_distance_mean,
            "center_distance_std": center_distance_std,
            "center_distance_max": center_distance_max,

            # Velocity
            "velocity_mean": velocity_mean,
            "velocity_std": velocity_std,
            "velocity_max": velocity_max,

            # Acceleration
            "acceleration_mean": acceleration_mean,
            "acceleration_std": acceleration_std,
            "acceleration_max": acceleration_max,

            # Fixation
            "fixation_mean_duration": fixation_mean_duration,
            "fixation_max_duration": fixation_max_duration,
            "fixation_count": fixation_count,

            "hull_area": hull_area,
        })

    all_features.append(pd.DataFrame(features))

feature_df = pd.concat(
    all_features,
    ignore_index=True
)
feature_df = feature_df[feature_df["num_samples"] >= 5]

feature_df.to_csv("feature.csv", index=False)

print(feature_df.head())
print(feature_df.shape)
print(events[["timestamp", "label"]].head())
print(gaze_df["gaze_timestamp"].head())
print(feature_df[feature_df.isna().any(axis=1)])


corr = feature_df.corr(numeric_only=True)

print(
    corr["movement_mean"]
    .sort_values(ascending=False)
)

import seaborn as sns
import matplotlib.pyplot as plt

corr = feature_df.select_dtypes(include=np.number).corr()

plt.figure(figsize=(15,12))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)
plt.show()

print(__file__)

print(feature_df.columns.tolist())

print(len(feature_df.columns))