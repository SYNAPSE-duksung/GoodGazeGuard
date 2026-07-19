import pandas as pd
import os

BASE_PATH = r"D:\OpenNeuro"

def load_gaze_data(subject):

    data_path = os.path.join(BASE_PATH, subject, "pupil")

    cols = [
        "pupil_timestamp",
        "gaze_timestamp",
        "confidence",
        "blink",
        "diameter_3d",
        "gaze_norm_pos_x",
        "gaze_norm_pos_y",
        "data_part"
    ]

    df = pd.read_csv(
        os.path.join(data_path, f"{subject}_task-memory_pupil.tsv"),
        sep="\t",
        usecols=cols
    )

    # Confidence filtering
    df = df[df["confidence"] > 0.6]

    gaze_df = (
        df[df["data_part"] == "gaze"].dropna(subset=[
            "gaze_norm_pos_x",
            "gaze_norm_pos_y"
        ]).copy()
    )

    # 정상 좌표만 사용 (normalized coordinate)
    gaze_df = gaze_df[
        (gaze_df["gaze_norm_pos_x"] >= 0) &
        (gaze_df["gaze_norm_pos_x"] <= 1) &
        (gaze_df["gaze_norm_pos_y"] >= 0) &
        (gaze_df["gaze_norm_pos_y"] <= 1)
    ].copy()

    return gaze_df