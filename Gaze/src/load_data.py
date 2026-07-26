import pandas as pd
import os

BASE_PATH = r"D:\OpenNeuro"

def load_gaze_data(subject, confidence_threshold=0.6):

    data_path = os.path.join(BASE_PATH, subject, "pupil")

    cols = [
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


    # Gaze 데이터만 선택
    gaze_df = df[df["data_part"] == "gaze"].copy()

    # Confidence filtering
    gaze_df = gaze_df[gaze_df["confidence"] >= confidence_threshold]

    # 결측 제거
    gaze_df = gaze_df.dropna(
        subset=[
            "gaze_norm_pos_x",
            "gaze_norm_pos_y"
        ]
    )

    file_path = os.path.join(
        data_path,
        f"{subject}_task-memory_pupil.tsv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    df = pd.read_csv(
        file_path,
        sep="\t",
        usecols=cols
    )

    # 정상 좌표만 사용 (normalized coordinate)
    gaze_df = gaze_df[
        (gaze_df["gaze_norm_pos_x"] >= 0) &
        (gaze_df["gaze_norm_pos_x"] <= 1) &
        (gaze_df["gaze_norm_pos_y"] >= 0) &
        (gaze_df["gaze_norm_pos_y"] <= 1)
    ].copy()

    return gaze_df