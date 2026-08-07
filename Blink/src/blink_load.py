import os
import pandas as pd

BASE_PATH = r"D:\OpenNeuro"


def load_blink_data(subject, confidence_threshold=0.6):
    """
    pupil.tsv 안에서 각 gaze 샘플에 붙어있는 blink 플래그(0/1)를 로드
    (blink는 별도 파일이 아니라 data_part == "gaze" 행에 포함된 컬럼)
    """

    data_path = os.path.join(BASE_PATH, subject, "pupil")

    file_path = os.path.join(
        data_path,
        f"{subject}_task-memory_pupil.tsv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    cols = [
        "gaze_timestamp",
        "confidence",
        "blink",
        "data_part"
    ]

    df = pd.read_csv(file_path, sep="\t", usecols=cols)

    # blink는 gaze row 기준 timeline을 사용 (gaze_feature.py와 동일한 timeline)
    blink_df = df[df["data_part"] == "gaze"].copy()

    # confidence 필터링 (gaze 쪽과 동일 기준 -> 나중에 merge할 때 일관성 유지)
    blink_df = blink_df[blink_df["confidence"] >= confidence_threshold]

    blink_df = blink_df.dropna(subset=["blink", "gaze_timestamp"])

    blink_df = blink_df.sort_values("gaze_timestamp").reset_index(drop=True)

    return blink_df