import os
import pandas as pd

BASE_PATH = r"D:\OpenNeuro"


def load_beh_data(subject):
    """
    beh 폴더에서 정답률(NCorrect) 데이터 로드
    trial 단위 accuracy = NCorrect / condition(자릿수)

    주의: 실제 beh 폴더 파일명이 아래 file_path와 다르면
    (예: sub-XXX_task-memory_beh.tsv가 아니라면) 이 부분만 수정하면 됨
    """
    data_path = os.path.join(BASE_PATH, subject, "beh")

    file_path = os.path.join(
        data_path,
        f"{subject}_task-memory_beh.tsv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path, sep="\t")

    df["accuracy"] = df["NCorrect"] / df["condition"]

    df["subject"] = subject
    df["difficulty"] = df["condition"].map(
        {5: "Low", 9: "Medium", 13: "High"}
    )

    return df