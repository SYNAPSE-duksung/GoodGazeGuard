import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from beh_loader import load_beh_data


def compute_subject_thresholds(beh_df):
    """
    피험자별로:
      1. 정답률(accuracy)을 자기 자신의 평균/표준편차 기준으로 Z-score화
      2. Z-score에 K-means(k=2)를 적용해 '정상' vs '과부하' 클러스터로 분리
      3. 두 클러스터 중심의 중간값을 그 피험자의 인지 과부하 경계값(threshold)으로 사용
    -> 피험자마다 기준선이 다르기 때문에 전체 공통 threshold 대신
       개인별 맞춤 threshold를 산출하는 방식
    """
    results = []

    for subject, group in beh_df.groupby("subject"):

        acc = group["accuracy"].values

        if len(acc) < 2 or acc.std() == 0:
            z = np.zeros_like(acc)
            threshold_z = 0.0
        else:
            z = (acc - acc.mean()) / acc.std()

            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            km.fit(z.reshape(-1, 1))

            centers = sorted(km.cluster_centers_.flatten())
            threshold_z = (centers[0] + centers[1]) / 2

        subject_df = group.copy()
        subject_df["accuracy_z"] = z
        subject_df["overload_threshold_z"] = threshold_z
        # 정답률이 threshold보다 낮은(=z가 낮은) 쪽을 과부하 상태로 정의
        subject_df["is_overload"] = subject_df["accuracy_z"] < threshold_z

        results.append(subject_df)

    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":

    split_df = pd.read_csv("subject_split.csv")
    subjects = split_df["subject_id"].tolist()

    all_beh = []
    for subject in subjects:
        try:
            all_beh.append(load_beh_data(subject))
        except FileNotFoundError as e:
            print(f"Skip {subject}: {e}")

    beh_df = pd.concat(all_beh, ignore_index=True)

    result_df = compute_subject_thresholds(beh_df)

    result_df.to_csv("overload_labels.csv", index=False)

    print(result_df.head())
    print(f"\nTotal subjects processed: {result_df['subject'].nunique()}")
    print(f"Overload trial ratio: {result_df['is_overload'].mean():.3f}")