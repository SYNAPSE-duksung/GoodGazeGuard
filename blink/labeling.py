"""
labeling.py
=============================================================
K-means(k=2) 기반 개인화된 인지과부하 라벨링.

전체 참가자를 한 번에 묶어 클러스터링하지 않고, 참가자(피험자)별로
각자의 정답률 분포에 대해 K-means(k=2)를 따로 수행
→ 사람마다 다른 인지과부하 발생 기준점을 그 사람 데이터
  안에서 스스로 찾아내므로, 전체 평균이나 고정 컷오프보다 개인차를
  훨씬 잘 반영 가능
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv('blink_overload_labels_personalized.csv')
print("--- 파일에 있는 컬럼 이름들 ---")
print(df.columns.tolist())

def label_overload_kmeans_personalized(df: pd.DataFrame, value_col: str,
                                        group_col: str = "participant_id",
                                        label_name: str = "overload_kmeans") -> pd.DataFrame:
    """
    df: 참가자별로 여러 개의 관측치(시행 단위)가 있는 long-format 데이터.
        예) beh_labeled.csv의 시행별 'accuracy' 컬럼
    value_col: 클러스터링 기준 값 (예: 'accuracy')
    group_col: 참가자 식별자 컬럼
    label_name: 결과로 추가될 라벨 컬럼 이름 (1 = 과부하, 0 = 정상)
    """
    out = df.copy()
    out[label_name] = np.nan

    for sub_id, sub_df in out.groupby(group_col):
        valid = sub_df[value_col].notna()
        if int(valid.sum()) < 2:
            continue  # 유효 관측치가 1개 이하면 클러스터링 불가 → NaN
        if sub_df.loc[valid, value_col].nunique() < 2:
            continue  # 값이 전부 동일하면(예: 항상 만점) 클러스터가 무의미 → NaN

        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        clusters = km.fit_predict(sub_df.loc[valid, [value_col]])

        # 두 클러스터 중 값이 더 낮은 쪽(정답률↓ = 과부하)을 1로 지정
        overload_cluster = int(np.argmin(km.cluster_centers_.flatten()))

        idx = sub_df.loc[valid].index
        out.loc[idx, label_name] = (clusters == overload_cluster).astype(int)

    return out


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def label_overload_kmeans_personalized(df: pd.DataFrame, value_col: str,
                                        group_col: str = "participant_id",
                                        label_name: str = "overload_kmeans") -> pd.DataFrame:
    out = df.copy()
    out[label_name] = np.nan

    for sub_id, sub_df in out.groupby(group_col):
        valid = sub_df[value_col].notna()
        if int(valid.sum()) < 2:
            continue  # 유효 관측치가 1개 이하면 클러스터링 불가 → NaN
        if sub_df.loc[valid, value_col].nunique() < 2:
            continue  # 값이 전부 동일하면(예: 항상 만점) 클러스터가 무의미 → NaN

        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        clusters = km.fit_predict(sub_df.loc[valid, [value_col]])

        # 두 클러스터 중 값이 더 낮은 쪽(정답률↓ = 과부하)을 1로 지정
        overload_cluster = int(np.argmin(km.cluster_centers_.flatten()))

        idx = sub_df.loc[valid].index
        out.loc[idx, label_name] = (clusters == overload_cluster).astype(int)

    return out
