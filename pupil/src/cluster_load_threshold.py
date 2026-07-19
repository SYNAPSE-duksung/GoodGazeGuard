"""
정답률(accuracy) 기반 K-means 클러스터링으로
"정상 vs 과부하" 경계값을 데이터 기반으로 도출하고 시각화하는 스크립트
============================================================
입력: beh_all.csv (컬럼: subject, trial, condition, accuracy)

이 스크립트가 하는 일:
1. 정답률만 가지고 K-means로 2개 그룹(정상/과부하)을 자동으로 나눔
2. 두 그룹 사이의 경계값(threshold)을 계산
3. condition(5/9/13)별로 각 그룹에 얼마나 속하는지 확인
4. 결과를 그래프 3종으로 시각화하여 저장
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ----- 설정 -----
INPUT_PATH = "pupil/data/beh_all.csv"
OUTPUT_PREFIX = "load_threshold"


def run_clustering(beh: pd.DataFrame):
    """정답률로 K-means 클러스터링을 수행하고 결과를 반환."""
    X = beh[["accuracy"]].values
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    beh = beh.copy()
    beh["cluster"] = km.fit_predict(X)

    cluster_means = beh.groupby("cluster")["accuracy"].mean().sort_values()
    overload_cluster = cluster_means.index[0]  # 정답률 낮은 쪽 = 과부하
    normal_cluster = cluster_means.index[1]

    beh["label"] = beh["cluster"].map({overload_cluster: "과부하", normal_cluster: "정상"})

    overload_acc = beh[beh["cluster"] == overload_cluster]["accuracy"]
    normal_acc = beh[beh["cluster"] == normal_cluster]["accuracy"]
    threshold = (overload_acc.max() + normal_acc.min()) / 2

    return beh, threshold, cluster_means


def print_summary(beh: pd.DataFrame, threshold: float, cluster_means: pd.Series):
    print("=" * 60)
    print("K-means 클러스터링 결과")
    print("=" * 60)
    print(f"\n과부하 그룹 평균 정답률: {cluster_means.min():.3f}")
    print(f"정상 그룹 평균 정답률: {cluster_means.max():.3f}")
    print(f"\n추정 경계값(threshold): 정답률 약 {threshold:.3f}")
    print(f"  -> 정답률 {threshold:.1%} 미만이면 '과부하'로 분류")

    print("\ncondition별 분류 비율:")
    ct = pd.crosstab(beh["condition"], beh["label"], normalize="index").round(3)
    print(ct)

    return ct


def visualize(beh: pd.DataFrame, threshold: float, ct: pd.DataFrame):
    # 한글 폰트가 없는 환경(Colab 기본, 일부 로컬 환경)에서 글자가 깨지는 것을 방지하기 위해
    # 그래프 내 텍스트는 영어로 표기 (콘솔 출력은 한글 그대로 유지됨)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    label_map = {"정상": "Normal", "과부하": "Overload"}
    beh = beh.copy()
    beh["label_en"] = beh["label"].map(label_map)

    # ① 정답률 분포 히스토그램 + 경계선
    ax = axes[0]
    for label, color in [("Normal", "tab:blue"), ("Overload", "tab:red")]:
        subset = beh[beh["label_en"] == label]["accuracy"]
        ax.hist(subset, bins=30, alpha=0.6, label=label, color=color)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"threshold={threshold:.2f}")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Number of trials")
    ax.set_title("Accuracy distribution & K-means threshold")
    ax.legend()

    # ② condition별 정상/과부하 비율 막대그래프
    ax = axes[1]
    ct_en = ct.rename(columns=label_map)
    ct_en[["Normal", "Overload"]].plot(kind="bar", stacked=True, ax=ax, color=["tab:blue", "tab:red"])
    ax.set_xlabel("Condition (load intensity)")
    ax.set_ylabel("Proportion")
    ax.set_title("Normal/Overload ratio by condition")
    ax.legend(title="Label")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # ③ condition별 정답률 boxplot + 경계선
    ax = axes[2]
    data_by_cond = [beh[beh["condition"] == c]["accuracy"].dropna() for c in [5, 9, 13]]
    ax.boxplot(data_by_cond, tick_labels=["5", "9", "13"])
    ax.axhline(threshold, color="black", linestyle="--", linewidth=2, label=f"threshold={threshold:.2f}")
    ax.set_xlabel("Condition (load intensity)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy distribution by condition")
    ax.legend()

    plt.tight_layout()
    out_path = f"pupil/outputs/{OUTPUT_PREFIX}_visualization.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n그래프 저장 완료: {out_path}")
    plt.show()


if __name__ == "__main__":
    beh = pd.read_csv(INPUT_PATH).dropna(subset=["accuracy"])
    beh_clustered, threshold, cluster_means = run_clustering(beh)
    ct = print_summary(beh_clustered, threshold, cluster_means)
    visualize(beh_clustered, threshold, ct)

    beh_clustered.to_csv(f"pupil/outputs/{OUTPUT_PREFIX}_labeled.csv", index=False)
    print(f"\n라벨링된 데이터 저장 완료: {OUTPUT_PREFIX}_labeled.csv")