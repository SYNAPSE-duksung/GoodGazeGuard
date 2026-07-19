"""
발표자료용 그래프 생성 스크립트
============================================================
pupil_baseline_pipeline.py 실행 후 나온 결과 파일(outputs/pupil_load_score_labeled.csv)을
읽어서, 결과를 보다 쉽게 이해하기 위해 시각화한다.

① condition별 평균 부하 정도(%) 막대그래프
② feature 중요도 막대그래프
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ----- 경로 설정 -----
LABELED_PATH = "pupil/outputs/pupil_load_score_labeled.csv"
FIG1_PATH = "pupil/outputs/fig_condition_load_pct.png"
FIG2_PATH = "pupil/outputs/fig_feature_importance.png"
FIG3_PATH = "pupil/outputs/fig_feature_correlation_heatmap.png"

FEATURE_COLS = ["peak_val", "peak_pos", "early_mean", "late_val", "decline", "plateau_len", "volatility"]
RF_PARAMS = dict(n_estimators=400, max_depth=8, min_samples_leaf=3, max_features=None, random_state=42)


def plot_condition_load_pct(df: pd.DataFrame):
    """condition별 평균 부하 정도(%) 막대그래프"""
    means = df.groupby("condition")["load_score_pct"].mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(means.index.astype(str), means.values, color=["#4C9AFF", "#FFAB00", "#FF5630"])
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}%",
                 ha="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Condition (digit span)")
    ax.set_ylabel("Average Load Score (%)")
    ax.set_title("Condition-wise Average Cognitive Load Score")
    ax.set_ylim(0, 100)
    ax.axhline(40.5, color="gray", linestyle="--", linewidth=1, label="Threshold (40.5%)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIG1_PATH, dpi=150)
    plt.close()
    print(f"저장 완료: {FIG1_PATH}")


def plot_feature_importance(df: pd.DataFrame):
    """feature 중요도 막대그래프 (모델을 다시 학습시켜 중요도 산출)"""
    X = df[FEATURE_COLS].values
    y = df["accuracy_z"].values

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(importances.index, importances.values, color="#36B37E")
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.1%}",
                 va="center", fontsize=10)

    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Random Forest)")
    ax.set_xlim(0, max(importances.values) * 1.25)

    plt.tight_layout()
    plt.savefig(FIG2_PATH, dpi=150)
    plt.close()
    print(f"저장 완료: {FIG2_PATH}")


def plot_correlation_heatmap(df: pd.DataFrame):
    """feature 간 상관관계 heatmap (다중공선성/중복 여부 확인용)"""
    corr = df[FEATURE_COLS].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right")
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_yticklabels(FEATURE_COLS)
    for i in range(len(FEATURE_COLS)):
        for j in range(len(FEATURE_COLS)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if abs(val) > 0.5 else "black", fontsize=9)
    ax.set_title("Feature Correlation Heatmap")
    plt.colorbar(im)

    plt.tight_layout()
    plt.savefig(FIG3_PATH, dpi=150)
    plt.close()
    print(f"저장 완료: {FIG3_PATH}")

    # 절대값 0.7 이상(심각한 중복)인 쌍 출력
    high_corr = []
    for i in range(len(FEATURE_COLS)):
        for j in range(i + 1, len(FEATURE_COLS)):
            if abs(corr.iloc[i, j]) >= 0.7:
                high_corr.append((FEATURE_COLS[i], FEATURE_COLS[j], corr.iloc[i, j]))
    if high_corr:
        print("  ⚠️ 상관계수 0.7 이상(다중공선성 우려) 쌍:")
        for a, b, v in high_corr:
            print(f"    {a} - {b}: {v:.2f}")
    else:
        print("  상관계수 0.7 이상인 쌍 없음 (심각한 다중공선성 없음)")


if __name__ == "__main__":
    df = pd.read_csv(LABELED_PATH)
    print(f"데이터 로딩 완료: {len(df)}개 시행")

    plot_condition_load_pct(df)
    plot_feature_importance(df)
    plot_correlation_heatmap(df)

    print("\n세 그래프 모두 생성 완료. outputs 폴더를 확인하세요.")