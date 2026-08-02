"""
visualize_branch1_results.py

train_branch1_lightgbm.py에서 확인한 두 가지 버전(원래 방식=데이터 누수 있음 vs
clean-signal=누수 제거)을 둘 다 학습시켜서 비교하고, 그 결과를 그래프로 남김.
"왜 --clean-signal 옵션을 항상 켜서 써야 하는지"를 팀/기록용으로 보여주는 용도.

그래프 구성
------------
1. fig_accuracy_comparison.png
   랜덤 찍기(33%) vs 원래 방식(구조적 누수 있음) vs clean-signal(진짜 신호만) 정확도 비교 막대그래프
2. fig_confusion_matrix.png
   clean-signal 모델의 test set 혼동행렬(실제 라벨 vs 예측 라벨)
3. fig_feature_importance.png
   clean-signal 모델의 feature importance 상위 15개
4. fig_pupil_signal_by_label.png
   가장 중요한 pupil feature(slope, std)가 Low/Medium/High별로 실제로
   다르게 나타나는지 boxplot으로 확인 -- "왜 이 feature가 예측에 쓰였는지"에 대한 생리학적 근거를 보여줌

(참고) 이 스크립트는 pupil/gaze/blink 파일이 각각 따로 있어야 실행됨(현재는
--merged 옵션 미지원, --blink는 생략 가능). 이미 병합된
dataset/gaze_pupil_blink_merged.csv만 갖고 있다면 참고용으로만 보면 됨 --
이미 결론(누수 있으면 95~100%, 제거하면 78~79%)까지 낸 비교라 재실행이 꼭
필요하진 않음.

사용법:
    python src/visualize_branch1_results.py \
        --pupil output/timeseries/pupil_trial_dataset_wide.csv \
        --gaze data/external/feature_all_gaze.csv \
        --out output/branch1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_branch1_lightgbm import load_and_merge, build_feature_table, LABEL_TO_INT

LABEL_NAMES = ["Low", "Medium", "High"]


def train_once(X, y, train_mask, valid_mask):
    train_data = lgb.Dataset(X[train_mask], label=y[train_mask])
    valid_data = lgb.Dataset(X[valid_mask], label=y[valid_mask], reference=train_data)
    params = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
              "verbosity": -1, "seed": 42}
    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[valid_data], callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


def fig_accuracy_comparison(acc_leaky, acc_clean, out_path):
    labels = ["Random guess\n(3-class baseline)", "Original features\n(length leaks via padding)",
              "Clean-signal features\n(length info removed)"]
    values = [1 / 3 * 100, acc_leaky * 100, acc_clean * 100]
    colors = ["#a0aec0", "#c0392b", "#2b6cb0"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=11)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Branch1 accuracy: random vs leaky features vs clean-signal features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(3)); ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticks(range(3)); ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Clean-signal model: confusion matrix (test set, row %)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_pct[i, j]:.1f}%\n(n={cm[i, j]})", ha="center", va="center",
                     color="white" if cm_pct[i, j] > 50 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, label="% of true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_importance(model, feature_cols, out_path, top_n=15):
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance["feature"], importance["importance"], color="#2b6cb0")
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"Clean-signal model: top {top_n} feature importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_signal_by_label(df, feature_series, feature_name, out_path):
    tmp = pd.DataFrame({"label": df["label"].values, "value": feature_series.values}).dropna()
    data = [tmp.loc[tmp.label == lab, "value"].values for lab in LABEL_NAMES]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    bp = ax.boxplot(data, labels=LABEL_NAMES, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#2b6cb0", "#68975f", "#c0392b"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel(feature_name)
    ax.set_title(f"{feature_name} by condition-based label (Low/Medium/High)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pupil", default="output/timeseries/pupil_trial_dataset_wide.csv")
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--blink", default=None)
    parser.add_argument("--out", default="output/branch1")
    args = parser.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(args.pupil, args.gaze, args.blink)
    train_mask = (df["split"] == "train").values
    valid_mask = (df["split"] == "valid").values
    test_mask = (df["split"] == "test").values

    print("\n[1/2] 원래 방식(길이 정보 그대로 남아있는 feature)으로 학습 중 ...")
    X_leaky, y_leaky, _ = build_feature_table(df, clean_signal=False)
    model_leaky = train_once(X_leaky, y_leaky, train_mask, valid_mask)
    pred_leaky = np.argmax(model_leaky.predict(X_leaky[test_mask]), axis=1)
    acc_leaky = accuracy_score(y_leaky[test_mask], pred_leaky)
    print(f"  test accuracy: {acc_leaky:.3f}")

    print("\n[2/2] clean-signal(길이 정보 제거) 방식으로 학습 중 ...")
    X_clean, y_clean, feature_cols = build_feature_table(df, clean_signal=True)
    model_clean = train_once(X_clean, y_clean, train_mask, valid_mask)
    pred_clean = np.argmax(model_clean.predict(X_clean[test_mask]), axis=1)
    acc_clean = accuracy_score(y_clean[test_mask], pred_clean)
    print(f"  test accuracy: {acc_clean:.3f}")

    print("\n그래프 생성 중 ...")
    fig_accuracy_comparison(acc_leaky, acc_clean, fig_dir / "fig_accuracy_comparison.png")
    fig_confusion(y_clean[test_mask], pred_clean, fig_dir / "fig_confusion_matrix.png")
    fig_importance(model_clean, feature_cols, fig_dir / "fig_feature_importance.png")
    fig_signal_by_label(df, X_clean["pupil_slope"], "pupil_slope", fig_dir / "fig_pupil_slope_by_label.png")
    fig_signal_by_label(df, X_clean["pupil_std"], "pupil_std", fig_dir / "fig_pupil_std_by_label.png")

    print(f"\n완료. {fig_dir}/ 에 5개 그래프 저장됨")
    print(f"  - 원래 방식 test accuracy: {acc_leaky:.3f}")
    print(f"  - clean-signal test accuracy: {acc_clean:.3f}")
    print(f"  - random baseline: {1/3:.3f}")


if __name__ == "__main__":
    main()
