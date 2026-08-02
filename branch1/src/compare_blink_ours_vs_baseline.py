"""
compare_blink_ours_vs_baseline.py

raw blink 플래그에서 직접 계산한 blink feature(61초 제약 없이 계산해서
커버리지가 20~33% -> 91~98%로 올라간 재계산 버전)를 실제로 Branch1에
추가했을 때, baseline(pupil+gaze만)보다 진짜 더 잘 맞히는지 확인함.

주의(중요, 데이터 누수 관련)
------------------------------------------------
처음에 n_blinks_ours, blink_rate_per_min_ours, trial_duration_sec까지 다 넣고
돌렸더니 95~98% accuracy가 나왔는데, 이건 실력이 아니라 누수였음. trial이
길수록(13자리 조건) 당연히 시간이 더 걸리고 blink도 더 많이 쌓이기 때문에,
이 세 값이 사실상 condition(5/9/13)을 그대로 다시 알려주는 것과 같았음
(gaze의 num_samples/scanpath_length/fixation_count와 동일한 종류의 문제).
그래서 이 세 개는 train_branch1_lightgbm.py의 DURATION_LEAKING_BLINK_COLS로
빼고, "깜빡임 간격"을 보는 mean_ibi/std_ibi/blink_entropy_trial만 남겼음.
그 결과 baseline 78.2% -> 79.3%로, 훨씬 더 믿을 만한 수치가 나옴.

(참고) 이 스크립트는 pupil/gaze/blink 파일이 각각 따로 있어야 실행됨(baseline
용/blink 추가용 데이터를 따로 만들어야 해서 --merged 옵션은 없음). 이미 결론까지
낸 비교이므로, 이미 병합된 dataset/gaze_pupil_blink_merged.csv만 있다면 재실행은
필수 아니고 참고용으로만 봐도 됨. 결과 그래프는 output/branch1_blink_ours/figures/에
있음.

그래프 구성
------------
1. fig_accuracy_baseline_vs_blink.png
   random(33%) vs baseline(pupil+gaze) vs pupil+gaze+blink(재계산) 정확도 비교
2. fig_feature_importance_blink_ours.png
   pupil+gaze+blink(재계산) 모델의 feature importance 상위 15개 (blink는 다른 색으로 표시)
3. fig_confusion_matrix_blink_ours.png
   pupil+gaze+blink(재계산) 모델의 test set 혼동행렬
4. fig_blink_ibi_by_label.png
   mean_ibi_ours/std_ibi_ours가 Low/Medium/High별로 다르게 나타나는지 boxplot

사용법:
    python src/compare_blink_ours_vs_baseline.py \
        --pupil output/timeseries/pupil_trial_dataset_wide.csv \
        --gaze data/branch/feature_all_gaze.csv \
        --blink output/blink/blink_features_ours.csv \
        --out output/branch1_blink_ours
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_branch1_lightgbm import load_and_merge, build_feature_table

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


def fig_accuracy_comparison(acc_baseline, acc_blink, out_path):
    labels = ["Random guess\n(3-class)", "Baseline\n(pupil+gaze)",
              "pupil+gaze+blink\n(our recomputed version)"]
    values = [1 / 3 * 100, acc_baseline * 100, acc_blink * 100]
    colors = ["#a0aec0", "#2b6cb0", "#2f855a"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=11)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Branch1: baseline vs blink added (recomputed, 61s-window-free)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm_pct, cmap="Greens", vmin=0, vmax=100)
    ax.set_xticks(range(3)); ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticks(range(3)); ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("pupil+gaze+blink(ours) model: confusion matrix (test, row %)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_pct[i, j]:.1f}%\n(n={cm[i, j]})", ha="center", va="center",
                     color="white" if cm_pct[i, j] > 50 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, label="% of true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_importance(model, feature_cols, out_path, top_n=15):
    blink_feats = {"mean_ibi_ours", "std_ibi_ours", "blink_entropy_trial_ours"}
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=True).tail(top_n)
    colors = ["#2f855a" if f in blink_feats else "#2b6cb0" for f in importance["feature"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance["feature"], importance["importance"], color=colors)
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"pupil+gaze+blink(ours) model: top {top_n} feature importance\n(green = blink feature)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_ibi_by_label(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, col, name in zip(axes, ["mean_ibi_ours", "std_ibi_ours"], ["mean_ibi_ours", "std_ibi_ours"]):
        tmp = pd.DataFrame({"label": df["label"].values, "value": df[col].values}).dropna()
        data = [tmp.loc[tmp.label == lab, "value"].values for lab in LABEL_NAMES]
        bp = ax.boxplot(data, labels=LABEL_NAMES, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], ["#2b6cb0", "#68975f", "#c0392b"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel(name)
        ax.set_title(f"{name} by label")
    fig.suptitle("Blink inter-blink-interval stats by cognitive load label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pupil", default="output/timeseries/pupil_trial_dataset_wide.csv")
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--blink", required=True)
    parser.add_argument("--out", default="output/branch1_blink_ours")
    args = parser.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df_base = load_and_merge(args.pupil, args.gaze, None)
    df_blink = load_and_merge(args.pupil, args.gaze, args.blink)

    train_mask = (df_base["split"] == "train").values
    valid_mask = (df_base["split"] == "valid").values
    test_mask = (df_base["split"] == "test").values

    print("\n[1/2] baseline(pupil+gaze) 학습 중 ...")
    X_base, y_base, _ = build_feature_table(df_base, clean_signal=True)
    model_base = train_once(X_base, y_base, train_mask, valid_mask)
    pred_base = np.argmax(model_base.predict(X_base[test_mask]), axis=1)
    acc_base = accuracy_score(y_base[test_mask], pred_base)
    f1_base = f1_score(y_base[test_mask], pred_base, average="macro")
    print(f"  test accuracy: {acc_base:.3f} / macro-F1: {f1_base:.3f}")

    print("\n[2/2] pupil+gaze+blink(재계산) 학습 중 ...")
    X_blink, y_blink, feature_cols = build_feature_table(df_blink, clean_signal=True)
    model_blink = train_once(X_blink, y_blink, train_mask, valid_mask)
    pred_blink = np.argmax(model_blink.predict(X_blink[test_mask]), axis=1)
    acc_blink = accuracy_score(y_blink[test_mask], pred_blink)
    f1_blink = f1_score(y_blink[test_mask], pred_blink, average="macro")
    print(f"  test accuracy: {acc_blink:.3f} / macro-F1: {f1_blink:.3f}")

    print("\n그래프 생성 중 ...")
    fig_accuracy_comparison(acc_base, acc_blink, fig_dir / "fig_accuracy_baseline_vs_blink.png")
    fig_confusion(y_blink[test_mask], pred_blink, fig_dir / "fig_confusion_matrix_blink_ours.png")
    fig_importance(model_blink, feature_cols, fig_dir / "fig_feature_importance_blink_ours.png")
    fig_ibi_by_label(df_blink, fig_dir / "fig_blink_ibi_by_label.png")

    print(f"\n완료. {fig_dir}/ 에 4개 그래프 저장됨")
    print(f"  - baseline(pupil+gaze) test accuracy: {acc_base:.3f} / macro-F1: {f1_base:.3f}")
    print(f"  - pupil+gaze+blink(재계산) test accuracy: {acc_blink:.3f} / macro-F1: {f1_blink:.3f}")
    print(f"  - 차이: {(acc_blink - acc_base)*100:+.1f}%p")


if __name__ == "__main__":
    main()
