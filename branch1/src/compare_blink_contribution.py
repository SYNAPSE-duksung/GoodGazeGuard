"""
compare_blink_contribution.py

blink이 전체 trial의 25~28%에만 있다 보니(팀원 원본 기준), "전체 trial 평균 정확도"로 비교하면 blink 효과가 나머지 75%(blink 없음)에 묻혀서 잘 안 보임.
그래서 **blink 값이 실제로 존재하는 trial만 따로 골라서**, 같은 trial들로
  (A) pupil+gaze만 쓴 모델
  (B) pupil+gaze+blink 다 쓴 모델
을 학습/비교함. 이러면 "blink를 넣었을 때 진짜 더 잘 맞히는지"를 공정하게 확인할 수 있음.

부가로: 전체 trial 기준으로 학습했을 때 blink/gaze feature들이 importance 순위
몇 등인지도 같이 출력함 (pupil 신호에 얼마나 묻히는지 확인용). feature 개수는
어떤 blink 파일을 쓰느냐에 따라 달라짐(팀원 원본은 PSD 포함 200개+, 재계산
버전은 blink 5개뿐이라 40개 이내).

(참고) 이 스크립트는 pupil/gaze/blink 파일이 각각 따로 있어야 실행됨 --
train_branch1_lightgbm.py처럼 --merged 옵션은 지원하지 않음. 이미 병합된
dataset/gaze_pupil_blink_merged.csv만 갖고 있다면 이 스크립트는 참고용으로만
보면 됨(이미 우리가 결론까지 낸 결과이므로 재실행이 꼭 필요하진 않음).

사용법:
    python src/compare_blink_contribution.py \
        --pupil output/timeseries/pupil_trial_dataset_wide.csv \
        --gaze data/branch/feature_all_gaze.csv \
        --blink output/branch1/blink_features_cleaned.csv \
        --out output/branch1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, classification_report

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_branch1_lightgbm import (
    load_and_merge, build_feature_table, LABEL_TO_INT, GAZE_FEATURE_COLS,
)


def train_eval(X, y, train_mask, valid_mask, test_mask, label: str):
    train_data = lgb.Dataset(X[train_mask], label=y[train_mask])
    valid_data = lgb.Dataset(X[valid_mask], label=y[valid_mask], reference=train_data)
    params = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
              "verbosity": -1, "seed": 42}
    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[valid_data], callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    pred = np.argmax(model.predict(X[test_mask]), axis=1)
    y_test = y[test_mask]
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    print(f"\n--- {label} (test n={test_mask.sum()}) ---")
    print(f"accuracy: {acc:.3f} / macro-F1: {macro_f1:.3f}")
    print(classification_report(y_test, pred, target_names=["Low", "Medium", "High"], zero_division=0))
    return model, acc, macro_f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pupil", default="output/timeseries/pupil_trial_dataset_wide.csv")
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--blink", required=True)
    parser.add_argument("--out", default="output/branch1")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(args.pupil, args.gaze, args.blink)

    # --- [1] 전체 trial 기준, blink/gaze feature가 importance 몇 등인지 ---
    X_all, y_all, feat_cols_all = build_feature_table(df, clean_signal=True)
    train_mask_all = (df["split"] == "train").values
    valid_mask_all = (df["split"] == "valid").values
    print("=" * 60)
    print(f"[1] 전체 trial({len(feat_cols_all)} feature) 기준 학습 -- blink/gaze 순위 확인용")
    print("=" * 60)
    model_all = lgb.train(
        {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss", "verbosity": -1, "seed": 42},
        lgb.Dataset(X_all[train_mask_all], label=y_all[train_mask_all]),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_all[valid_mask_all], label=y_all[valid_mask_all])],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    importance = pd.DataFrame({
        "feature": feat_cols_all,
        "importance": model_all.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importance["rank"] = importance.index + 1

    blink_cols = [c for c in feat_cols_all if c in df.columns and (
        c.startswith("blink_") or c in ("n_blinks", "mean_ibi", "std_ibi"))]
    gaze_cols = [c for c in feat_cols_all if c in GAZE_FEATURE_COLS]
    print(f"blink feature 중 최고 순위: {importance[importance.feature.isin(blink_cols)].head(3).to_string(index=False)}")
    print(f"\ngaze feature 중 최고 순위: {importance[importance.feature.isin(gaze_cols)].head(3).to_string(index=False)}")
    importance.to_csv(out_dir / "feature_importance_ranked.csv", index=False)

    # --- [2] blink 값이 실제로 존재하는 trial만 골라서 A/B 비교 ---
    print("\n" + "=" * 60)
    print("[2] blink 값 있는 trial만 골라서 pupil+gaze vs pupil+gaze+blink 비교")
    print("=" * 60)
    has_blink = df["n_blinks"].notna()
    print(f"전체 {len(df)}행 중 blink 값 있는 행: {has_blink.sum()}개 ({has_blink.mean()*100:.1f}%)")

    df_sub = df[has_blink].reset_index(drop=True)
    train_mask = (df_sub["split"] == "train").values
    valid_mask = (df_sub["split"] == "valid").values
    test_mask = (df_sub["split"] == "test").values
    print(f"이 부분집합 안에서 train {train_mask.sum()} / valid {valid_mask.sum()} / test {test_mask.sum()}")

    # (A) pupil+gaze만
    X_pg, y_pg, _ = build_feature_table(df_sub, clean_signal=True)
    blink_feat_cols = [c for c in X_pg.columns if c in blink_cols]
    X_pg_only = X_pg.drop(columns=blink_feat_cols)
    _, acc_a, f1_a = train_eval(X_pg_only, y_pg, train_mask, valid_mask, test_mask,
                                 "(A) pupil+gaze만 (blink 있는 trial 대상)")

    # (B) pupil+gaze+blink
    _, acc_b, f1_b = train_eval(X_pg, y_pg, train_mask, valid_mask, test_mask,
                                 "(B) pupil+gaze+blink (blink 있는 trial 대상)")

    print("\n" + "=" * 60)
    print("결론")
    print("=" * 60)
    print(f"(A) pupil+gaze만      : accuracy {acc_a:.3f} / macro-F1 {f1_a:.3f}")
    print(f"(B) pupil+gaze+blink  : accuracy {acc_b:.3f} / macro-F1 {f1_b:.3f}")
    diff = acc_b - acc_a
    print(f"차이: {diff:+.3f} ({'blink이 도움 됨' if diff > 0.01 else 'blink 효과 거의 없음' if abs(diff) <= 0.01 else 'blink 넣으니 오히려 나빠짐'})")


if __name__ == "__main__":
    main()
