"""
personalize_gaze_blink.py

gaze/blink feature를 사람별(subject_id 기준) 상대값(personal z-score)으로 바꿔서
학습해보고, 원본(절대값) 그대로 쓴 것과 성능을 비교함.

배경
------------------------------------------------
pupil은 이미 참가자 전체 trial 기준으로 z-score 정규화가 되어 있음
(src/pupil_preprocessing/pupil_config.py의 zscore_normalize, 참가자 개인
평균/표준편차 기준 -- 사람마다 원래 동공 크기/반응성이 다른 걸 제거하는 목적).
반면 gaze/blink는 지금까지 절대값 그대로 썼음. 사람마다 원래 눈 움직임 폭이나
평소 깜빡임 빈도가 다르니, pupil과 같은 방식으로 "그 사람 평소 대비 얼마나
벗어났는지"로 바꾸면 신호가 더 뚜렷해질 수 있다는 가설을 확인하는 스크립트.

방법
------------------------------------------------
각 gaze/blink feature 컬럼에 대해, subject_id별로 그 사람의 전체 trial 값의
평균/표준편차를 구해서 (value - 그사람평균) / 그사람표준편차 로 치환함.
참가자당 값이 2개 미만이거나 표준편차가 0이면(변화가 없으면) 정규화 불가로
보고 NaN 처리 (LightGBM은 NaN을 자체적으로 처리할 수 있어 그대로 둬도 됨).

주의
------------------------------------------------
- split(train/valid/test)이 참가자 단위로 나뉘어 있음(한 사람이 한 split에만
  속함, 84명 전부 확인됨) -- 그래서 그 사람의 모든 trial(자기 자신 것만)로
  평균/표준편차를 구해도 다른 split 정보가 섞여 들어가는 게 아님. 라벨을 쓰는
  것도 아니라서 데이터 누수 아님.
- DURATION_LEAKING 컬럼(누수 확정된 것들)은 애초에 build_feature_table에서
  제외되니 여기서 정규화해도 어차피 안 쓰임 -- 정규화는 살아남는 컬럼에만 적용

사용법:
    python src/branch1/personalize_gaze_blink.py \
        --merged output/merged_dataset/gaze_pupil_blink_merged.csv \
        --out output/branch1_personalized
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
    build_feature_table, GAZE_FEATURE_COLS,
    DURATION_LEAKING_GAZE_COLS, DURATION_LEAKING_BLINK_COLS,
)

BASELINE_TEST_ACC = 0.804  # 원본(정규화 없음) clean-signal 기준 현재 최종 모델


def personalize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    n_skipped = 0
    for col in cols:
        if col not in out.columns:
            continue
        z = pd.Series(np.nan, index=out.index)
        for sub, idx in out.groupby("subject_id").groups.items():
            vals = out.loc[idx, col]
            valid = vals.dropna()
            if len(valid) < 2 or valid.std() == 0 or not np.isfinite(valid.std()):
                n_skipped += 1
                continue
            mean, std = valid.mean(), valid.std()
            z.loc[idx] = (vals - mean) / std
        out[col] = z
    print(f"정규화 불가(표준편차 0 또는 값 부족)로 NaN 처리된 (컬럼 x 참가자) 조합: {n_skipped}개")
    return out


def train_and_eval(X, y, train_mask, valid_mask, test_mask, label):
    train_data = lgb.Dataset(X[train_mask], label=y[train_mask])
    valid_data = lgb.Dataset(X[valid_mask], label=y[valid_mask], reference=train_data)
    params = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
              "verbosity": -1, "seed": 42}
    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[valid_data], callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    pred = np.argmax(model.predict(X[test_mask], num_iteration=model.best_iteration), axis=1)
    acc = accuracy_score(y[test_mask], pred)
    f1 = f1_score(y[test_mask], pred, average="macro")
    print(f"\n=== {label} ===")
    print(f"test accuracy: {acc:.3f} / macro-F1: {f1:.3f}")
    print(classification_report(y[test_mask], pred, target_names=["Low", "Medium", "High"]))
    return model, acc, f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", default="output/merged_dataset/gaze_pupil_blink_merged.csv")
    parser.add_argument("--out", default="output/branch1_personalized")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.merged)
    if df["remember"].dtype == object:
        df["remember"] = df["remember"].map({"True": True, "False": False})

    train_mask = (df["split"] == "train").values
    valid_mask = (df["split"] == "valid").values
    test_mask = (df["split"] == "test").values

    # 정규화 대상: 누수 컬럼 뺀 실제 살아남는 gaze/blink feature만
    gaze_cols = [c for c in GAZE_FEATURE_COLS if c in df.columns and c not in DURATION_LEAKING_GAZE_COLS]
    known_cols = {"subject_id", "trial_index", "block_index", "position_in_block", "condition",
                  "label", "remember", "split", "correct", "accuracy", "personal_zscore",
                  "overload_percentile", "overload_label"}
    digit_cols = [c for c in df.columns if c.startswith("digit_") and c.endswith("_zscore")]
    blink_cols = [c for c in df.columns if c not in known_cols and c not in digit_cols
                  and c not in GAZE_FEATURE_COLS and c not in DURATION_LEAKING_BLINK_COLS]
    print(f"정규화 대상 gaze 컬럼 {len(gaze_cols)}개, blink 컬럼 {len(blink_cols)}개")

    # 1) 원본(정규화 없음) -- 비교 기준
    X0, y0, _ = build_feature_table(df, clean_signal=True)
    _, acc_raw, f1_raw = train_and_eval(X0, y0, train_mask, valid_mask, test_mask, "원본 (정규화 없음)")

    # 2) 개인별 정규화 적용
    df_personalized = personalize(df, gaze_cols + blink_cols)
    X1, y1, feature_cols = build_feature_table(df_personalized, clean_signal=True)
    model_p, acc_p, f1_p = train_and_eval(X1, y1, train_mask, valid_mask, test_mask, "gaze/blink 개인별 정규화 적용")

    print(f"\n{'='*60}")
    print(f"원본:        test accuracy {acc_raw:.3f} (macro-F1 {f1_raw:.3f})")
    print(f"개인화 적용: test accuracy {acc_p:.3f} (macro-F1 {f1_p:.3f})")
    print(f"차이: {(acc_p - acc_raw)*100:+.1f}%p")
    print(f"{'='*60}")

    model_path = out_dir / "branch1_lightgbm_personalized.txt"
    model_p.save_model(str(model_path))

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model_p.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    importance.to_csv(out_dir / "feature_importance_personalized.csv", index=False)
    print(f"\n모델 저장: {model_path}")
    print("\nfeature importance 상위 10개 (정규화 적용 모델 기준):")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
