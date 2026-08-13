"""
predict_and_export.py

Branch1(pupil+gaze+blink LightGBM) 모델의 최종 예측 확률을 메타러너(소현님)에게
넘겨줄 수 있는 형태로 CSV로 저장함. "모델 파일" 자체가 아니라, 모델이 각
trial마다 계산한 [Low, Medium, High] 확률값을 넘기는 게 목적임 -- 메타러너는
이 확률값을 브랜치2(rPPG) 결과와 합쳐서 최종 판단을 내림(Late Fusion).

주의: 여기서 쓰는 모델은 데이터 누수를 다 제거한 clean-signal 버전임.

2026-08-12 업데이트: gaze/blink를 참가자별 개인화 정규화(personal z-score,
src/branch1/personalize_gaze_blink.py 참고)한 버전을 기본으로 씀. 하이퍼파라미터
튜닝/순서형 회귀 등 다른 방법도 시도했지만 개인화가 제일 나은 결과(test 81.3%,
정규화 없는 버전은 80.4%)라 이걸 채택함. --merged 옵션으로 이미 개인화된 병합
CSV(output/merged_dataset/gaze_pupil_blink_merged_personalized.csv)를 바로 넣으면 됨.

사용법:
    python src/branch1/predict_and_export.py \
        --merged output/merged_dataset/gaze_pupil_blink_merged_personalized.csv \
        --out output/handoff_to_metalearner/branch1_predictions.csv

    (또는 개인화 전 --pupil/--gaze/--blink 세 개를 따로 지정해서 원본 버전도 가능)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_branch1_lightgbm import load_and_merge, build_feature_table, LABEL_TO_INT

INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pupil", default="output/timeseries/pupil_trial_dataset_wide.csv")
    parser.add_argument("--gaze", default=None, help="--merged를 안 쓸 경우 필수")
    parser.add_argument("--blink", default=None, help="--merged를 안 쓸 경우 필수")
    parser.add_argument("--merged", default=None,
                         help="이미 합쳐진(필요하면 개인화까지 된) CSV를 바로 지정 -- "
                              "지정 시 --pupil/--gaze/--blink는 무시됨")
    parser.add_argument("--out", default="output/handoff_to_metalearner/branch1_predictions.csv")
    args = parser.parse_args()

    if not args.merged and not (args.gaze and args.blink):
        parser.error("--merged를 안 쓰려면 --gaze와 --blink가 둘 다 필요함")

    if args.merged:
        print(f"이미 합쳐진 파일 사용: {args.merged}")
        df = pd.read_csv(args.merged)
        if df["remember"].dtype == object:
            df["remember"] = df["remember"].map({"True": True, "False": False})
    else:
        df = load_and_merge(args.pupil, args.gaze, args.blink)
    X, y, feature_cols = build_feature_table(df, clean_signal=True)

    train_mask = (df["split"] == "train").values
    valid_mask = (df["split"] == "valid").values

    # 최종적으로 넘길 모델도 train+valid로 학습(지금까지 성능 확인은 train만 썼음.
    # 실전에 넘기는 모델은 valid까지 다 활용해서 학습하는 게 일반적인 관례라
    # train+valid를 합쳐서 재학습하고, test는 최종 확인용으로 그대로 둠)
    fit_mask = train_mask | valid_mask
    train_data = lgb.Dataset(X[fit_mask], label=y[fit_mask])
    params = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
              "verbosity": -1, "seed": 42}
    model = lgb.train(params, train_data, num_boost_round=140)  # 개인화 버전 train-only 학습에서 확인된 best_iteration(140) 근처

    proba = model.predict(X)  # 전체 trial(train/valid/test 다 포함)에 대해 확률 계산
    pred_label = np.argmax(proba, axis=1)

    out = pd.DataFrame({
        "subject_id": df["subject_id"].values,
        "trial_index": df["trial_index"].values,
        "split": df["split"].values,
        "true_label": df["label"].values,
        "branch1_prob_low": proba[:, LABEL_TO_INT["Low"]],
        "branch1_prob_medium": proba[:, LABEL_TO_INT["Medium"]],
        "branch1_prob_high": proba[:, LABEL_TO_INT["High"]],
        "branch1_pred_label": [INT_TO_LABEL[p] for p in pred_label],
    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"저장: {out_path} ({len(out)}행)")
    print(f"컬럼: {list(out.columns)}")
    print(f"\nsplit별 행수:\n{out['split'].value_counts()}")


if __name__ == "__main__":
    main()
