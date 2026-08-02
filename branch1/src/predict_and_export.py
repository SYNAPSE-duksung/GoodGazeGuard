"""
predict_and_export.py

Branch1(pupil+gaze+blink LightGBM) 모델의 최종 예측 확률을 메타러너(소현님)에게
넘겨줄 수 있는 형태로 CSV로 저장함. "모델 파일" 자체가 아니라, 모델이 각
trial마다 계산한 [Low, Medium, High] 확률값을 넘기는 게 목적임 -- 메타러너는
이 확률값을 브랜치2(rPPG) 결과와 합쳐서 최종 판단을 내림(Late Fusion).

주의: 여기서 쓰는 모델은 데이터 누수를 다 제거한 clean-signal 버전이고,
blink은 raw blink 플래그에서 직접 계산한 재계산 버전(61초 제약 없이 계산,
커버리지 91~98%)을 사용함. (경과: baseline 78.2% -> blink 추가 후 79.3%,
자세한 배경은 Branch1_모델_보고서.docx 참고)

(참고) 이 스크립트는 pupil/gaze/blink 파일이 각각 따로 있어야 실행됨(현재는
--merged 옵션 미지원). 이미 병합된 dataset/gaze_pupil_blink_merged.csv만
갖고 있고 예측 확률 CSV를 다시 뽑아야 하면, train_branch1_lightgbm.py의
--merged 예시처럼 pd.read_csv로 바로 불러오게 고쳐 쓰면 됨.

사용법:
    python src/predict_and_export.py \
        --pupil output/timeseries/pupil_trial_dataset_wide.csv \
        --gaze data/branch/feature_all_gaze.csv \
        --blink output/blink/blink_features_ours.csv \
        --out output/handoff_to_metalearner/branch1_predictions.csv
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
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--blink", required=True)
    parser.add_argument("--out", default="output/handoff_to_metalearner/branch1_predictions.csv")
    args = parser.parse_args()

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
    model = lgb.train(params, train_data, num_boost_round=305)  # 이전 학습에서 확인된 best_iteration 근처

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
