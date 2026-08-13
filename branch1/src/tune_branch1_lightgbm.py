"""
tune_branch1_lightgbm.py

Branch1(pupil+gaze+blink) LightGBM 모델의 하이퍼파라미터(모델 학습 방식을 조절하는
설정값들)를 랜덤 탐색으로 찾아서, 현재 80.4% 정확도를 더 끌어올릴 수 있는지 확인함.

방법
------------------------------------------------
- learning_rate(한 걸음에 얼마나 크게 학습할지), num_leaves(트리 하나의 복잡도),
  min_child_samples(가지를 나눌 때 필요한 최소 샘플 수, 클수록 과적합 방지),
  feature_fraction/bagging_fraction(매 라운드마다 feature/샘플을 얼마나 랜덤하게
  일부만 써서 과적합을 줄일지), lambda_l1/l2(가중치가 너무 커지지 않게 누르는 정도)
  를 정해둔 범위 안에서 랜덤하게 N개 조합 뽑아서 하나씩 학습해봄
- 각 조합은 train으로 학습, valid로 early stopping 판단 -> valid accuracy로 비교
- valid accuracy가 제일 높았던 조합을 최종 후보로 뽑아서 test accuracy까지 확인
  (valid로 고르고 test로 마지막 검증하는 것 -- test를 탐색에 아예 안 써야 "진짜"
  일반화 성능임)

주의
------------------------------------------------
- 이건 단일 train/valid/test 분할 기준 결과라, Group K-Fold로 바뀌면 최적 조합이
  달라질 수 있음 (지금은 "쉽게 시도해볼 수 있는 개선 방법"을 확인하는 용도)
- optuna 같은 전문 탐색 라이브러리 없이 순수 랜덤 탐색만 씀 (의존성 추가 안 하려고,
  N=40 정도면 충분히 쓸만한 조합을 찾을 수 있음)

사용법:
    python src/branch1/tune_branch1_lightgbm.py \
        --merged output/merged_dataset/gaze_pupil_blink_merged.csv \
        --out output/branch1_tuned \
        --n-trials 40
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, classification_report

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_branch1_lightgbm import build_feature_table

BASELINE_TEST_ACC = 0.804  # --clean-signal 기본 파라미터 기준 현재 최종 모델 (비교 기준)


def sample_params(rng: random.Random) -> dict:
    return {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "seed": 42,
        "learning_rate": rng.choice([0.03, 0.05, 0.08, 0.1]),
        "num_leaves": rng.choice([15, 31, 63, 127]),
        "max_depth": rng.choice([-1, 4, 6, 8]),
        "min_child_samples": rng.choice([10, 20, 30, 50, 80]),
        "feature_fraction": rng.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
        "bagging_fraction": rng.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
        "bagging_freq": rng.choice([0, 1, 5]),
        "lambda_l1": rng.choice([0.0, 0.1, 0.5, 1.0]),
        "lambda_l2": rng.choice([0.0, 0.1, 0.5, 1.0]),
    }


def train_and_eval(params, X, y, train_mask, valid_mask, test_mask):
    train_data = lgb.Dataset(X[train_mask], label=y[train_mask])
    valid_data = lgb.Dataset(X[valid_mask], label=y[valid_mask], reference=train_data)

    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[valid_data], callbacks=[lgb.early_stopping(25, verbose=False)],
    )

    valid_pred = np.argmax(model.predict(X[valid_mask], num_iteration=model.best_iteration), axis=1)
    valid_acc = accuracy_score(y[valid_mask], valid_pred)

    test_pred = np.argmax(model.predict(X[test_mask], num_iteration=model.best_iteration), axis=1)
    test_acc = accuracy_score(y[test_mask], test_pred)
    test_f1 = f1_score(y[test_mask], test_pred, average="macro")

    return model, valid_acc, test_acc, test_f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", default="output/merged_dataset/gaze_pupil_blink_merged.csv")
    parser.add_argument("--out", default="output/branch1_tuned")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"로드: {args.merged}")
    df = pd.read_csv(args.merged)
    if df["remember"].dtype == object:
        df["remember"] = df["remember"].map({"True": True, "False": False})

    X, y, feature_cols = build_feature_table(df, clean_signal=True)
    train_mask = (df["split"] == "train").values
    valid_mask = (df["split"] == "valid").values
    test_mask = (df["split"] == "test").values
    print(f"train {train_mask.sum()} / valid {valid_mask.sum()} / test {test_mask.sum()}")

    rng = random.Random(args.seed)
    results = []
    best_valid_acc = -1.0
    best_params = None
    best_model = None
    best_test_acc = None
    best_test_f1 = None

    print(f"\n랜덤 탐색 시작 ({args.n_trials}개 조합, valid accuracy 기준으로 최고 선택) ...")
    for i in range(args.n_trials):
        params = sample_params(rng)
        model, valid_acc, test_acc, test_f1 = train_and_eval(params, X, y, train_mask, valid_mask, test_mask)
        results.append({**{k: v for k, v in params.items() if k not in
                            ("objective", "num_class", "metric", "verbosity", "seed")},
                         "valid_acc": valid_acc, "test_acc": test_acc, "test_f1": test_f1,
                         "best_iteration": model.best_iteration})
        marker = ""
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_params = params
            best_model = model
            best_test_acc = test_acc
            best_test_f1 = test_f1
            marker = "  <- new best (valid 기준)"
        print(f"[{i+1:2d}/{args.n_trials}] valid={valid_acc:.3f} test={test_acc:.3f}{marker}")

    results_df = pd.DataFrame(results).sort_values("valid_acc", ascending=False)
    results_path = out_dir / "tuning_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\n{'='*60}")
    print(f"기존(clean-signal 기본 파라미터) test accuracy: {BASELINE_TEST_ACC:.3f} (참고용, 이번 탐색과 다른 실행이라 약간의 노이즈 있을 수 있음)")
    print(f"탐색 중 valid accuracy 최고 조합의 test accuracy: {best_test_acc:.3f} (macro-F1 {best_test_f1:.3f})")
    print(f"차이: {(best_test_acc - BASELINE_TEST_ACC)*100:+.1f}%p")
    print(f"{'='*60}")
    print("\n최고 조합 파라미터:")
    for k, v in best_params.items():
        if k not in ("objective", "num_class", "metric", "verbosity", "seed"):
            print(f"  {k}: {v}")

    model_path = out_dir / "branch1_lightgbm_tuned.txt"
    best_model.save_model(str(model_path))

    pred = np.argmax(best_model.predict(X[test_mask], num_iteration=best_model.best_iteration), axis=1)
    print("\n최종 모델 test set 리포트:")
    print(classification_report(y[test_mask], pred, target_names=["Low", "Medium", "High"]))

    print(f"\n탐색 결과 저장: {results_path}")
    print(f"최고 모델 저장: {model_path}")


if __name__ == "__main__":
    main()
