"""
train_branch1_lightgbm.py

브랜치1(pupil+gaze+blink Early Fusion) LightGBM 모델 학습 스크립트.
pupil/gaze/blink의 trial 단위 feature를 옆으로 합쳐서 하나의 표로 만들고, LightGBM 멀티클래스 분류로 condition 기반
공통 라벨(Low/Medium/High)을 예측함. 예측 결과(확률)는 나중에 소현님의 메타러너(Late Fusion)에서 브랜치2(rPPG) 결과와 합쳐질 입력값이 됨.

주의
------------------------------------------------
- pupil/gaze/blink 세 신호 다 갖춰서 학습함. blink는 raw blink 플래그에서 직접
  재계산한 버전(61초 윈도우 제약 없이 계산, 커버리지 91~98%)을 사용함.
- --clean-signal 옵션을 꼭 켜서 학습할 것. 이 옵션 없이 돌리면 trial 길이(condition)를
  간접적으로 드러내는 컬럼들(digit_1~13_zscore의 NaN 패딩 패턴, gaze의 num_samples/
  scanpath_length/fixation_count, blink의 n_blinks_ours/blink_rate_per_min_ours)이
  그대로 feature로 들어가서 정확도가 95~100%까지 비정상적으로 높게 나옴(데이터 누수
  -- 실제로 겪었던 문제임). DURATION_LEAKING_GAZE_COLS / DURATION_LEAKING_BLINK_COLS
  참고.
- 검증은 우리가 만든 participant_split.csv 기준 train/valid/test로 진행함. Group
  K-Fold로 바뀌면 이 split 대신 subject_id 기준으로 그룹을 새로 나눠서 쓰면 됨
  (한 사람의 trial이 train/test에 걸쳐 섞이면 안 됨).
- pupil의 trial_index(0-indexed)와 gaze의 trial_id(0-indexed)는 값이 동일한 것으로
  확인됨. blink도 subject_id/trial_index 컬럼 기준으로 이미 맞춰져 있음.

사용법
------------------------------------------------
(1) pupil+gaze+blink를 이미 하나로 합쳐둔 CSV가 있으면 --merged 하나로 충분함:
    python src/train_branch1_lightgbm.py \
        --merged dataset/gaze_pupil_blink_merged.csv \
        --out output/branch1 \
        --clean-signal

(2) pupil/gaze/blink 파일이 따로따로 있으면:
    python src/train_branch1_lightgbm.py \
        --pupil output/timeseries/pupil_trial_dataset_wide.csv \
        --gaze data/external/feature_all_gaze.csv \
        --blink output/blink/blink_features_ours.csv \
        --out output/branch1 \
        --clean-signal
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score

LABEL_TO_INT = {"Low": 0, "Medium": 1, "High": 2}

GAZE_FEATURE_COLS = [
    "movement_mean", "movement_std", "movement_max", "movement_min", "movement_median",
    "movement_p95", "movement_p99", "movement_iqr", "movement_cv", "movement_skew", "movement_kurtosis",
    "scanpath_length", "num_samples", "gaze_dispersion", "dispersion_x", "dispersion_y",
    "center_distance_mean", "center_distance_std", "center_distance_max",
    "gaze_velocity_mean", "gaze_velocity_std", "gaze_velocity_max",
    "acceleration_mean", "acceleration_std", "acceleration_max",
    "fixation_mean_duration", "fixation_max_duration", "fixation_count", "hull_area",
]

# trial 길이(=condition)에 비례해서 커지는 "총량/개수" 성격의 feature들.
# trial이 길수록(13자리) 당연히 샘플/경로/고정횟수가 더 많이 쌓이기 때문에,
# 이 값들은 사실상 condition을 그대로 다시 알려주는 것과 같음(데이터 누수).
# --clean-signal 모드에서는 이것들을 제외함.
DURATION_LEAKING_GAZE_COLS = {"num_samples", "scanpath_length", "fixation_count"}

# blink 쪽도 똑같은 문제가 있었음: n_blinks_ours(원시 개수)는 trial이 길수록
# 당연히 더 많이 쌓이므로 condition을 그대로 드러냄(실제로 넣고 돌려봤더니
# 95%까지 나와서 확인됨). blink_rate_per_min_ours(개수/시간)는 얼핏 시간으로
# 나눠서 괜찮아 보였지만, 실제로 빼고 비교해보니 이것도 제외해야 baseline과
# 비슷한(78~79%) 믿을 만한 수치가 나옴 -- 아마 tree 모델이 duration과 개수의
# 조합(이산적인 패턴)을 통해 여전히 condition을 간접적으로 알아내는 것으로 보임.
# mean_ibi/std_ibi/entropy(깜빡임 "간격"의 통계)는 이런 패턴이 없어서 그대로 둠.
DURATION_LEAKING_BLINK_COLS = {"n_blinks_ours", "blink_rate_per_min_ours", "trial_duration_sec"}


def load_pupil(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # remember 컬럼이 "True"/"False" 문자열로 저장돼있을 수 있어 안전하게 변환
    if df["remember"].dtype == object:
        df["remember"] = df["remember"].map({"True": True, "False": False})
    return df


def load_gaze(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"subject": "subject_id", "trial_id": "trial_index"})


def load_blink(path: str) -> pd.DataFrame:
    """blink feature 파일을 읽음. 두 가지 출처를 지원함:
    (1) clean_blink_dataset.py로 정리된 팀원 원본 blink 파일(PSD 포함,
        n_blinks 이상치 결측 처리됨), (2) raw blink 플래그에서 직접 계산한
        재계산 버전(61초 제약 없이 n_blinks 등 5개 지표만 있음, PSD 없음).
        둘 다 subject_id/trial_index 컬럼 기준으로 병합됨.
    """
    df = pd.read_csv(path)
    assert "subject_id" in df.columns and "trial_index" in df.columns, (
        f"{path}가 clean_blink_dataset.py를 거치지 않은 원본 파일인 것 같음 -- "
        f"먼저 `python src/clean_blink_dataset.py`로 정리부터 할 것"
    )
    return df


def load_and_merge(pupil_path: str, gaze_path: str, blink_path: str = None) -> pd.DataFrame:
    pupil = load_pupil(pupil_path)
    gaze = load_gaze(gaze_path)

    merged = pupil.merge(
        gaze[["subject_id", "trial_index"] + [c for c in GAZE_FEATURE_COLS if c in gaze.columns]],
        on=["subject_id", "trial_index"], how="inner",
    )
    print(f"pupil {len(pupil)}행 + gaze {len(gaze)}행 -> 병합 후 {len(merged)}행 "
          f"(둘 다 있는 trial만 남김)")

    if blink_path:
        blink = load_blink(blink_path)
        # pupil 쪽에 이미 있는 메타 컬럼(block_index/position_in_block/label/remember 등)이
        # blink 파일에도 같이 들어있는 경우가 있어서, 실제 blink 신호 feature만 골라내고
        # 메타 컬럼은 제외함 (안 그러면 병합 시 중복/충돌)
        NON_FEATURE_BLINK_COLS = {
            "subject_id", "trial_index", "task", "condition",
            "block_index", "position_in_block", "label", "remember",
            # trial_duration_sec: 재계산 스크립트가 참고용으로 같이 저장한 컬럼인데,
            # trial 길이는 곧 condition(5/9/13)을 거의 그대로 드러내는 값이라(gaze의
            # num_samples/scanpath_length와 같은 종류의 누수) feature로 쓰면 안 됨.
            # 실제로 넣고 돌려보니 98% 정확도(누수 확정)가 나와서 제외함.
            "trial_duration_sec",
        }
        blink_feature_cols = [c for c in blink.columns if c not in NON_FEATURE_BLINK_COLS]
        merged = merged.merge(
            blink[["subject_id", "trial_index"] + blink_feature_cols],
            on=["subject_id", "trial_index"], how="left",  # blink 없는 trial(control 등)은 NaN으로 남김
        )
        print(f"blink 붙인 후: {len(merged)}행 (blink 매칭 안 된 행은 NaN 유지)")

    return merged


def _pupil_summary_features(df: pd.DataFrame, digit_cols: list) -> pd.DataFrame:
    """digit_1..13_zscore(위치별 컬럼, NaN 패딩)을 그대로 쓰면 "몇 번째까지 값이
    있는지"가 곧 condition을 알려줘서 누수가 됨. 그래서 위치 정보를 없애고
    "실제로 존재하는 값들"만 가지고 요약 통계로 바꿔서 trial 길이를 직접
    노출하지 않게 함 (n_valid 같은 개수 자체도 넣지 않음)."""
    arr = df[digit_cols].values  # (n_trial, 13), NaN=패딩
    with np.errstate(all="ignore"):
        mean = np.nanmean(arr, axis=1)
        std = np.nanstd(arr, axis=1)
        mn = np.nanmin(arr, axis=1)
        mx = np.nanmax(arr, axis=1)

    first = np.full(len(arr), np.nan)
    last = np.full(len(arr), np.nan)
    slope = np.full(len(arr), np.nan)
    for i, row in enumerate(arr):
        valid_idx = np.where(~np.isnan(row))[0]
        if len(valid_idx) == 0:
            continue
        first[i] = row[valid_idx[0]]
        last[i] = row[valid_idx[-1]]
        if len(valid_idx) >= 2:
            slope[i] = np.polyfit(valid_idx, row[valid_idx], 1)[0]

    return pd.DataFrame({
        "pupil_mean": mean, "pupil_std": std, "pupil_min": mn, "pupil_max": mx,
        "pupil_first": first, "pupil_last": last, "pupil_slope": slope,
    }, index=df.index)


def build_feature_table(df: pd.DataFrame, clean_signal: bool = False):
    digit_cols = sorted([c for c in df.columns if c.startswith("digit_") and c.endswith("_zscore")])
    gaze_cols = [c for c in GAZE_FEATURE_COLS if c in df.columns]
    known_cols = set(["subject_id", "trial_index", "block_index", "position_in_block", "condition",
                       "label", "remember", "split", "correct", "accuracy", "personal_zscore",
                       "overload_percentile", "overload_label"])
    blink_cols = [c for c in df.columns
                  if c not in known_cols and c not in digit_cols and c not in gaze_cols]

    if clean_signal:
        # trial 길이(condition)를 간접적으로 노출하는 요소를 다 제거하고
        # "진짜 신호값" 위주로만 구성 -- 이게 얼마나 설명력 있는지 확인하는 모드
        gaze_cols = [c for c in gaze_cols if c not in DURATION_LEAKING_GAZE_COLS]
        blink_cols = [c for c in blink_cols if c not in DURATION_LEAKING_BLINK_COLS]
        pupil_feat = _pupil_summary_features(df, digit_cols)
        X = pd.concat([pupil_feat, df[gaze_cols + blink_cols].reset_index(drop=True)], axis=1)
        X.index = df.index
        feature_cols = list(X.columns)
    else:
        # 주의: condition(5/9/13)은 label(Low/Medium/High)을 그대로 결정하는 값이라
        # feature로 넣으면 데이터 누수. remember만 추가로 포함.
        feature_cols = digit_cols + gaze_cols + blink_cols + ["remember"]
        X = df[feature_cols].copy()
        X["remember"] = X["remember"].astype(int)

    y = df["label"].map(LABEL_TO_INT)

    mode = "clean-signal(누수 제거)" if clean_signal else "기본"
    print(f"\n[{mode}] 사용된 feature 개수: {len(feature_cols)}개 -> {feature_cols}")
    return X, y, feature_cols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pupil", default="output/timeseries/pupil_trial_dataset_wide.csv")
    parser.add_argument("--gaze", default=None, help="--merged를 안 쓸 경우 필수")
    parser.add_argument("--blink", default=None, help="아직 없으면 생략 가능 (나중에 받으면 경로 지정)")
    parser.add_argument("--merged", default=None,
                         help="pupil/gaze/blink를 이미 하나로 합쳐둔 CSV(예: "
                              "dataset/gaze_pupil_blink_merged.csv)가 있으면 이 옵션 "
                              "하나로 지정 가능 -- 지정 시 --pupil/--gaze/--blink는 무시됨")
    parser.add_argument("--out", default="output/branch1")
    parser.add_argument("--clean-signal", action="store_true",
                         help="trial 길이(condition)를 간접 노출하는 feature를 제거하고 "
                              "순수 신호 기반 요약 feature로만 학습 (누수 점검용, 항상 켜서 쓸 것)")
    args = parser.parse_args()

    if not args.merged and not args.gaze:
        parser.error("--merged를 안 쓰려면 --gaze는 필수임")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merged:
        print(f"이미 합쳐진 파일 사용: {args.merged}")
        df = pd.read_csv(args.merged)
        if df["remember"].dtype == object:
            df["remember"] = df["remember"].map({"True": True, "False": False})
    else:
        df = load_and_merge(args.pupil, args.gaze, args.blink)
    X, y, feature_cols = build_feature_table(df, clean_signal=args.clean_signal)

    train_mask = (df["split"] == "train").values
    valid_mask = (df["split"] == "valid").values
    test_mask = (df["split"] == "test").values
    print(f"train {train_mask.sum()}개 / valid {valid_mask.sum()}개 / test {test_mask.sum()}개")

    train_data = lgb.Dataset(X[train_mask], label=y[train_mask])
    valid_data = lgb.Dataset(X[valid_mask], label=y[valid_mask], reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "seed": 42,
    }

    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[train_data, valid_data], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    for name, mask in [("valid", valid_mask), ("test", test_mask)]:
        pred_proba = model.predict(X[mask], num_iteration=model.best_iteration)
        pred = np.argmax(pred_proba, axis=1)
        acc = accuracy_score(y[mask], pred)
        print(f"\n=== {name} 세트 (n={mask.sum()}) ===")
        print(f"accuracy: {acc:.3f}")
        print(classification_report(y[mask], pred, target_names=["Low", "Medium", "High"]))

    model_path = out_dir / "branch1_lightgbm.txt"
    model.save_model(str(model_path))

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    importance_path = out_dir / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    print(f"\n모델 저장: {model_path}")
    print(f"feature importance 저장: {importance_path}")
    print("\nfeature importance 상위 10개:")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
