"""
cognitive_load_pipeline.py
---------------------------
UBFC-rPPG 데이터셋 전체 subject 폴더를 순회하며:
  1. 영상 -> rPPG 신호 추출 (rppg_pos.py)
  2. rPPG -> HRV 윈도우 특징 추출 (hrv_features.py)
  3. 라벨(labels.csv) 매칭
  4. RandomForest 분류기 학습 및 subject-independent 평가(Leave-One-Subject-Out)

--------------------------------------------------------------------
[중요] 라벨(label)에 대하여
--------------------------------------------------------------------
UBFC-rPPG 자체에는 "인지 과부하" 라벨이 없습니다. 두 가지 방식 중 하나로
라벨을 준비해서 사용하세요.

(A) DATASET_2 프로토콜 기반 (안정 상태 -> 시간제한 수학 게임):
    각 subject 폴더에 대해 baseline 구간과 task(과부하) 구간의
    시작/끝 시간(초)을 알고 있다면 아래 형식의 CSV를 준비:

        subject,start_sec,end_sec,label
        subject1,0,60,0        # 0 = baseline(저부하)
        subject1,60,180,1      # 1 = task(고부하/스트레스)
        subject2,0,55,0
        subject2,55,190,1
        ...

(B) 시선/동공 팀에서 이미 세션 단위 과부하 라벨을 산출해준 경우:
    동일한 CSV 포맷(subject,start_sec,end_sec,label)으로 맞춰서 넣으면 됨.

라벨 CSV 없이 실행하면, 학습 없이 특징 추출 결과(csv)만 저장합니다.

--------------------------------------------------------------------
데이터셋 폴더 구조 가정 (UBFC-rPPG DATASET_2 기준)
--------------------------------------------------------------------
    UBFC_ROOT/
        subject1/
            vid.avi
            ground_truth.txt
        subject2/
            vid.avi
            ground_truth.txt
        ...

사용법:
    python cognitive_load_pipeline.py \
        --data_root /path/to/UBFC_ROOT \
        --labels /path/to/labels.csv \
        --out_dir ./results
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

from rppg_pos import extract_rppg_from_video
from hrv_features import extract_windowed_features


FEATURE_COLUMNS = [
    "mean_hr", "sdnn", "rmssd", "pnn50",
    "lf_power", "hf_power", "lf_hf_ratio",
    "signal_std", "signal_skew", "signal_kurtosis",
]


# -----------------------------------------------------------------
# 1. 전체 subject 순회하며 특징 추출
# -----------------------------------------------------------------
def build_feature_table(data_root: str, window_sec=10.0, step_sec=5.0, use_mediapipe=True) -> pd.DataFrame:
    subject_dirs = sorted(glob.glob(os.path.join(data_root, "*")))
    all_rows = []

    for subj_dir in subject_dirs:
        if not os.path.isdir(subj_dir):
            continue
        subject_id = os.path.basename(subj_dir)
        video_path = os.path.join(subj_dir, "vid.avi")
        if not os.path.exists(video_path):
            print(f"[skip] vid.avi 없음: {subj_dir}")
            continue

        print(f"[처리중] {subject_id} ...")
        try:
            rppg_signal, fps, _ = extract_rppg_from_video(video_path, use_mediapipe=use_mediapipe)
        except Exception as e:
            print(f"  [실패] rPPG 추출 오류: {e}")
            continue

        feat_df = extract_windowed_features(rppg_signal, fps, window_sec, step_sec)
        feat_df["subject"] = subject_id
        all_rows.append(feat_df)
        print(f"  -> 윈도우 {len(feat_df)}개 추출 완료 (fps={fps:.1f})")

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


# -----------------------------------------------------------------
# 2. 라벨 CSV를 윈도우 특징 테이블에 매칭
# -----------------------------------------------------------------
def attach_labels(feature_df: pd.DataFrame, labels_csv: str) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_csv)

    def find_label(row):
        subj_labels = labels_df[labels_df["subject"] == row["subject"]]
        window_mid = (row["window_start_sec"] + row["window_end_sec"]) / 2.0
        match = subj_labels[
            (subj_labels["start_sec"] <= window_mid) & (window_mid < subj_labels["end_sec"])
        ]
        if len(match) == 0:
            return np.nan
        return match.iloc[0]["label"]

    feature_df = feature_df.copy()
    feature_df["label"] = feature_df.apply(find_label, axis=1)
    return feature_df.dropna(subset=["label"])


# -----------------------------------------------------------------
# 3. Leave-One-Subject-Out 학습/평가
# -----------------------------------------------------------------
def train_and_evaluate(feature_df: pd.DataFrame, out_dir: str):
    X = feature_df[FEATURE_COLUMNS].values
    y = feature_df["label"].astype(int).values
    groups = feature_df["subject"].values

    # NaN이 있는 특징 행 제거 (짧은 윈도우 등에서 발생 가능)
    valid_mask = ~np.isnan(X).any(axis=1)
    X, y, groups = X[valid_mask], y[valid_mask], groups[valid_mask]

    logo = LeaveOneGroupOut()
    all_true, all_pred, all_prob = [], [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"
            )),
        ])
        pipe.fit(X[train_idx], y[train_idx])

        pred = pipe.predict(X[test_idx])
        prob = pipe.predict_proba(X[test_idx])[:, 1] if len(np.unique(y)) == 2 else None

        all_true.extend(y[test_idx])
        all_pred.extend(pred)
        if prob is not None:
            all_prob.extend(prob)

    print("\n=== Leave-One-Subject-Out 결과 ===")
    print(classification_report(all_true, all_pred, digits=3))
    print("Confusion matrix:\n", confusion_matrix(all_true, all_pred))
    if all_prob:
        try:
            auc = roc_auc_score(all_true, all_prob)
            print(f"ROC-AUC: {auc:.3f}")
        except ValueError:
            pass

    os.makedirs(out_dir, exist_ok=True)
    result_df = pd.DataFrame({"y_true": all_true, "y_pred": all_pred})
    result_df.to_csv(os.path.join(out_dir, "loso_predictions.csv"), index=False)

    # 최종 모델은 전체 데이터로 재학습해서 저장
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"
        )),
    ])
    final_pipe.fit(X, y)

    import joblib
    joblib.dump(final_pipe, os.path.join(out_dir, "cognitive_load_model.joblib"))
    print(f"\n최종 모델 저장 완료: {os.path.join(out_dir, 'cognitive_load_model.joblib')}")


# -----------------------------------------------------------------
# 4. 실행부
# -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="UBFC-rPPG 기반 인지 과부하 분류 파이프라인")
    parser.add_argument("--data_root", required=True, help="UBFC-rPPG subject 폴더들이 있는 루트 경로")
    parser.add_argument("--labels", default=None, help="라벨 CSV 경로 (subject,start_sec,end_sec,label)")
    parser.add_argument("--out_dir", default="./results", help="결과 저장 경로")
    parser.add_argument("--window_sec", type=float, default=10.0, help="특징 추출 윈도우 길이(초)")
    parser.add_argument("--step_sec", type=float, default=5.0, help="윈도우 슬라이딩 간격(초)")
    parser.add_argument("--no_mediapipe", action="store_true", help="mediapipe 미사용(Haar cascade로 대체)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=== 1단계: rPPG 신호 추출 + HRV 특징 계산 ===")
    feature_df = build_feature_table(
        args.data_root,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        use_mediapipe=not args.no_mediapipe,
    )

    if feature_df.empty:
        print("추출된 특징이 없습니다. data_root 경로/폴더 구조를 확인하세요.")
        return

    feat_path = os.path.join(args.out_dir, "hrv_features_all_subjects.csv")
    feature_df.to_csv(feat_path, index=False)
    print(f"\n특징 테이블 저장 완료: {feat_path} (총 {len(feature_df)} 윈도우)")

    if args.labels is None:
        print("\n[안내] --labels 를 지정하지 않아 여기서 종료합니다.")
        print("라벨 CSV(subject,start_sec,end_sec,label)를 준비한 뒤 다시 실행하면 분류기까지 학습됩니다.")
        return

    print("\n=== 2단계: 라벨 매칭 ===")
    labeled_df = attach_labels(feature_df, args.labels)
    print(f"라벨 매칭된 윈도우 수: {len(labeled_df)} / {len(feature_df)}")

    if labeled_df["label"].nunique() < 2:
        print("라벨이 한 클래스뿐입니다. 분류기 학습을 진행할 수 없습니다.")
        return

    print("\n=== 3단계: Leave-One-Subject-Out 학습 및 평가 ===")
    train_and_evaluate(labeled_df, args.out_dir)


if __name__ == "__main__":
    main()