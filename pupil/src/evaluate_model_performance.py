"""
모델 성능 종합 평가 스크립트 (참가자 단위 교차검증 기준)
============================================================
pupil_baseline_pipeline.py 실행 후 나온 결과 파일(outputs/pupil_load_score_labeled.csv)을
읽어서, 회귀/이진분류 성능 지표를 모두 한 번에 계산합니다.

★ 중요: 참가자를 5그룹으로 나누어 각 그룹은 학습에서 완전히
  제외한 뒤 예측하는 방식(GroupKFold)으로 "처음 보는 사람"에 대한 성능만
  집계합니다. pupil_baseline_pipeline.py의 검증 방식과 동일합니다.

계산하는 지표
------------
[회귀 성능] - "부하 정도(연속값)"를 얼마나 잘 예측했는가
  - R^2 (교차검증 기준)
  - 상관계수(Pearson correlation) (교차검증 기준)

[이진분류 성능] - 40.5% 경계값 기준 "정상/과부하"를 얼마나 잘 맞혔는가
  - Accuracy (교차검증 기준)
  - Precision / Recall / F1-score (정상, 과부하 각각)
  - Confusion Matrix
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, percentileofscore
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score, accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

LABELED_PATH = "pupil/outputs/pupil_load_score_labeled.csv"
LOAD_THRESHOLD_PCT = 40.5
N_FOLDS = 5
FEATURE_COLS = ["peak_val", "peak_pos", "early_mean", "late_val", "decline", "plateau_len", "volatility"]
RF_PARAMS = dict(n_estimators=400, max_depth=8, min_samples_leaf=3, max_features=None, random_state=42)


def get_out_of_fold_predictions(df: pd.DataFrame) -> pd.Series:
    """
    참가자 단위 5-fold 교차검증으로, 각 시행에 대해 "그 시행의 참가자가
    학습에 전혀 쓰이지 않은 모델"의 예측값만 모아서 반환한다.
    (= 실제 배포 시 "처음 보는 사람"에게 나올 성능을 정직하게 반영)
    """
    X = df[FEATURE_COLS].values
    y = df["accuracy_z"].values
    groups = df["subject"].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_pred = np.zeros(len(df))

    for train_idx, test_idx in gkf.split(X, y, groups):
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X[train_idx], y[train_idx])
        oof_pred[test_idx] = model.predict(X[test_idx])

    return pd.Series(oof_pred, index=df.index)


def evaluate_regression(df: pd.DataFrame, oof_pred: pd.Series):
    """회귀 성능: 실제 부하 정도(accuracy_z) vs 교차검증 예측(out-of-fold)"""
    y_true = df["accuracy_z"]
    y_pred = oof_pred

    r2 = r2_score(y_true, y_pred)
    corr, p = pearsonr(y_true, y_pred)

    print("=" * 55)
    print("[회귀 성능 - 참가자 단위 교차검증] 부하 정도(연속값) 예측")
    print("=" * 55)
    print(f"  R^2       = {r2:.4f}")
    print(f"  상관계수   = {corr:.4f} (p={p:.2e})")
    print()


def evaluate_classification(df: pd.DataFrame, oof_pred: pd.Series):
    """
    이진분류 성능: 실제 라벨(정답률 기준 40.5% 경계) vs 교차검증 예측 라벨

    실제 라벨과 예측 라벨 모두, 각자의 분포에서 백분위 변환 후
    동일한 경계값(40.5%)을 적용해 산출한다. (모델 학습에는 정답이 전혀
    쓰이지 않았으므로, 이 라벨 비교는 순수하게 "처음 보는 사람" 성능이다)
    """
    df = df.copy()

    # 실제 정답 기준 라벨
    z_true_all = df["accuracy_z"]
    df["true_pct"] = df["accuracy_z"].apply(lambda v: 100 - percentileofscore(z_true_all, v))
    df["true_label"] = (df["true_pct"] >= LOAD_THRESHOLD_PCT).astype(int)

    # 교차검증 예측(out-of-fold) 기준 라벨
    df["pred_z_oof"] = oof_pred
    z_pred_all = df["pred_z_oof"]
    df["pred_pct"] = df["pred_z_oof"].apply(lambda v: 100 - percentileofscore(z_pred_all, v))
    df["pred_label"] = (df["pred_pct"] >= LOAD_THRESHOLD_PCT).astype(int)

    y_true = df["true_label"]
    y_pred = df["pred_label"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("=" * 55)
    print(f"[이진분류 성능 - 참가자 단위 교차검증] 정상/과부하 (경계값 {LOAD_THRESHOLD_PCT}%)")
    print("=" * 55)
    print(f"  Accuracy  = {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Precision(과부하) = {prec:.4f}")
    print(f"  Recall(과부하)    = {rec:.4f}")
    print(f"  F1-score(과부하)  = {f1:.4f}")
    print()
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"                 예측:정상   예측:과부하")
    print(f"  실제:정상        {cm[0][0]:>6}      {cm[0][1]:>6}")
    print(f"  실제:과부하       {cm[1][0]:>6}      {cm[1][1]:>6}")
    print()
    print("상세 리포트:")
    print(classification_report(y_true, y_pred, target_names=["정상", "과부하"]))


if __name__ == "__main__":
    df = pd.read_csv(LABELED_PATH)
    print(f"데이터 로딩 완료: {len(df)}개 시행")
    print("참가자 단위 교차검증 예측 계산 중... (몇 분 소요될 수 있습니다)\n")

    oof_pred = get_out_of_fold_predictions(df)

    evaluate_regression(df, oof_pred)
    evaluate_classification(df, oof_pred)