"""
GazeGuard - 동공정보 기반 인지 부하 정도 예측 파이프라인 (Baseline)
============================================================
전체 흐름 
----------------
1. 동공 데이터 로딩 + remember(실제 암기)/control(단순 청취) 구간 분리
2. 시행 단위 동공 반응 feature 7개 생성
   (peak_val, peak_pos, early_mean, late_val, decline, plateau_len, volatility)
3. 정답률 기반 "개인별 상대 부하 정도" 라벨 생성
   - 각 참가자의 쉬운 조건(5,9) 정답률로 개인 baseline(평균/표준편차) 계산
   - 전체 시행의 정답률을 이 baseline 기준으로 Z-score화
   - Z-score가 낮을수록(=평소보다 못할수록) 부하가 큰 것으로 해석
   ※ condition(난이도)을 그대로 라벨로 쓰지 않는 이유:
     동일 난이도라도 개인마다 실제로 느끼는 부하가 다르기 때문
4. 동공 feature 7개로 이 부하 정도(Z-score)를 예측하는 회귀 모델 학습
   (Random Forest, 참가자 단위 교차검증으로 튜닝된 하이퍼파라미터 사용)
5. 예측값을 백분위 기반으로 0~100% "부하 정도 점수"로 변환
6. 데이터 기반으로 도출된 경계값(40.5%)을 적용해 정상/과부하 이진 라벨도 함께 산출

입력 파일
--------
- data/combined_pupil_positions.csv : 참가자별 동공 반응 원시 처리 결과
- data/beh_all.csv                  : 참가자별 시행별 정답률
"""

import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# ----- 경로 설정 -----
PUPIL_PATH = "pupil/data/combined_pupil_positions.csv"
BEH_PATH = "pupil/data/beh_all.csv"
OUTPUT_PATH = "pupil/outputs/pupil_load_score_labeled.csv"

# ----- 확정된 설정값 (근거: 판단방식_결정과정_보고서.md 참고) -----
N_FOLDS = 5
RF_PARAMS = dict(n_estimators=400, max_depth=8, min_samples_leaf=3, max_features=None, random_state=42)
LOAD_THRESHOLD_PCT = 40.5  # K-means로 도출된 정상/과부하 경계값 (부하 정도 % 기준)
FEATURE_COLS = ["peak_val", "peak_pos", "early_mean", "late_val", "decline", "plateau_len", "volatility"]


# =========================================================
# STEP 1. remember/control 시행 구분
# =========================================================
def split_remember_control(df: pd.DataFrame) -> pd.DataFrame:
    """
    참가자별 trial_seg 등장 순서를 global_trial_num(1~162)으로 재정렬하고,
    18개 단위 블록에서 4~15번째(가운데 12개)를 remember, 나머지를 control로 분류.
    (이 규칙은 beh.tsv의 trial 번호 패턴 분석으로 검증됨)
    """
    parts = []
    for sub, g in df.groupby("subject"):
        g = g.copy()
        order_map = {seg: i + 1 for i, seg in enumerate(sorted(g["trial_seg"].unique()))}
        g["global_trial_num"] = g["trial_seg"].map(order_map)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out = out[out["global_trial_num"] <= 162].copy()  # 이상 segment 제외

    out["block_position"] = ((out["global_trial_num"] - 1) % 18) + 1
    out["task_type"] = np.where(out["block_position"].between(4, 15), "remember", "control")
    return out


# =========================================================
# STEP 2. 시행 단위 동공 feature 생성
# =========================================================
def summarize_trial(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("position")
    rel = g["rel"].values
    diffs = np.diff(rel)

    peak_val = rel.max()
    peak_pos = g["position"].iloc[np.argmax(rel)]
    early_mean = g[g["position"] <= 5]["rel"].mean()
    late_val = rel[-1]
    decline = peak_val - late_val
    plateau_len = np.sum(np.abs(diffs) < 0.05) if len(diffs) > 0 else 0
    volatility = np.std(diffs) if len(diffs) > 1 else 0

    return pd.Series({
        "peak_val": peak_val, "peak_pos": peak_pos, "early_mean": early_mean,
        "late_val": late_val, "decline": decline,
        "plateau_len": plateau_len, "volatility": volatility,
        "condition": g["condition"].iloc[0],
        "global_trial_num": g["global_trial_num"].iloc[0],
    })


def build_pupil_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = split_remember_control(raw)
    remember = df[df["task_type"] == "remember"].copy()
    remember["rel"] = remember["diameter_3d"] - remember["baseline_pretrial"]
    remember = remember.dropna(subset=["rel"])

    summary = (
        remember.groupby(["subject", "trial_seg"])
        .apply(summarize_trial, include_groups=False)
        .reset_index()
        .dropna()
    )
    return summary


# =========================================================
# STEP 3. 개인별 상대 부하 정도(Z-score) 라벨 생성
# =========================================================
def build_load_labels(beh: pd.DataFrame) -> pd.DataFrame:
    """
    condition 5, 9(상대적으로 쉬운 조건)의 정답률로 참가자 개인의 baseline(평균/표준편차)을 구하고,
    전체 시행의 정답률을 이 baseline 기준 Z-score로 변환.
    """
    baseline = beh[beh["condition"].isin([5, 9])].groupby("subject")["accuracy"].agg(["mean", "std"])
    beh = beh.merge(baseline, on="subject", how="left", suffixes=("", "_baseline"))
    beh["accuracy_z"] = np.where(
        beh["std"].fillna(0) > 0,
        (beh["accuracy"] - beh["mean"]) / beh["std"],
        np.nan,
    )
    return beh.dropna(subset=["accuracy_z"])


# =========================================================
# STEP 4. 회귀 모델 학습 (참가자 단위 교차검증)
# =========================================================
def train_and_evaluate(merged: pd.DataFrame):
    X = merged[FEATURE_COLS].values
    y = merged["accuracy_z"].values
    groups = merged["subject"].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    all_pred, all_true = [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        reg = RandomForestRegressor(**RF_PARAMS)
        reg.fit(X[train_idx], y[train_idx])
        pred = reg.predict(X[test_idx])
        all_pred.extend(pred)
        all_true.extend(y[test_idx])

    r2 = r2_score(all_true, all_pred)
    corr, p = pearsonr(all_true, all_pred)
    print(f"[모델 성능 - 참가자 단위 {N_FOLDS}-fold 교차검증]")
    print(f"  R^2 = {r2:.4f}")
    print(f"  상관계수 = {corr:.4f} (p={p:.2e})")

    # 전체 데이터로 최종 모델 학습 (실제 예측/배포용)
    final_model = RandomForestRegressor(**RF_PARAMS)
    final_model.fit(X, y)

    print("\n[Feature 중요도]")
    for feat, imp in sorted(zip(FEATURE_COLS, final_model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

    return final_model


# =========================================================
# STEP 5. 부하 정도(%) 변환 + 이진 라벨 산출
# =========================================================
def add_load_score(merged: pd.DataFrame, model: RandomForestRegressor) -> pd.DataFrame:
    X = merged[FEATURE_COLS].values
    merged = merged.copy()
    merged["predicted_z"] = model.predict(X)

    # 백분위 기반 변환: 극단치에 흔들리지 않도록, 실제 값 순위 기준으로 0~100% 산정
    z_all = merged["predicted_z"]
    merged["load_score_pct"] = merged["predicted_z"].apply(
        lambda v: 100 - percentileofscore(z_all, v)
    )

    # 데이터 기반으로 도출된 경계값 적용 (근거: 판단방식_결정과정_보고서.md 6장)
    merged["load_label"] = np.where(merged["load_score_pct"] >= LOAD_THRESHOLD_PCT, "과부하", "정상")

    return merged


# =========================================================
# 메인 실행
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1-2. 동공 데이터 로딩 및 feature 생성")
    print("=" * 60)
    raw = pd.read_csv(PUPIL_PATH)
    pupil_features = build_pupil_features(raw)
    print(f"시행 단위 feature {len(pupil_features)}개 생성 완료 (참가자 {pupil_features['subject'].nunique()}명)")

    print("\n" + "=" * 60)
    print("STEP 3. 개인별 상대 부하 정도(라벨) 생성")
    print("=" * 60)
    beh = pd.read_csv(BEH_PATH)
    beh_labeled = build_load_labels(beh)
    print(f"라벨 생성 완료: {len(beh_labeled)}개 시행")

    merged = pupil_features.merge(
        beh_labeled[["subject", "trial", "accuracy_z"]],
        left_on=["subject", "global_trial_num"], right_on=["subject", "trial"],
        how="inner",
    )
    print(f"동공 feature - 라벨 매칭 완료: {len(merged)}개 시행")

    print("\n" + "=" * 60)
    print("STEP 4. 회귀 모델 학습 및 평가")
    print("=" * 60)
    model = train_and_evaluate(merged)

    print("\n" + "=" * 60)
    print("STEP 5. 부하 정도(%) 변환 및 이진 라벨 산출")
    print("=" * 60)
    result = add_load_score(merged, model)
    print(result.groupby("condition")["load_score_pct"].mean().round(1))
    print()
    print("정상/과부하 분류 비율:")
    print(result["load_label"].value_counts(normalize=True).round(3))

    # import os
    # os.makedirs("./outputs", exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\n최종 결과 저장 완료: {OUTPUT_PATH}")