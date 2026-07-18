"""
cognitive_load_score.py
------------------------
라벨(정상/과부하 이진 라벨) 없이도 계산 가능한 "인지 부하 연속 점수(0~100)"를 산출.

기존 cognitive_load_pipeline.py 의 1단계(특징 추출) 결과인
hrv_features_all_subjects.csv 를 입력으로 받아서 동작.

핵심 아이디어
-------------
인지 과부하(교감신경 활성화) 시 HRV 특징들이 특정 방향으로 움직인다는
생리학적 근거가 있음 (라벨 없이도 "방향"은 알 수 있음):

    ↑ 증가하면 과부하 방향 : mean_hr, lf_hf_ratio
    ↓ 감소하면 과부하 방향 : sdnn, rmssd, pnn50, hf_power

이 방향 정보를 이용해서:
    1. UBFC-rPPG 전체 윈도우를 population 삼아 각 특징을 z-score 표준화
    2. 방향에 맞게 부호를 준 가중합(composite z-score) 계산
    3. sigmoid로 0~100 점수로 매핑 (해석 용이 + 다른 모달리티와 fusion 시 스케일 통일)

나중에 팀원과 같은 영상으로 실전 적용할 때, 여기서 학습 시(UBFC-rPPG) 계산한
mean/std/weight를 "reference_stats.json"으로 저장해뒀다가 그대로 재사용해야
같은 기준으로 점수가 나옴 (다시 fit 하면 점수 스케일이 흔들림).

사용 예:
    # 1) UBFC-rPPG로 reference 통계 산출 + 점수 계산
    python cognitive_load_score.py \
        --features ./results/hrv_features_all_subjects.csv \
        --out_dir ./results \
        --mode fit

    # 2) 나중에 팀 공용 영상 특징에 "같은 기준"으로 점수만 매기고 싶을 때
    python cognitive_load_score.py \
        --features ./results/team_video_features.csv \
        --out_dir ./results \
        --mode apply \
        --reference ./results/reference_stats.json
"""

import argparse
import json
import numpy as np
import pandas as pd


# 특징별 방향(+1 = 높을수록 과부하, -1 = 낮을수록 과부하)과
# 상대적 중요도(가중치)를 명시. 가중치는 HRV 문헌에서 스트레스 반응성이
# 큰 순서(RMSSD/HF power가 급성 스트레스에 가장 민감)를 참고해 정함.
FEATURE_DIRECTIONS = {
    "mean_hr":      (+1, 1.0),
    "sdnn":         (-1, 1.0),
    "rmssd":        (-1, 1.5),   # 부교감 활성도 지표 - 급성 스트레스에 민감
    "pnn50":        (-1, 1.0),
    "lf_hf_ratio":  (+1, 1.5),   # 교감/부교감 균형 - 스트레스 시 가장 직접적으로 반응
    "hf_power":     (-1, 1.0),
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# -----------------------------------------------------------------
# 1. Fit: population 통계 계산 + composite score 산출
# -----------------------------------------------------------------
def fit_reference_stats(feature_df: pd.DataFrame) -> dict:
    stats = {}
    for feat, (direction, weight) in FEATURE_DIRECTIONS.items():
        col = feature_df[feat].dropna()
        stats[feat] = {
            "mean": float(col.mean()),
            "std": float(col.std() if col.std() > 1e-8 else 1e-8),
            "direction": direction,
            "weight": weight,
        }
    return stats


def compute_composite_zscore(feature_df: pd.DataFrame, stats: dict) -> np.ndarray:
    """각 윈도우의 (부호 있는) 가중 z-score 합을 계산."""
    total_weight = sum(s["weight"] for s in stats.values())
    composite = np.zeros(len(feature_df))
    valid_mask = np.ones(len(feature_df), dtype=bool)

    for feat, s in stats.items():
        col = feature_df[feat].values
        z = (col - s["mean"]) / s["std"]
        composite += s["direction"] * s["weight"] * z
        valid_mask &= ~np.isnan(col)

    composite = composite / total_weight
    composite[~valid_mask] = np.nan
    return composite


def zscore_to_100_scale(composite_z: np.ndarray, scale: float = 1.5) -> np.ndarray:
    """
    composite z-score를 0~100 점수로 매핑.
    scale이 클수록 중간 구간(50점 근처)에서 점수 변화가 완만해짐.
    """
    return 100.0 * sigmoid(composite_z / scale)


# -----------------------------------------------------------------
# 2. 실행부
# -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HRV 기반 인지 부하 연속 점수 산출")
    parser.add_argument("--features", required=True, help="hrv_features_*.csv 경로")
    parser.add_argument("--out_dir", default="./results")
    parser.add_argument("--mode", choices=["fit", "apply"], default="fit",
                         help="fit: 이 데이터로 새 reference 통계 산출 / apply: 기존 reference 재사용")
    parser.add_argument("--reference", default=None,
                         help="--mode apply 일 때 사용할 reference_stats.json 경로")
    args = parser.parse_args()

    feature_df = pd.read_csv(args.features)

    missing = [f for f in FEATURE_DIRECTIONS if f not in feature_df.columns]
    if missing:
        raise ValueError(f"특징 테이블에 다음 컬럼이 없음: {missing}")

    if args.mode == "fit":
        stats = fit_reference_stats(feature_df)
        ref_path = f"{args.out_dir}/reference_stats.json"
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"reference 통계 저장 완료: {ref_path}")
        print("(나중에 팀 공용 영상에 같은 기준으로 점수 매길 때 --mode apply --reference 로 재사용)")
    else:
        if args.reference is None:
            raise ValueError("--mode apply 사용 시 --reference 경로가 필요함")
        with open(args.reference, "r", encoding="utf-8") as f:
            stats = json.load(f)
        print(f"reference 통계 로드: {args.reference}")

    composite_z = compute_composite_zscore(feature_df, stats)
    score = zscore_to_100_scale(composite_z)

    feature_df["cognitive_load_composite_z"] = composite_z
    feature_df["cognitive_load_score"] = score

    out_path = f"{args.out_dir}/cognitive_load_scores.csv"
    feature_df.to_csv(out_path, index=False)
    print(f"윈도우별 점수 저장 완료: {out_path}")

    valid = feature_df["cognitive_load_score"].dropna()
    print(f"\n=== 점수 분포 요약 (전체 {len(valid)} 윈도우) ===")
    print(valid.describe())

    if "subject" in feature_df.columns:
        print("\n=== subject별 평균 점수 ===")
        print(feature_df.groupby("subject")["cognitive_load_score"].mean().sort_values(ascending=False))


if __name__ == "__main__":
    main()