"""
clean_blink_dataset.py

data/branch/trial_dataset_blink.csv (팀원 원본 blink+PSD 파일)를 브랜치1
학습에 바로 쓸 수 있게 정리함.

(참고) 지금 배포하는 dataset/gaze_pupil_blink_merged.csv에는 이미 재계산된
blink 값(n_blinks_ours 등)이 포함되어 있어서, 이 스크립트를 따로 돌릴 필요는
없음. 팀원이 준 원본 PSD 파일을 처음부터 직접 정제해야 하는 경우에만 참고.

정리 내용
------------------------------------------------
1. n_blinks > N_BLINKS_MAX(100)인 행 -- 담당자 본인이 "아이트래커 오류로
   보이니 결측치 처리해도 된다"고 확인해준 부분. 이 행들의 blink 관련 값
   전부(원본 + z-score 버전 둘 다)를 NaN으로 바꿈. n_blinks 하나만 이상한
   게 아니라 그 trial의 blink 계산 자체가 잘못됐을 가능성이 높아서, 파생된
   값(blink_rate_per_min, mean_ibi, std_ibi, entropy, PSD 93개) 전부 같이
   버림.
2. blink_psd_* 결측(전체의 72%, 담당자 확인 결과 "PSD 계산에 필요한
   61초 윈도우를 trial 길이가 못 채워서" 생기는 구조적 한계)은 그대로 둠 --
   억지로 채우지 않고 LightGBM의 기본 NaN 처리에 맡김.
3. 컬럼명을 pupil/gaze 쪽 병합 키(subject_id, trial_index)에 맞춤:
   participant_id -> subject_id, trial(1-indexed) -> trial_index(0-indexed).
4. accuracy/difficulty/overload_score 등 beh.tsv 파생 컬럼은 pupil 쪽에
   이미 있는 정보(label, correct 등)라서 여기서는 빼고 순수 blink feature만
   남김 -- 안 그러면 pupil 병합할 때 중복/충돌나고, condition을 우회해서
   다시 라벨을 흘려보내는 것과 비슷한 위험이 생길 수 있음.

사용법:
    python src/clean_blink_dataset.py \
        --in data/branch/trial_dataset_blink.csv \
        --out output/branch1/blink_features_cleaned.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_BLINKS_MAX = 100  # 이 값을 넘으면 계산 오류로 보고 결측 처리 (담당자에게 의견 구함)

# 진짜 blink "신호" feature만 골라냄 (beh.tsv 파생 정답/난이도 컬럼은 제외 --
# 그건 pupil 쪽에 이미 label/correct로 들어있어서 중복이고, 잘못 쓰면 누수 위험)
BLINK_RAW_COLS = ["n_blinks", "blink_rate_per_min", "mean_ibi", "std_ibi", "blink_entropy_trial"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    blink_psd_cols = [c for c in df.columns if c.startswith("blink_psd_") and not c.endswith("_z")]
    blink_z_cols = [c for c in df.columns if c.endswith("_z") and (
        c.replace("_z", "") in BLINK_RAW_COLS or c.startswith("blink_psd_"))]
    all_blink_cols = BLINK_RAW_COLS + blink_psd_cols + blink_z_cols

    bad = df["n_blinks"] > N_BLINKS_MAX
    print(f"n_blinks > {N_BLINKS_MAX} 인 행: {bad.sum()}개 ({bad.mean()*100:.1f}%) -- blink 관련 값 전부 결측 처리")
    df.loc[bad, all_blink_cols] = np.nan

    keep_cols = ["participant_id", "trial"] + all_blink_cols
    out = df[keep_cols].copy()
    out = out.rename(columns={"participant_id": "subject_id"})
    out["trial_index"] = out["trial"] - 1  # 1-indexed -> 0-indexed (기존에 확인해둔 오프셋)
    out = out.drop(columns=["trial"])

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", default="data/branch/trial_dataset_blink.csv")
    parser.add_argument("--out", default="output/branch1/blink_features_cleaned.csv")
    args = parser.parse_args()

    print(f"로드: {args.in_path}")
    df = pd.read_csv(args.in_path, low_memory=False)
    print(f"원본: {len(df)}행, {df['participant_id'].nunique()}명")

    out = clean(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"저장: {out_path} ({len(out)}행, {len(out.columns)}컬럼)")


if __name__ == "__main__":
    main()
