"""
clean_blink_newfile.py

blink 담당자가 새로 보내준 blink_features_ours.csv(재계산 버전)를 정제함.

문제
------------------------------------------------
blink_feature.py의 trial 구간 계산 로직이:
    end = trials[trial_id + 1].iloc[0]["timestamp"]   # 다음 trial 시작 전까지
을 그대로 쓰고 있어서, 블록 마지막 trial(trial_index 17/35/53/71/89/107/125/
143/161 근처)에서 다음 trial 시작까지의 텀(블록 간 텀으로 추정)까지 trial
길이에 통째로 포함됨. 그 결과 trial_duration_sec이 최대 2189초(36분)까지,
n_blinks_ours가 최대 528회까지 나오는 등 물리적으로 불가능한 값이 470개행
(전체의 3.5%)에서 발생함 (13자리 조건이라도 실제로는 26초 안팎이 정상 범위).

우리도 src/blink/build_blink_features.py에서 동일한 버그를 겪었고
MAX_TRIAL_DURATION_SEC=60.0 상한을 둬서 해결했었음. 이 파일은 raw 신호가
아니라 이미 집계된 CSV라 그 방식대로 재계산은 못 하고, 대신 같은 기준
(60초 초과 = 신뢰 불가)으로 해당 행의 blink 관련 값을 결측 처리함.

검증 결과 (참고)
------------------------------------------------
60초 초과 470개 행을 결측 처리해도 Branch1 test accuracy는 81.0% -> 80.4%로
거의 변하지 않음(-0.6%p, 노이즈 수준) -- 즉 이번에 좋아진 성능은 이 버그 덕이
아니라 새로 추가된 blink_duration_max/blink_ratio 같은 지표가 실제로
유의미하기 때문으로 보임. 그래도 물리적으로 불가능한 값을 모델에 넣을
이유는 없으므로 정제 후 버전을 기본으로 사용함.

산출물
------------------------------------------------
- --out: 정제된 버전 (60초 초과 행의 blink 값 결측 처리, 기본 사용 권장)
- --out-raw: 원본 그대로 복사본 (참고/백업용, 담당자에게 같이 전달)

사용법:
    python src/branch1/clean_blink_newfile.py \
        --in data/branch/blink_features_ours.csv \
        --out output/blink/blink_features_ours_cleaned.csv \
        --out-raw output/blink/blink_features_ours_raw.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MAX_TRIAL_DURATION_SEC = 60.0  # 이보다 길면 블록 간 텀이 섞인 것으로 보고 결측 처리


def clean(df: pd.DataFrame) -> pd.DataFrame:
    blink_feature_cols = [c for c in df.columns if c not in ("subject_id", "trial_index")]
    bad = df["trial_duration_sec"] > MAX_TRIAL_DURATION_SEC
    print(f"trial_duration_sec > {MAX_TRIAL_DURATION_SEC:.0f}초 인 행: "
          f"{bad.sum()}개 ({bad.mean()*100:.1f}%) -- blink 관련 값 전부 결측 처리")

    out = df.copy()
    out.loc[bad, blink_feature_cols] = np.nan
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", default="output/blink/blink_features_ours_cleaned.csv")
    parser.add_argument("--out-raw", default=None,
                         help="지정하면 원본을 그대로 복사해서 같이 저장 (참고/백업용)")
    args = parser.parse_args()

    print(f"로드: {args.in_path}")
    df = pd.read_csv(args.in_path)
    print(f"원본: {len(df)}행, {df['subject_id'].nunique()}명")

    out = clean(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"정제본 저장: {out_path} ({len(out)}행)")

    if args.out_raw:
        raw_path = Path(args.out_raw)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
        print(f"원본 복사본 저장: {raw_path} ({len(df)}행)")


if __name__ == "__main__":
    main()
