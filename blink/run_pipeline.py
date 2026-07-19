"""
run_pipeline.py
=============================================================
전체 blink 전처리 파이프라인의 메인 실행 파일
"""

import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from ds003838_pipeline import get_subject_list, load_tsv_from_s3
from blink_features import (
    extract_blink_onsets,
    blink_lombscargle_spectrogram,
    blink_entropy,
    blink_rate_per_min,
)
from labeling import label_overload_kmeans_personalized

from ds003838_pipeline import get_subject_list, load_tsv_from_s3, load_beh_from_s3 # 추가

def _load_completed_subjects(summary_path: str = config.SUMMARY_CSV) -> set:
    if not os.path.exists(summary_path):
        return set()
    try:
        done = pd.read_csv(summary_path)
        return set(done["participant_id"].astype(str))
    except Exception:
        return set()

def _append_row_to_csv(row: dict, path: str = config.SUMMARY_CSV):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def run_blink_pipeline(beh_labeled_path: str = config.BEH_LABELED_CSV, limit: int = None):
    os.makedirs(config.OUTPUT_SPEC_DIR, exist_ok=True)
    if not os.path.exists(beh_labeled_path):
        print("💾 행동 데이터 파일이 없습니다. S3에서 다운로드하여 생성합니다...")
        subjects = get_subject_list()
        all_beh = []
        for s in tqdm(subjects, desc="행동 데이터 수집"):
            try:
                df = load_beh_from_s3(s)
                df['participant_id'] = s
                all_beh.append(df)
            except:
                continue
        combined_beh = pd.concat(all_beh)
        combined_beh.to_csv(beh_labeled_path, index=False)
        print(f"✅ '{beh_labeled_path}' 생성 완료! 라벨링을 시작합니다.")

    subjects = get_subject_list()
    subjects = sorted(subjects)
    print(f"✅ 이번에 처리할 대상 피험자 수: {len(subjects)}명", flush=True)

    completed = _load_completed_subjects()
    remaining = [s for s in subjects if s not in completed]
    if completed:
        print(f"⏭️  이미 처리된 {len(completed)}명은 건너뜁니다. 남은 {len(remaining)}명만 처리합니다.", flush=True)

    if limit is not None and len(remaining) > limit:
        print(f"🔢 배치 실행: 이번엔 {limit}명까지만 처리하고 종료합니다.", flush=True)
        remaining = remaining[:limit]

    failed = []

    if remaining:
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = {executor.submit(load_tsv_from_s3, s, "pupil"): s for s in remaining}
            pbar = tqdm(as_completed(futures), total=len(remaining), desc="Blink 처리", unit="명")

            for future in pbar:
                sub_id = futures[future]
                pbar.set_postfix_str(sub_id)

                try:
                    pupil = future.result()
                except Exception as e:
                    failed.append((sub_id, str(e)))
                    tqdm.write(f"  ⚠️ {sub_id} 다운로드 실패: {e}")
                    continue

                if config.BLINK_COL not in pupil.columns:
                    failed.append((sub_id, f"'{config.BLINK_COL}' 컬럼 없음"))
                    tqdm.write(f"  ⚠️ {sub_id} 실패: '{config.BLINK_COL}' 컬럼 없음")
                    continue

                onsets = extract_blink_onsets(pupil)
                t_start = float(pupil[config.TIME_COL_PUPIL].min())
                t_end = float(pupil[config.TIME_COL_PUPIL].max())
                del pupil

                spec = blink_lombscargle_spectrogram(onsets, t_start, t_end)
                be = blink_entropy(spec)
                br = blink_rate_per_min(onsets, t_start, t_end)

                spec_path = os.path.join(config.OUTPUT_SPEC_DIR, f"{sub_id}_spectrogram.npy")
                np.save(spec_path, spec)
                n_windows = int(spec.shape[0])
                del spec

                _append_row_to_csv({
                    "participant_id": sub_id,
                    "n_windows": n_windows,
                    "blink_rate_per_min": br,
                    "blink_entropy": be,
                    "spectrogram_path": spec_path,
                })

    if failed:
        print(f"\n⚠️ 이번 실행에서 실패한 피험자 {len(failed)}명:", flush=True)
        for sub_id, err in failed:
            print(f"   {sub_id}: {err}", flush=True)

    if not os.path.exists(config.SUMMARY_CSV):
        print("❌ 처리된 데이터가 없습니다.", flush=True)
        return None

    blink_summary = pd.read_csv(config.SUMMARY_CSV)

    if os.path.exists(beh_labeled_path):
        beh = pd.read_csv(beh_labeled_path)
        if "accuracy" not in beh.columns:
            # 조건별 만점 기준 딕셔너리
            max_scores = {5: 5.0, 9: 9.0, 13: 13.0}

            # 각 행의 condition에 맞는 만점 기준으로 나누기
            beh['accuracy'] = beh.apply(
                lambda row: row['NCorrect'] / max_scores.get(row['condition'], 13.0),
                axis=1
            )
        if "accuracy" in beh.columns and config.CONDITION_COL in beh.columns:
            task_trials = beh[~beh[config.CONDITION_COL].isin(config.BASELINE_CONDITIONS)].copy()
            task_trials = label_overload_kmeans_personalized(
                task_trials, value_col="accuracy",
                group_col="participant_id", label_name="overload_kmeans_objective",
            )
            task_trials.to_csv(config.OVERLOAD_LABELS_CSV, index=False)
            n_labeled = task_trials["overload_kmeans_objective"].notna().sum()
            print(f"💾 '{config.OVERLOAD_LABELS_CSV}' 저장됨 (라벨링된 시행 수: {n_labeled}건)", flush=True)

    print(f"\n🎉 처리 완료! '{config.SUMMARY_CSV}' 저장됨.", flush=True)
    return blink_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    blink_summary = run_blink_pipeline(limit=args.limit)
