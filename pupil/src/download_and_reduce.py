# ============================================================
# STEP 1: 84명 원본 동공 데이터 다운로드 + 전처리
# 로컬(VS Code 등) / Colab 어디서든 그대로 실행 가능
# ============================================================

import subprocess
import os
import gc
import numpy as np
import pandas as pd

subprocess.run(["pip", "install", "awscli", "-q"], capture_output=True)

# ----- 설정: S3에서 실제 존재하는 참가자 목록을 직접 조회 -----
EXCLUDE = {"017", "094"}  # README 기준 동공 데이터 없는 참가자 (자동 제외)


def get_actual_subject_list():
    """1~N으로 추측하지 않고, S3 버킷에 실제로 존재하는 sub- 폴더 목록을 가져온다."""
    result = subprocess.run(
        ["aws", "s3", "ls", "--no-sign-request", "s3://openneuro.org/ds003838/"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"참가자 목록 조회 실패: {result.stderr}")

    subs = []
    for line in result.stdout.splitlines():
        # 출력 예시: "                           PRE sub-018/"
        line = line.strip()
        if line.startswith("PRE") and "sub-" in line:
            folder_name = line.split("PRE")[-1].strip().rstrip("/")
            sub_id = folder_name.replace("sub-", "")
            subs.append(sub_id)
    return sorted(subs)


SUBJECTS = [s for s in get_actual_subject_list() if s not in EXCLUDE]
print(f"S3에서 확인된 실제 참가자 수: {len(SUBJECTS)}")
print("참가자 목록:", SUBJECTS)

DATA_DIR = "./data/pupil_raw"
os.makedirs(DATA_DIR, exist_ok=True)

RESULT_PATH = "pupil/data/combined_pupil_positions.csv"
DONE_LOG_PATH = "pupil/data/done_subjects.txt"

S3_BASE = "s3://openneuro.org/ds003838"

# ----- 이어서 하기(resume) 위한 처리 완료 목록 불러오기 -----
if os.path.exists(DONE_LOG_PATH):
    with open(DONE_LOG_PATH) as f:
        done_subjects = set(line.strip() for line in f if line.strip())
    print(f"이미 처리된 참가자 {len(done_subjects)}명은 건너뜁니다.")
else:
    done_subjects = set()


def download_subject(sub_id: str) -> bool:
    """한 참가자의 beh, events, pupil 파일을 받는다. 성공 여부 반환."""
    files = {
        "beh": f"sub-{sub_id}/beh/sub-{sub_id}_task-memory_beh.tsv",
        "events": f"sub-{sub_id}/pupil/sub-{sub_id}_task-memory_events.tsv",
        "pupil": f"sub-{sub_id}/pupil/sub-{sub_id}_task-memory_pupil.tsv",
    }
    local_paths = {}
    for key, rel_path in files.items():
        local_path = os.path.join(DATA_DIR, os.path.basename(rel_path))
        result = subprocess.run(
            ["aws", "s3", "cp", "--no-sign-request", f"{S3_BASE}/{rel_path}", local_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ⚠️ sub-{sub_id} {key} 다운로드 실패: {result.stderr[:200]}")
            return False
        local_paths[key] = local_path
    return local_paths


def reduce_pupil(pupil_path: str) -> pd.DataFrame:
    """
    pupil.tsv를 전처리하여 반환.
    1) 참가자별로 confidence가 더 높은 눈(eye_id) 하나만 선택
    2) 3d 방식, 생리학적 범위(2~8mm)만 남김
    3) 급격히 튀는 이상치를 제거 후 선형보간
    """
    needed = ["pupil_timestamp", "eye_id", "confidence", "diameter_3d", "method"]

    # ----- 1차 통과: 눈(eye_id)별 평균 confidence 계산 (더 나은 눈 선택용) -----
    conf_sum = {0.0: 0.0, 1.0: 0.0}
    conf_count = {0.0: 0, 1.0: 0}

    reader = pd.read_csv(
        pupil_path, sep="\t",
        usecols=lambda c: c in ["eye_id", "confidence", "method"],
        chunksize=300_000, low_memory=True,
    )
    for chunk in reader:
        chunk = chunk[chunk["method"] == "3d c++"]
        for eye in [0.0, 1.0]:
            sub = chunk[chunk["eye_id"] == eye]
            conf_sum[eye] += sub["confidence"].sum()
            conf_count[eye] += len(sub)

    mean_conf = {
        eye: (conf_sum[eye] / conf_count[eye] if conf_count[eye] > 0 else -1)
        for eye in [0.0, 1.0]
    }
    best_eye = max(mean_conf, key=mean_conf.get)

    # ----- 2차 통과: 선택된 눈만 필터링해서 실제 데이터 수집 -----
    chunks = []
    reader = pd.read_csv(
        pupil_path, sep="\t",
        usecols=lambda c: c in needed,
        chunksize=300_000, low_memory=True,
    )
    for chunk in reader:
        chunk = chunk[(chunk["method"] == "3d c++") & (chunk["eye_id"] == best_eye)]
        chunk = chunk[
            (chunk["diameter_3d"] >= 2) & (chunk["diameter_3d"] <= 8) & (chunk["confidence"] > 0.6)
        ]
        if len(chunk) > 0:
            chunks.append(chunk.copy())

    if not chunks:
        return pd.DataFrame(columns=needed)

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    df = df.sort_values("pupil_timestamp").reset_index(drop=True)

    # ----- 이상치 탐지: 직전 값 대비 급격히(1.5mm 초과) 튀는 값은 결측 처리 -----
    diffs = df["diameter_3d"].diff().abs()
    is_outlier = diffs > 1.5
    df.loc[is_outlier, "diameter_3d"] = np.nan

    # ----- 결측 구간 선형보간 (너무 긴 공백은 보간하지 않고 NaN 유지) -----
    df["diameter_3d"] = df["diameter_3d"].interpolate(method="linear", limit=25, limit_direction="both")
    df = df.dropna(subset=["diameter_3d"])

    df["eye_id_used"] = best_eye
    return df


def build_trial_events(events_path: str) -> pd.DataFrame:
    """이벤트 timestamp 간격으로 trial 경계와 순번(position)을 재구성."""
    ev = pd.read_csv(events_path, sep="\t")
    ev = ev.sort_values("timestamp").reset_index(drop=True)
    gaps = ev["timestamp"].diff()
    new_trial = (gaps > 3.0) | (gaps.isna())
    ev["trial_seg"] = new_trial.cumsum()
    ev["position"] = ev.groupby("trial_seg").cumcount() + 1
    ev["seg_len"] = ev.groupby("trial_seg")["trial_seg"].transform("count")
    return ev


# ----- 전체 참가자 처리 (참가자마다 즉시 저장하여 중간에 끊겨도 안전) -----
for sub_id in SUBJECTS:
    if sub_id in done_subjects:
        continue  # 이미 처리됨 -> 건너뜀

    print(f"\n=== sub-{sub_id} 처리 중 ({len(done_subjects)+1}/{len(SUBJECTS)}) ===")
    paths = download_subject(sub_id)
    if not paths:
        # 다운로드 실패(원래 없는 참가자 번호 등)도 완료 처리하여 다음에 재시도 안 하도록 기록
        with open(DONE_LOG_PATH, "a") as f:
            f.write(sub_id + "\n")
        continue

    try:
        pupil_df = reduce_pupil(paths["pupil"])
        ev_df = build_trial_events(paths["events"])
        ev_df = ev_df[ev_df["seg_len"].isin([5, 9, 13])].copy()

        merged = pd.merge_asof(
            ev_df.sort_values("timestamp"),
            pupil_df.sort_values("pupil_timestamp"),
            left_on="timestamp", right_on="pupil_timestamp",
            direction="nearest", tolerance=1.0
        )
        merged["subject"] = sub_id
        merged["condition"] = merged["seg_len"]

        # ----- baseline 후보 ①: 논문 방식 (trial 시작 전 2초 구간 평균) -----
        # trial_seg별로 첫 자극 제시 시점(trial_start)을 구하고,
        # 그 직전 2초간의 동공 크기 평균을 baseline으로 계산
        trial_starts = ev_df.groupby("trial_seg")["timestamp"].min()
        pretrial_baselines = {}
        pupil_sorted = pupil_df.sort_values("pupil_timestamp")
        for seg, start_ts in trial_starts.items():
            window = pupil_sorted[
                (pupil_sorted["pupil_timestamp"] >= start_ts - 2.0) &
                (pupil_sorted["pupil_timestamp"] < start_ts)
            ]
            pretrial_baselines[seg] = window["diameter_3d"].mean() if len(window) > 0 else None

        merged["baseline_pretrial"] = merged["trial_seg"].map(pretrial_baselines)

        # ----- baseline 후보 ②: position 1 값 (기존 방식, 비교용으로 유지) -----
        pos1_values = (
            merged[merged["position"] == 1]
            .set_index("trial_seg")["diameter_3d"]
        )
        merged["baseline_pos1"] = merged["trial_seg"].map(pos1_values)

        # ----- baseline 후보 ③: 세션 전체 평균 (참가자당 하나의 값, 표본이 많아 안정적) -----
        session_baseline = pupil_df["diameter_3d"].mean()
        merged["baseline_session"] = session_baseline

        result_chunk = merged[[
            "subject", "condition", "trial_seg", "position",
            "diameter_3d", "baseline_pretrial", "baseline_pos1", "baseline_session"
        ]]

        # 결과 파일에 이어붙이기 (파일 없으면 헤더 포함해서 새로 생성)
        write_header = not os.path.exists(RESULT_PATH)
        result_chunk.to_csv(RESULT_PATH, mode="a", index=False, header=write_header)

        print(f"  성공: {len(result_chunk)}개 이벤트 매칭됨 (누적 저장 완료)")

    except Exception as e:
        print(f"  ⚠️ 처리 중 오류(이 참가자는 건너뜀): {e}")

    finally:
        # 원본 큰 파일은 용량 절약을 위해 삭제
        for p in paths.values():
            if os.path.exists(p):
                os.remove(p)
        # 처리 완료(성공/실패 무관) 기록 -> 다음 실행 시 재시도 안 함
        with open(DONE_LOG_PATH, "a") as f:
            f.write(sub_id + "\n")
        done_subjects.add(sub_id)

        # 메모리 명시적으로 해제 (RAM 누적 방지)
        for var_name in ["pupil_df", "ev_df", "merged", "result_chunk"]:
            try:
                del globals()[var_name]
            except KeyError:
                pass
        gc.collect()

print("\n\n=== 전체 처리 완료 ===")
if os.path.exists(RESULT_PATH):
    final = pd.read_csv(RESULT_PATH)
    print(f"결과 저장 위치: {RESULT_PATH}")
    print(f"총 행 수: {len(final):,}, 참가자 수: {final['subject'].nunique()}")
else:
    print("처리된 참가자가 없습니다.")