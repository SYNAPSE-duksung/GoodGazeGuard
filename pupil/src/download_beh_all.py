# ============================================================
# STEP 2: 84명 전체 beh.tsv 수집 (용량 작아서 삭제 없이 통째로 보관)
# 로컬(VS Code 등) / Colab 어디서든 그대로 실행 가능
# STEP 1과 독립적으로 실행 가능합니다 (SUBJECTS 목록을 다시 조회함).
# ============================================================

import subprocess
import os
import pandas as pd

S3_BASE = "s3://openneuro.org/ds003838"
DATA_DIR = "pupil/data/beh_raw"
os.makedirs(DATA_DIR, exist_ok=True)


def get_actual_subject_list():
    result = subprocess.run(
        ["aws", "s3", "ls", "--no-sign-request", "s3://openneuro.org/ds003838/"],
        capture_output=True, text=True
    )
    subs = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("PRE") and "sub-" in line:
            folder_name = line.split("PRE")[-1].strip().rstrip("/")
            subs.append(folder_name.replace("sub-", ""))
    return sorted(subs)


EXCLUDE = {"017", "094"}
SUBJECTS = [s for s in get_actual_subject_list() if s not in EXCLUDE]
print(f"대상 참가자 수: {len(SUBJECTS)}")


def parse_accuracy(trigger_string) -> float:
    """triggerCorrect 비트스트링을 정답률(0~1)로 변환"""
    if pd.isna(trigger_string):
        return None
    bits = [int(c) for c in str(trigger_string) if c in "01"]
    return sum(bits) / len(bits) if bits else None


all_beh = []
failed = []

for sub_id in SUBJECTS:
    local_path = os.path.join(DATA_DIR, f"sub-{sub_id}_beh.tsv")
    result = subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request",
         f"{S3_BASE}/sub-{sub_id}/beh/sub-{sub_id}_task-memory_beh.tsv", local_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        failed.append(sub_id)
        continue

    df = pd.read_csv(local_path, sep="\t")
    df["subject"] = int(sub_id)
    df["accuracy"] = df["triggerCorrect"].apply(parse_accuracy)
    all_beh.append(df[["subject", "trial", "condition", "accuracy"]])

    os.remove(local_path)  # 원본 tsv는 지우고 요약만 유지 (용량은 작지만 정리 차원)

beh_all = pd.concat(all_beh, ignore_index=True)
os.makedirs("pupil/data", exist_ok=True)
beh_all.to_csv("pupil/data/beh_all.csv", index=False)

print(f"\n수집 완료: {len(beh_all)}행, 참가자 {beh_all['subject'].nunique()}명")
if failed:
    print(f"실패한 참가자: {failed}")
print("저장 위치: pupil/data/beh_all.csv")