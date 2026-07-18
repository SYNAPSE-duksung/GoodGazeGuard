"""
validate_rppg_accuracy.py
--------------------------
UBFC-rPPG의 ground_truth.txt(실측 PPG/HR)와, 우리 파이프라인이 rPPG로 추출한
HR을 비교해서 MAE(Mean Absolute Error)를 계산하는 검증 스크립트.

[중요] 이 스크립트는 "인지 부하 점수"가 맞는지를 검증하는 게 아님.
UBFC-rPPG엔 인지 부하 자체에 대한 ground truth가 없어서 그건 검증 불가.
대신 이 스크립트는 "우리가 뽑은 rPPG 신호(mean_hr 등)가 실제 맥박산소측정기
값과 얼마나 일치하는지"를 검증함 — 즉 인지 부하 점수의 재료가 되는 신호
자체의 신뢰도를 확인하는 단계.

UBFC-rPPG ground_truth.txt 포맷 (일반적으로 알려진 구조):
    1행: PPG 파형 값 (공백 구분)
    2행: HR(bpm) 값 (공백 구분)
    3행: timestamp(초) 값 (공백 구분)

파일마다 줄 순서/구분자가 다를 수 있어서, 먼저 --preview 옵션으로
실제 구조를 눈으로 확인하는 걸 권장.

사용법:
    # 1) 먼저 파일 구조 확인 (권장)
    python validate_rppg_accuracy.py --data_root ./UBFC_ROOT --preview subject1

    # 2) 전체 subject에 대해 MAE 계산
    python validate_rppg_accuracy.py --data_root ./UBFC_ROOT --out_dir ./results
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd

from rppg_pos import extract_rppg_from_video
from hrv_features import detect_peaks


# -----------------------------------------------------------------
# 1. ground_truth.txt 파싱
# -----------------------------------------------------------------
def parse_ground_truth(gt_path: str) -> dict:
    """
    3줄 포맷(PPG waveform / HR / timestamp)을 우선 시도.
    각 줄이 공백으로 구분된 숫자열이라고 가정.
    """
    with open(gt_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"ground_truth.txt 줄 수가 예상과 다름 ({len(lines)}줄): {gt_path}")

    def to_array(line):
        # 공백 또는 쉼표 구분 모두 대응
        sep = "," if "," in line else None
        return np.array([float(x) for x in line.split(sep)])

    result = {"raw_lines": len(lines)}

    try:
        result["ppg_waveform"] = to_array(lines[0])
    except ValueError:
        result["ppg_waveform"] = None

    if len(lines) >= 2:
        try:
            result["hr_bpm"] = to_array(lines[1])
        except ValueError:
            result["hr_bpm"] = None

    if len(lines) >= 3:
        try:
            result["timestamp_sec"] = to_array(lines[2])
        except ValueError:
            result["timestamp_sec"] = None

    return result


def preview_ground_truth(gt_path: str):
    """--preview 옵션용: 파일의 실제 구조를 사람이 눈으로 확인하도록 출력."""
    with open(gt_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    print(f"\n=== {gt_path} 구조 미리보기 ===")
    print(f"총 {len(lines)}줄")
    for i, line in enumerate(lines):
        values_preview = line[:120] + ("..." if len(line) > 120 else "")
        n_values = len(line.split(",")) if "," in line else len(line.split())
        print(f"  [{i}행] 값 개수={n_values} | 미리보기: {values_preview}")


# -----------------------------------------------------------------
# 2. subject 하나에 대한 rPPG HR vs ground truth HR "윈도우 단위" 비교
# -----------------------------------------------------------------
def compare_subject_windowed(
    subj_dir: str,
    window_sec: float = 10.0,
    step_sec: float = 5.0,
    use_mediapipe: bool = True,
) -> list:
    """
    영상 전체 평균이 아니라, window_sec 길이 구간을 step_sec 간격으로 슬라이딩하며
    같은 시간대의 gt_hr(ground truth 평균)과 our_hr(우리 rPPG 기반 추정)을 짝지어 비교.
    반환: 윈도우별 딕셔너리 리스트 (subject 실패 시 상태 메시지 1개짜리 리스트)
    """
    subject_id = os.path.basename(subj_dir)
    video_path = os.path.join(subj_dir, "vid.avi")
    gt_path = os.path.join(subj_dir, "ground_truth.txt")

    if not os.path.exists(video_path) or not os.path.exists(gt_path):
        return [{"subject": subject_id, "status": "파일 없음"}]

    gt = parse_ground_truth(gt_path)
    if gt.get("hr_bpm") is None or gt.get("timestamp_sec") is None:
        return [{"subject": subject_id, "status": "ground_truth HR/timestamp 파싱 실패"}]

    gt_hr = gt["hr_bpm"]
    gt_ts = gt["timestamp_sec"]

    try:
        rppg_signal, fps, _ = extract_rppg_from_video(video_path, use_mediapipe=use_mediapipe)
    except Exception as e:
        return [{"subject": subject_id, "status": f"rPPG 추출 실패: {e}"}]

    win_len = int(window_sec * fps)
    step_len = int(step_sec * fps)
    N = len(rppg_signal)

    if N < win_len:
        return [{"subject": subject_id, "status": f"영상이 window_sec({window_sec}s)보다 짧음"}]

    rows = []
    for start in range(0, N - win_len + 1, step_len):
        end = start + win_len
        start_sec = start / fps
        end_sec = end / fps

        window_signal = rppg_signal[start:end]
        _, rr = detect_peaks(window_signal, fps)
        our_hr = float(np.mean(60.0 / rr)) if len(rr) >= 2 else np.nan

        gt_mask = (gt_ts >= start_sec) & (gt_ts < end_sec)
        gt_hr_window = float(np.mean(gt_hr[gt_mask])) if gt_mask.any() else np.nan

        abs_error = abs(gt_hr_window - our_hr) if not (np.isnan(gt_hr_window) or np.isnan(our_hr)) else np.nan

        rows.append({
            "subject": subject_id,
            "status": "OK",
            "window_start_sec": start_sec,
            "window_end_sec": end_sec,
            "gt_hr": gt_hr_window,
            "our_hr": our_hr,
            "abs_error": abs_error,
        })

    if not rows:
        return [{"subject": subject_id, "status": "윈도우 생성 실패"}]

    return rows


# -----------------------------------------------------------------
# 3. 실행부
# -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="rPPG 추출 정확도(윈도우 단위 HR 비교) 검증")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", default="./results")
    parser.add_argument("--window_sec", type=float, default=10.0, help="비교 윈도우 길이(초)")
    parser.add_argument("--step_sec", type=float, default=5.0, help="윈도우 슬라이딩 간격(초)")
    parser.add_argument("--preview", default=None,
                         help="ground_truth.txt 구조만 미리 확인할 subject 이름 (예: subject1)")
    parser.add_argument("--no_mediapipe", action="store_true")
    args = parser.parse_args()

    if args.preview:
        gt_path = os.path.join(args.data_root, args.preview, "ground_truth.txt")
        if not os.path.exists(gt_path):
            print(f"파일을 찾을 수 없음: {gt_path}")
            return
        preview_ground_truth(gt_path)
        print("\n위 구조를 보고 1행=PPG파형, 2행=HR, 3행=timestamp 가정이 맞는지 확인하세요.")
        print("다르다면 parse_ground_truth() 함수의 lines[0]/[1]/[2] 매핑을 실제 구조에 맞게 조정해야 합니다.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    subject_dirs = sorted(glob.glob(os.path.join(args.data_root, "*")))

    all_rows = []
    for subj_dir in subject_dirs:
        if not os.path.isdir(subj_dir):
            continue
        subject_id = os.path.basename(subj_dir)
        print(f"[검증중] {subject_id} ...")
        rows = compare_subject_windowed(
            subj_dir, window_sec=args.window_sec, step_sec=args.step_sec,
            use_mediapipe=not args.no_mediapipe,
        )
        all_rows.extend(rows)

        ok_rows = [r for r in rows if r.get("status") == "OK" and not np.isnan(r.get("abs_error", np.nan))]
        if ok_rows:
            subj_mae = np.mean([r["abs_error"] for r in ok_rows])
            print(f"  윈도우 {len(ok_rows)}개 비교, subject 평균 MAE={subj_mae:.2f} bpm")
        else:
            status = rows[0].get("status", "알 수 없음")
            print(f"  [실패/데이터 없음] {status}")

    result_df = pd.DataFrame(all_rows)
    out_path = os.path.join(args.out_dir, "rppg_accuracy_validation_windowed.csv")
    result_df.to_csv(out_path, index=False)

    ok_df = result_df[(result_df.get("status") == "OK")].dropna(subset=["gt_hr", "our_hr"])
    print(f"\n=== 검증 완료: 전체 윈도우 {len(ok_df)}개 (subject {result_df['subject'].nunique()}명) ===")

    if len(ok_df) > 1:
        errors = (ok_df["gt_hr"] - ok_df["our_hr"]).values
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        corr = np.corrcoef(ok_df["gt_hr"], ok_df["our_hr"])[0, 1]

        # Bland-Altman 스타일 통계: 두 측정치 차이의 평균/표준편차, 95% 일치 한계
        bias = np.mean(errors)
        loa_std = np.std(errors)
        loa_upper = bias + 1.96 * loa_std
        loa_lower = bias - 1.96 * loa_std

        print(f"윈도우 단위 MAE: {mae:.2f} bpm")
        print(f"윈도우 단위 RMSE: {rmse:.2f} bpm")
        print(f"Pearson 상관계수(r): {corr:.3f}")
        print(f"Bland-Altman bias(평균 차이): {bias:+.2f} bpm")
        print(f"Bland-Altman 95% 일치 한계: [{loa_lower:+.2f}, {loa_upper:+.2f}] bpm")
        print("\n(참고) 공개 논문들에서 UBFC-rPPG로 POS 알고리즘 벤치마크시 MAE가 대략 1~3 bpm대,")
        print("상관계수는 보통 0.9 이상으로 보고되는 경우가 많음. 이보다 많이 못 미치면 특정")
        print("subject/구간에서 얼굴 검출이나 피크 검출이 불안정한 경우가 있는지 csv에서 직접 확인 필요.")

        print("\n=== subject별 평균 MAE (나쁜 순) ===")
        subj_mae = ok_df.assign(abs_err=np.abs(errors)).groupby("subject")["abs_err"].mean()
        print(subj_mae.sort_values(ascending=False).head(10))
    else:
        print("비교 가능한 윈도우가 부족함 (gt/our 둘 다 유효한 윈도우가 2개 미만)")

    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()