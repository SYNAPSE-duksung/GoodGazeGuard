"""
generate_result_figures.py
----------------------------
지금까지 만든 결과 csv들을 이용해서 발표/보고서용 그래프(PNG)를 생성.

만드는 그래프:
  1. HR 산점도 (gt_hr vs our_hr) — validate_rppg_accuracy.py 결과 이용
  2. Bland-Altman plot — 같은 결과로 두 측정치의 일치도 시각화
  3. 인지 부하 점수 분포 히스토그램 — cognitive_load_score.py 결과 이용
  4. subject별 평균 인지 부하 점수 막대그래프
  5. (선택) rPPG 파형 + 검출된 피크 예시 — 특정 subject 영상 하나를 골라
     피크 검출이 제대로 되는지 시각적으로 확인

그래프 텍스트는 전부 영어로 작성 (한글 폰트 미설치 서버에서 깨지는 것 방지).

의존 라이브러리:
    pip install matplotlib pandas numpy

사용 예:
    # HR 검증 결과 + 점수 결과 그래프 한 번에 생성
    python generate_result_figures.py \
        --validation_csv ./results/rppg_accuracy_validation_windowed.csv \
        --scores_csv ./results/cognitive_load_scores.csv \
        --out_dir ./results/figures

    # 특정 subject의 rPPG 파형 + 피크 예시도 같이 그리고 싶을 때
    python generate_result_figures.py \
        --validation_csv ./results/rppg_accuracy_validation_windowed.csv \
        --scores_csv ./results/cognitive_load_scores.csv \
        --out_dir ./results/figures \
        --example_video ./UBFC_ROOT/subject20/vid.avi
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 서버 환경(디스플레이 없음)에서도 저장 가능하도록
import matplotlib.pyplot as plt


# -----------------------------------------------------------------
# 1. HR 산점도 (gt vs our)
# -----------------------------------------------------------------
def plot_hr_scatter(validation_csv: str, out_dir: str):
    df = pd.read_csv(validation_csv)
    df = df[df.get("status") == "OK"].dropna(subset=["gt_hr", "our_hr"])
    if df.empty:
        print("[스킵] HR 산점도: 유효한 윈도우 데이터 없음")
        return

    corr = np.corrcoef(df["gt_hr"], df["our_hr"])[0, 1]
    lims = [min(df["gt_hr"].min(), df["our_hr"].min()) - 5,
            max(df["gt_hr"].max(), df["our_hr"].max()) + 5]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["gt_hr"], df["our_hr"], alpha=0.4, s=15, color="#3366cc")
    ax.plot(lims, lims, "--", color="gray", linewidth=1, label="Perfect agreement (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Ground truth HR (bpm)")
    ax.set_ylabel("rPPG-estimated HR (bpm)")
    ax.set_title(f"HR Estimation Accuracy (windowed)\nPearson r = {corr:.3f}, n = {len(df)}")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, "hr_scatter.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"저장: {out_path}")


# -----------------------------------------------------------------
# 2. Bland-Altman plot
# -----------------------------------------------------------------
def plot_bland_altman(validation_csv: str, out_dir: str):
    df = pd.read_csv(validation_csv)
    df = df[df.get("status") == "OK"].dropna(subset=["gt_hr", "our_hr"])
    if df.empty:
        print("[스킵] Bland-Altman: 유효한 윈도우 데이터 없음")
        return

    mean_hr = (df["gt_hr"] + df["our_hr"]) / 2.0
    diff_hr = df["our_hr"] - df["gt_hr"]  # our - gt (양수면 우리가 더 높게 추정)

    bias = diff_hr.mean()
    sd = diff_hr.std()
    upper_loa = bias + 1.96 * sd
    lower_loa = bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_hr, diff_hr, alpha=0.4, s=15, color="#cc3366")
    ax.axhline(bias, color="black", linewidth=1.2, label=f"Bias = {bias:+.2f} bpm")
    ax.axhline(upper_loa, color="gray", linestyle="--", linewidth=1,
               label=f"+1.96 SD = {upper_loa:+.2f}")
    ax.axhline(lower_loa, color="gray", linestyle="--", linewidth=1,
               label=f"-1.96 SD = {lower_loa:+.2f}")
    ax.set_xlabel("Mean of GT and rPPG HR (bpm)")
    ax.set_ylabel("Difference (rPPG - GT), bpm")
    ax.set_title("Bland-Altman Plot: rPPG vs Ground Truth HR")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, "bland_altman.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"저장: {out_path}")


# -----------------------------------------------------------------
# 3. 인지 부하 점수 분포 히스토그램
# -----------------------------------------------------------------
def plot_score_distribution(scores_csv: str, out_dir: str):
    df = pd.read_csv(scores_csv)
    if "cognitive_load_score" not in df.columns:
        print("[스킵] 점수 분포: cognitive_load_score 컬럼 없음")
        return
    scores = df["cognitive_load_score"].dropna()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(scores, bins=25, color="#4c9f70", edgecolor="white", alpha=0.85)
    ax.axvline(scores.mean(), color="black", linestyle="--", linewidth=1.2,
               label=f"Mean = {scores.mean():.1f}")
    ax.set_xlabel("Cognitive Load Score (0-100)")
    ax.set_ylabel("Number of windows")
    ax.set_title(f"Cognitive Load Score Distribution (n = {len(scores)} windows)")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, "score_distribution.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"저장: {out_path}")


# -----------------------------------------------------------------
# 4. subject별 평균 점수 막대그래프
# -----------------------------------------------------------------
def plot_subject_scores_bar(scores_csv: str, out_dir: str):
    df = pd.read_csv(scores_csv)
    if "cognitive_load_score" not in df.columns or "subject" not in df.columns:
        print("[스킵] subject별 점수: 필요한 컬럼 없음")
        return

    subj_mean = df.groupby("subject")["cognitive_load_score"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4, len(subj_mean) * 0.25)))
    colors = plt.cm.RdYlGn_r((subj_mean - subj_mean.min()) / (subj_mean.max() - subj_mean.min() + 1e-9))
    ax.barh(subj_mean.index, subj_mean.values, color=colors)
    ax.invert_yaxis()  # 위에서부터 높은 점수 순
    ax.set_xlabel("Mean Cognitive Load Score (0-100)")
    ax.set_title("Mean Cognitive Load Score by Subject")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "subject_scores_bar.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"저장: {out_path}")


# -----------------------------------------------------------------
# 5. (선택) rPPG 파형 + 검출된 피크 예시
# -----------------------------------------------------------------
def plot_example_waveform(video_path: str, out_dir: str, duration_sec: float = 15.0):
    from rppg_pos import extract_rppg_from_video
    from hrv_features import detect_peaks

    print(f"[처리중] 예시 파형 추출: {video_path}")
    rppg_signal, fps, timestamps = extract_rppg_from_video(video_path)

    n_samples = int(duration_sec * fps)
    signal_clip = rppg_signal[:n_samples]
    time_clip = timestamps[:n_samples]

    peaks, rr = detect_peaks(signal_clip, fps)
    est_hr = 60.0 / np.mean(rr) if len(rr) >= 2 else None

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_clip, signal_clip, color="#3366cc", linewidth=1)
    ax.plot(time_clip[peaks], signal_clip[peaks], "ro", markersize=5, label="Detected peaks")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("rPPG signal (a.u.)")
    title = f"Example rPPG Waveform with Detected Peaks (first {duration_sec:.0f}s)"
    if est_hr:
        title += f"\nEstimated HR in this segment: {est_hr:.1f} bpm"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    subject_name = os.path.basename(os.path.dirname(video_path))
    out_path = os.path.join(out_dir, f"example_waveform_{subject_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"저장: {out_path}")


# -----------------------------------------------------------------
# 실행부
# -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="결과 그래프(PNG) 생성")
    parser.add_argument("--validation_csv", default=None,
                         help="validate_rppg_accuracy.py 가 만든 rppg_accuracy_validation_windowed.csv 경로")
    parser.add_argument("--scores_csv", default=None,
                         help="cognitive_load_score.py 가 만든 cognitive_load_scores.csv 경로")
    parser.add_argument("--out_dir", default="./results/figures")
    parser.add_argument("--example_video", default=None,
                         help="rPPG 파형 예시를 그릴 특정 subject의 vid.avi 경로 (선택)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.validation_csv:
        plot_hr_scatter(args.validation_csv, args.out_dir)
        plot_bland_altman(args.validation_csv, args.out_dir)

    if args.scores_csv:
        plot_score_distribution(args.scores_csv, args.out_dir)
        plot_subject_scores_bar(args.scores_csv, args.out_dir)

    if args.example_video:
        plot_example_waveform(args.example_video, args.out_dir)

    if not any([args.validation_csv, args.scores_csv, args.example_video]):
        print("--validation_csv, --scores_csv, --example_video 중 최소 하나는 지정해야 함")


if __name__ == "__main__":
    main()