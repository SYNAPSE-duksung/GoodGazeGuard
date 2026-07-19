"""
plot_blink_spectrogram.py
=============================================================
run_pipeline.py가 저장한 참가자별 blink 스펙트로그램
(blink_spectrograms/{sub_id}_spectrogram.npy)을 히트맵으로 시각화

실행 전 준비:
    pip install matplotlib --break-system-packages

사용법 (PyCharm 터미널에서):
    python plot_blink_spectrogram.py sub-01
    (참가자 ID를 안 주면 blink_spectrograms/ 폴더에서 처음 찾은 파일을 사용)
"""

import sys
import os
import glob
import platform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import config

# ---- 한글 폰트 설정 ----
# matplotlib 기본 폰트(DejaVu Sans)는 한글을 지원하지 않아서, 설정 안 하면
# 그래프의 한글 라벨이 네모(□)로 깨져서 보임. OS별로 기본 탑재된 한글 폰트를
# 자동으로 골라 씀.
_SYSTEM = platform.system()
if _SYSTEM == "Darwin":       # macOS
    matplotlib.rcParams["font.family"] = "AppleGothic"
elif _SYSTEM == "Windows":
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
else:                          # Linux (나눔고딕 설치 필요: apt install fonts-nanum)
    matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False  # 한글 폰트 사용 시 마이너스 기호(-) 깨짐 방지


def load_spectrogram(sub_id: str) -> np.ndarray:
    path = os.path.join(config.OUTPUT_SPEC_DIR, f"{sub_id}_spectrogram.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}'를 찾을 수 없습니다.\n"
            f"→ run_pipeline.py를 먼저 실행해서 '{sub_id}'의 스펙트로그램을 생성해주세요."
        )
    return np.load(path)


def pick_first_available_subject() -> str:
    """참가자 ID를 안 넘겼을 때, 저장된 것 중 아무거나 하나 골라줌"""
    files = sorted(glob.glob(os.path.join(config.OUTPUT_SPEC_DIR, "*_spectrogram.npy")))
    if not files:
        raise FileNotFoundError(
            f"'{config.OUTPUT_SPEC_DIR}/' 안에 저장된 스펙트로그램이 없습니다.\n"
            f"→ run_pipeline.py를 먼저 실행하세요."
        )
    return os.path.basename(files[0]).replace("_spectrogram.npy", "")


def plot_spectrogram(sub_id: str, save_path: str = None, show: bool = True):
    spec = load_spectrogram(sub_id)  # shape: (n_windows, N_FREQ_BINS)

    if spec.size == 0:
        print(f"⚠️ {sub_id}: 스펙트로그램이 비어 있습니다 "
              f"(녹화 길이가 {config.WINDOW_SEC}초 미만이었을 가능성이 있음).")
        return None

    n_windows, n_freq = spec.shape

    # x축: 시간 (분 단위가 초 단위보다 읽기 편함)
    time_min = np.arange(n_windows) * config.STEP_SEC / 60.0

    # y축: 분당 깜빡임 횟수(blinks/min)
    freqs_hz = np.linspace(config.FREQ_MIN_HZ, config.FREQ_MAX_HZ, n_freq)
    blinks_per_min = freqs_hz * 60.0

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.pcolormesh(time_min, blinks_per_min, spec.T, shading="auto", cmap="viridis")
    ax.set_xlabel("시간 (분)")
    ax.set_ylabel("깜빡임 리듬 (blinks/min)")
    ax.set_title(f"Blink Spectrogram — {sub_id}\n(Lomb-Scargle periodogram, Cho 2021 방법론)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Lomb-Scargle Power (정규화됨)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"💾 저장됨: {save_path}")
    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    sub_id = sys.argv[1] if len(sys.argv) > 1 else pick_first_available_subject()
    print(f"📊 {sub_id} 스펙트로그램 시각화 중...")
    plot_spectrogram(sub_id, save_path=f"{sub_id}_spectrogram_heatmap.png", show=True)
