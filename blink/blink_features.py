"""
blink_features.py
=============================================================
Cho, Y. (2021). "Rethinking Eye-blink: Assessing Task Difficulty through
Physiological Representation of Spontaneous Blinking." CHI 2021.
방법론(Section 3.2)을 그대로 구현한 blink 신호처리 모듈.

이 파일이 하는 일:
  1) blink 불리언 컬럼 → blink onset(깜빡임 시작) 이벤트 시각 추출
  2) Lomb-Scargle periodogram 기반 time-frequency 스펙트로그램 생성
  3) Blink Entropy(BE) 계산 — 논문 식(1)
  4) 비교용 표준 지표(분당 깜빡임 횟수)
"""

import numpy as np
from scipy.signal import lombscargle

import config


# =============================================================
# 1. Blink onset 이벤트 추출
# =============================================================
def extract_blink_onsets(pupil_df) -> np.ndarray:
    """blink 불리언 컬럼에서 '깜빡임 시작 시점(rising edge)'의 timestamp만 추출."""
    blink = pupil_df[config.BLINK_COL].astype(bool).to_numpy()
    t = pupil_df[config.TIME_COL_PUPIL].to_numpy()
    if len(blink) < 2:
        return np.array([])
    rising = np.where((~blink[:-1]) & (blink[1:]))[0] + 1
    return t[rising]


def blink_interval_series(onset_times: np.ndarray):
    """
    Blink onset 시각들을 '깜빡임 간격(IBI, Inter-Blink Interval)' 시계열로 변환.

    심박변이도(HRV) 분석에서 R-R interval을 심박 발생 시각에 매칭해
    Lomb-Scargle periodogram을 적용하는 것과 동일한 방식. 두 번째 blink부터,
    그 시각에 '직전 blink와의 간격(초)'을 값으로 대응
    (단순히 각 이벤트에 상수 1을 대응시키면 신호에 분산이 없어 Lomb-Scargle이
    아무 주파수 성분도 못 잡아냄 — 반드시 실제로 변화하는 IBI 값을 써야 함.)

    Returns
    -------
    times : np.ndarray  - onset_times[1:] (간격이 정의되는 시각)
    values: np.ndarray  - 대응하는 IBI (초)
    """
    if len(onset_times) < 2:
        return np.array([]), np.array([])
    return onset_times[1:], np.diff(onset_times)


# =============================================================
# 2. Lomb-Scargle 기반 blink 스펙트로그램
# =============================================================
def blink_lombscargle_spectrogram(onset_times: np.ndarray,
                                   t_start: float, t_end: float,
                                   sub_id: str = "", show_progress: bool = False) -> np.ndarray:
  

    Returns
    -------
    np.ndarray, shape (n_windows, config.N_FREQ_BINS)
    """
    freqs = np.linspace(config.FREQ_MIN_HZ, config.FREQ_MAX_HZ, config.N_FREQ_BINS)
    angular_freqs = 2 * np.pi * freqs

    n_windows = int((t_end - t_start - config.WINDOW_SEC) // config.STEP_SEC) + 1
    if n_windows <= 0:
        return np.empty((0, config.N_FREQ_BINS))

    ibi_times, ibi_values = blink_interval_series(onset_times)
    spectrogram = np.zeros((n_windows, config.N_FREQ_BINS))

    window_range = range(n_windows)
    if show_progress:
        from tqdm import tqdm
        desc = f"   ㄴ {sub_id} 윈도우 처리" if sub_id else "   ㄴ 윈도우 처리"
        window_range = tqdm(window_range, total=n_windows, desc=desc, unit="win", leave=False)

    for w in window_range:
        w_start = t_start + w * config.STEP_SEC
        w_end = w_start + config.WINDOW_SEC
        mask = (ibi_times >= w_start) & (ibi_times < w_end)
        win_t, win_y = ibi_times[mask], ibi_values[mask]

        if len(win_t) < 2:
            continue  # 윈도우 안에 간격을 계산할 blink가 2개 미만이면 파워 0

        y = win_y - win_y.mean()  # DC(평균) 성분 제거
        spectrogram[w, :] = lombscargle(win_t, y, angular_freqs, normalize=True)

    return spectrogram


# =============================================================
# 3. Blink Entropy (논문 식 1) + 비교용 표준 지표
# =============================================================
def blink_entropy(spectrogram: np.ndarray, n_bins: int = config.ENTROPY_HIST_BINS) -> float:
    """
    BE(X) = -Σ p(x_ij) log2 p(x_ij)
    스펙트로그램 진폭 값의 정규화된 히스토그램을 확률분포 p로 사용.
    """
    if spectrogram.size == 0:
        return np.nan
    hist, _ = np.histogram(spectrogram.flatten(), bins=n_bins, density=False)
    total = hist.sum()
    if total == 0:
        return np.nan
    p = hist / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def blink_rate_per_min(onset_times: np.ndarray, t_start: float, t_end: float) -> float:
    """비교용 표준 지표: 분당 깜빡임 횟수."""
    duration_min = (t_end - t_start) / 60.0
    if duration_min <= 0:
        return np.nan
    return len(onset_times) / duration_min
