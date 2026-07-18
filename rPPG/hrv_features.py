"""
hrv_features.py
---------------
rPPG 파형에서 피크(심박)를 검출하고, 슬라이딩 윈도우 단위로
HRV(심박변이도) 기반 인지 과부하 특징을 추출하는 모듈.

인지 과부하/스트레스 상태일 때 일반적으로 나타나는 경향:
  - 교감신경 활성화 → 평균 HR 증가, HRV(SDNN, RMSSD) 감소
  - LF/HF ratio 증가 (교감/부교감 균형이 교감 쪽으로 치우침)
  - pNN50 감소

의존 라이브러리:
    pip install numpy scipy pandas
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch


# -----------------------------------------------------------------
# 1. 피크(심박) 검출 → RR interval 계산
# -----------------------------------------------------------------
def estimate_hr_fft(rppg_signal: np.ndarray, fps: float, min_hr=42, max_hr=180):
    """
    시간영역 피크 검출 없이, 주파수 스펙트럼에서 가장 강한 성분을 찾아
    대략적인 심박수를 추정. dicrotic notch 같은 파형 형태 문제에 영향을
    덜 받기 때문에, 피크 검출 시 탐색 범위를 좁히는 사전 추정치로 사용.
    """
    if len(rppg_signal) < fps * 2:  # 너무 짧으면 신뢰 불가
        return None

    freqs, psd = welch(rppg_signal, fs=fps, nperseg=min(len(rppg_signal), int(fps * 8)))
    band = (freqs >= min_hr / 60.0) & (freqs <= max_hr / 60.0)
    if not band.any() or psd[band].max() <= 0:
        return None

    dominant_freq = freqs[band][np.argmax(psd[band])]
    return dominant_freq * 60.0  # bpm


def detect_peaks(rppg_signal: np.ndarray, fps: float, min_hr=42, max_hr=180):
    """
    rPPG 신호에서 심박 피크 인덱스를 찾고, RR interval(초 단위)을 반환.

    [수정 사항] 심박수가 낮을 때(한 주기가 길어질 때) dicrotic notch(이완기의
    작은 반사파)가 별도 피크로 잘못 검출되어 실제 HR의 거의 2배로 잡히는
    문제가 있었음. 이를 방지하기 위해:
      1) FFT로 대략적인 HR을 먼저 추정해서 탐색 가능한 최대 HR 범위를 좁히고
      2) prominence(피크 돌출 정도) 기준을 추가해 작은 notch 피크를 걸러냄
    """
    approx_hr = estimate_hr_fft(rppg_signal, fps, min_hr, max_hr)
    if approx_hr is not None:
        # FFT 추정치의 1.4배까지만 허용 (진짜 배음 검출 오류 방지, 약간의 여유는 둠)
        adaptive_max_hr = min(max_hr, approx_hr * 1.4)
    else:
        adaptive_max_hr = max_hr

    min_distance = int(fps * 60.0 / adaptive_max_hr)
    prominence = 0.35 * np.std(rppg_signal) if np.std(rppg_signal) > 1e-8 else None

    peaks, _ = find_peaks(rppg_signal, distance=max(1, min_distance), prominence=prominence)

    if len(peaks) < 2:
        return peaks, np.array([])

    rr_intervals = np.diff(peaks) / fps  # seconds
    # 생리학적으로 불가능한 RR (너무 짧거나 긴) 제거
    valid = (rr_intervals > 60.0 / max_hr) & (rr_intervals < 60.0 / min_hr)
    rr_intervals = rr_intervals[valid]

    return peaks, rr_intervals


# -----------------------------------------------------------------
# 2. HRV 시간영역 특징
# -----------------------------------------------------------------
def time_domain_features(rr_intervals: np.ndarray) -> dict:
    if len(rr_intervals) < 2:
        return {"mean_hr": np.nan, "sdnn": np.nan, "rmssd": np.nan, "pnn50": np.nan}

    hr = 60.0 / rr_intervals
    diff_rr = np.diff(rr_intervals) * 1000.0  # ms

    return {
        "mean_hr": hr.mean(),
        "sdnn": rr_intervals.std() * 1000.0,          # ms
        "rmssd": np.sqrt(np.mean(diff_rr ** 2)),       # ms
        "pnn50": np.mean(np.abs(diff_rr) > 50) * 100,  # %
    }


# -----------------------------------------------------------------
# 3. HRV 주파수영역 특징 (LF/HF)
# -----------------------------------------------------------------
def freq_domain_features(rr_intervals: np.ndarray, resample_hz=4.0) -> dict:
    """
    불균일 간격의 RR interval을 균일 간격으로 리샘플링한 뒤 Welch PSD로
    LF(0.04-0.15Hz), HF(0.15-0.4Hz) 파워와 LF/HF ratio를 계산.
    """
    if len(rr_intervals) < 4:
        return {"lf_power": np.nan, "hf_power": np.nan, "lf_hf_ratio": np.nan}

    rr_times = np.cumsum(rr_intervals)
    rr_times -= rr_times[0]

    uniform_times = np.arange(0, rr_times[-1], 1.0 / resample_hz)
    if len(uniform_times) < 8:
        return {"lf_power": np.nan, "hf_power": np.nan, "lf_hf_ratio": np.nan}

    interp_rr = np.interp(uniform_times, rr_times, rr_intervals)

    freqs, psd = welch(interp_rr, fs=resample_hz, nperseg=min(256, len(interp_rr)))

    lf_band = (freqs >= 0.04) & (freqs < 0.15)
    hf_band = (freqs >= 0.15) & (freqs < 0.4)

    lf_power = np.trapz(psd[lf_band], freqs[lf_band]) if lf_band.any() else np.nan
    hf_power = np.trapz(psd[hf_band], freqs[hf_band]) if hf_band.any() else np.nan
    lf_hf_ratio = lf_power / hf_power if hf_power and hf_power > 0 else np.nan

    return {"lf_power": lf_power, "hf_power": hf_power, "lf_hf_ratio": lf_hf_ratio}


# -----------------------------------------------------------------
# 4. 신호 품질 / 파형 형태 특징 (모션·노이즈 proxy)
# -----------------------------------------------------------------
def waveform_shape_features(rppg_window: np.ndarray) -> dict:
    if len(rppg_window) < 4 or rppg_window.std() < 1e-8:
        return {"signal_std": np.nan, "signal_skew": np.nan, "signal_kurtosis": np.nan}

    from scipy.stats import skew, kurtosis
    return {
        "signal_std": rppg_window.std(),
        "signal_skew": skew(rppg_window),
        "signal_kurtosis": kurtosis(rppg_window),
    }


# -----------------------------------------------------------------
# 5. 슬라이딩 윈도우로 전체 특징 테이블 생성
# -----------------------------------------------------------------
def extract_windowed_features(
    rppg_signal: np.ndarray,
    fps: float,
    window_sec: float = 10.0,
    step_sec: float = 5.0,
) -> pd.DataFrame:
    """
    window_sec 길이 윈도우를 step_sec 간격으로 슬라이딩하며 특징 테이블 생성.
    각 행 = 한 윈도우, 열 = HRV/파형 특징.
    """
    win_len = int(window_sec * fps)
    step_len = int(step_sec * fps)
    N = len(rppg_signal)

    rows = []
    for start in range(0, N - win_len + 1, step_len):
        end = start + win_len
        window = rppg_signal[start:end]

        _, rr = detect_peaks(window, fps)
        feats = {}
        feats.update(time_domain_features(rr))
        feats.update(freq_domain_features(rr))
        feats.update(waveform_shape_features(window))
        feats["window_start_sec"] = start / fps
        feats["window_end_sec"] = end / fps

        rows.append(feats)

    return pd.DataFrame(rows)