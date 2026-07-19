"""
config.py
=============================================================
전체 파이프라인(ds003838_pipeline / blink_features / labeling /
run_pipeline / plot_blink_spectrogram)이 공유하는 설정값.
"""

import os

# =============================================================
# OpenNeuro / S3
# =============================================================
DATASET_ID = "ds003838"
BUCKET = "openneuro.org"
HTTPS_ROOT = f"https://s3.amazonaws.com/{BUCKET}/{DATASET_ID}"
DEFAULT_TASK = "memory"

SUBJECTS_MISSING_PUPIL = {17, 94}
FS = 120

# =============================================================
# 네트워크 (다운로드 행/hang 방지)
# =============================================================
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 60
HARD_TIMEOUT_SEC = 1000        
MAX_RETRIES = 1
MAX_WORKERS = 1

# =============================================================
# 로컬 캐시
# =============================================================
USE_CACHE = True
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s3_cache")
MANUAL_DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manual_downloads")

# =============================================================
# pupil.tsv 컬럼명 / beh 라벨링 관련
# =============================================================
TIME_COL_PUPIL = "pupil_timestamp"
BLINK_COL = "blink"
CONDITION_COL = "condition"
BASELINE_CONDITIONS = [5, 9]

# =============================================================
# Blink 스펙트로그램
# =============================================================
FREQ_MIN_HZ = 0.033
FREQ_MAX_HZ = 0.4167
WINDOW_SEC = 61
STEP_SEC = 1
N_FREQ_BINS = 93
ENTROPY_HIST_BINS = 32

# =============================================================
# 출력 파일 경로
# =============================================================
OUTPUT_SPEC_DIR = "./blink_spectrograms"
SUMMARY_CSV = "blink_features_summary.csv"
BEH_LABELED_CSV = "beh_labeled.csv"
OVERLOAD_LABELS_CSV = "blink_overload_labels_personalized.csv"
