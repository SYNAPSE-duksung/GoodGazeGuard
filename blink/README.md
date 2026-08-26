# GazeGuard - Blink 전처리 파이프라인

OpenNeuro [ds003838](https://openneuro.org/datasets/ds003838) 데이터셋을 이용해
Blink 신호로부터 trial 단위 feature를 추출하고, pupil/gaze 결과와 병합 가능한
형태로 export하는 파이프라인입니다.

GazeGuard 팀 프로젝트(Gaze/Pupil/Blink 기반 인지 과부하 측정을 통한 진술 신뢰성
평가 및 강압 수사 방지 시스템)의 Blink 파트 + 공통 전처리 파이프라인
(시계열 시간 동기화, 단위 Trial 분할) 담당 부분입니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `blink_load.py` | `pupil.tsv`에서 blink 플래그(0/1)를 로드 |
| `difficulty_label.py` | 자릿수(5/9/13) → Low/Medium/High 난이도 라벨 통일 |
| `beh_loader.py` | `beh` 폴더의 정답률(NCorrect) 데이터 로드 |
| `blink_feature.py` | trial 단위 blink feature 추출, `train_branch1_lightgbm.py`와 병합 가능한 형태로 export |
| `overload_threshold.py` | 정답률 Z-score + K-means로 피험자별 인지 과부하 경계값 산출 |

## 데이터 구조 (BIDS)

```
D:\OpenNeuro\
└── sub-XXX\
    ├── beh\
    │   └── sub-XXX_task-memory_beh.tsv       # NCorrect, condition 등 정답률 데이터
    ├── ecg\
    ├── eeg\
    └── pupil\
        ├── sub-XXX_task-memory_events.tsv    # 이벤트 라벨 (trial 분할용)
        └── sub-XXX_task-memory_pupil.tsv     # gaze/pupil/blink 원신호 (data_part 컬럼으로 구분)
```

- blink은 별도 파일이 아니라 `pupil.tsv`의 `data_part == "gaze"` 행에 붙는
  0/1 플래그입니다 (`sub-XXX_task-memory_eyetrack.json` sidecar 참고).
- `BASE_PATH`(현재 `D:\OpenNeuro`)는 실행 환경에 맞게 각 파일 상단에서 수정하세요.

## 실행 순서

```bash
# 1. subject_split.csv (train/valid/test 피험자 분리) 필요
#    - 별도 스크립트(data_split.py, 저장소 미포함)로 생성하거나 직접 준비

# 2. blink feature 추출 (output/blink/blink_features_ours.csv 생성)
python blink_feature.py

# 3. 정답률 Z-score + K-means 임계값 산출 (overload_labels.csv 생성)
python overload_threshold.py
```

이후 `output/blink/blink_features_ours.csv`는 pupil/gaze 브랜치의
`train_branch1_lightgbm.py --blink` 인자로 넘겨서 병합할 수 있습니다.

## 요구 사항

```bash
pip install -r requirements.txt
```