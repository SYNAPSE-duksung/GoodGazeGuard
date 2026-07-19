# 시선 움직임 기반 인지부하 분석 파이프라인

시선 움직임(Gaze Movement) 특징을 이용하여 인지부하(Cognitive Load)를 추정하기 위한 모듈입니다.

본 모듈은 [OpenNeuro Eye Tracking Dataset](https://openneuro.org/datasets/ds003838/versions/1.0.6)의 시선 좌표 데이터를 전처리한 뒤, 인지부하와 관련된 Gaze Feature를 추출하고 K-Means Clustering을 통해 Feature 공간에서 인지부하 수준이 자연스럽게 분리되는지 분석합니다.

향후에는 Pupil, Blink, rPPG와 Feature-level Fusion을 수행하여 멀티모달 기반 인지부하 및 거짓말 판별 시스템으로 확장하는 것을 목표로 합니다.


# 전체 파이프라인

```
OpenNeuro Eye Tracking Dataset
(EEG + Eye Tracking + ECG + PPG + Behavioral Data)
                │
                ▼
load_data.py
(Eye Tracking Data Loading)
                │
                ▼
event_parser.py
(Fixation Event Detection)
                │
                ▼
feature_extract.py
(Gaze Feature Extraction)
                │
                ▼
feature.csv
                │
                ▼
kmeans_clustering.py
(Unsupervised Clustering)
```

# 파일 구성

| 파일 | 역할 |
|------|------|
| load_data.py | Eye Tracking 데이터 로드 및 Trial 단위 분리 |
| event_parser.py | Fixation Event Parsing |
| feature_extract.py | Gaze Feature 추출 |
| kmeans_clustering.py | K-Means Clustering 및 시각화 |

# 사용 데이터셋

### OpenNeuro Digit Span Dataset

본 프로젝트에서는 OpenNeuro에서 제공하는 Digit Span Task 데이터를 사용하였습니다.

데이터셋은 다음과 같은 생체신호를 함께 제공합니다.

- Eye Tracking
- EEG
- Pupillometry
- ECG
- Photoplethysmography (PPG)
- Behavioral Data

현재 Gaze 모듈에서는 Eye Tracking 데이터만 사용하였습니다.

### Task
- Rest
- Digit Span Task

### Cognitive Load Level

Digit Span Task 난이도를 이용하여 인지부하 수준을 다음과 같이 가정하였습니다.

| Task | Cognitive Load |
|------|----------------|
| 5 Digit | Low |
| 9 Digit | Medium |
| 13 Digit | High |

※ 데이터셋에는 공식적인 Cognitive Load Label이 존재하지 않으며, Task Difficulty를 인지부하 수준으로 가정하여 분석하였습니다.


# 사용한 Feature

## Spatial Features

- Scanpath Length
- Gaze Dispersion
- Hull Area
- Center Distance STD

## Movement Features

- Movement Mean
- Movement Coefficient of Variation (CV)
- Movement Skewness

## Temporal Features

- Velocity Mean
- Velocity STD

## Fixation Features

- Mean Fixation Duration
- Fixation Count

최종적으로 Feature 간 상관관계 분석(Correlation Analysis)을 통해 중복성이 높은 Feature를 제거하고 총 10개의 Feature를 사용하였습니다.


# 전처리

- Confidence > 0.6
- Normalized Gaze Coordinate 사용
- 화면 밖 좌표 제거
- Trial 단위 분리
- Sample 수가 5 미만인 Trial 제거


# Clustering

추출된 Gaze Feature를 이용하여 K-Means Clustering을 수행하였습니다.

Silhouette Score를 기준으로 최적의 Cluster 개수를 비교하였습니다.

| k | Silhouette Score |
|---|------------------|
| 2 | 0.375 |
| 3 | 0.204 |
| 4 | 0.224 |
| 5 | 0.231 |

가장 높은 Silhouette Score는 k=2 (0.375)에서 나타났습니다. 이는 현재 사용한 Gaze Feature만으로는 3단계 인지부하(5/9/13 digit)를 명확히 구분하기보다, 2개의 주요 군집으로 분리되는 경향이 있음을 시사한다. 따라서 Gaze Feature만으로는 인지부하 수준을 충분히 구분하기 어려우며, 향후 Pupil, Blink, rPPG 등 다른 생체신호와의 멀티모달 융합이 필요하다고 판단하였습니다.


# 인지부하와 관련된 주요 Feature

| Feature | High Cognitive Load |
|----------|---------------------|
| Fixation Duration | ↑ |
| Scanpath Length | 변화 |
| Gaze Dispersion | ↑ |
| Velocity | 변화 |
| Blink Rate | ↓ (향후 추가 예정) |
| Pupil Diameter | ↑ (향후 추가 예정) |

※ 대부분의 연구에서는 절대 Threshold보다 Baseline 대비 변화량 또는 Machine Learning 기반 추정을 사용합니다.


# 현재 한계

- Gaze Feature만으로는 인지부하를 완전히 구분하기 어려움
- Ground Truth Cognitive Load Label 부재
- K-Means Cluster가 Task Difficulty와 완전히 대응하지 않음
- Pupil 및 Blink Feature 미포함


# 향후 계획

- Pupil Feature 추가
- Blink Feature 추가
- rPPG Feature와 Feature-level Fusion
- Temporal Synchronization
- Cognitive Load Classification
- Lie Detection Pipeline 구축
