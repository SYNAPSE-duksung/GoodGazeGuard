# 동공 반응 기반 인지 부하 분석 파이프라인

동공(Pupil) 반응 특징을 이용하여 인지 부하(Cognitive Load) 정도를 정량화하기 위한 모듈입니다.

본 모듈은 [OpenNeuro ds003838](https://openneuro.org/datasets/ds003838/versions/1.0.6) 데이터셋의 동공 지름 데이터를 전처리한 뒤, 참가자별 상대적 부하 정도(개인화 라벨)를 정답률 기반으로 산출하고, Random Forest 회귀 모델로 동공 반응만으로 이 부하 정도를 예측하는 것을 목표로 합니다.

향후에는 Gaze, Blink, rPPG와 Feature-level Fusion을 수행하여 멀티모달 기반 인지 부하 및 거짓말 판별 시스템으로 확장하는 것을 목표로 합니다.
작성: soobin

---

## 전체 파이프라인

```
OpenNeuro ds003838 (S3)
    │  colab_cell1_download_and_reduce.py
    │  (신뢰도 높은 눈 선택 → 이상치 제거 → 선형보간 → baseline 정규화)
    ▼
combined_pupil_positions.csv (참가자 84명, 시행×순번 단위 동공 반응)
    │
    │  colab_cell4_download_beh_all.py
    │  (84명 시행별 정답률 수집)
    ▼
beh_all.csv
    │
    ▼
pupil_baseline_pipeline.py
    ├─ remember/control 시행 구분 (18-trial 블록 규칙)
    ├─ 시행 단위 Feature 7종 추출 (peak_val, peak_pos, early_mean,
    │   late_val, decline, plateau_len, volatility)
    ├─ 개인별 상대 부하 정도(Z-score) 라벨 생성
    ├─ Random Forest 회귀 모델 학습 (참가자 단위 GroupKFold)
    └─ 백분위 변환 → 부하 정도(%) + 정상/과부하 이진 라벨 산출
    ▼
pupil_load_score_labeled.csv (최종 결과)
    │
    ├─► evaluate_model_performance.py  → 참가자 단위 교차검증 성능 평가
    └─► generate_report_figures.py     → 결과 그래프 3종 생성
```

## 파일 구성

| 파일명 | 역할 |
| :--- | :--- |
| `colab_cell1_download_and_reduce.py` | OpenNeuro S3에서 84명 원본 pupil.tsv를 청크 단위로 다운로드하며 전처리(눈 선택·이상치 제거·보간·baseline 정규화)까지 수행, `combined_pupil_positions.csv` 생성 |
| `colab_cell4_download_beh_all.py` | 84명 beh.tsv 수집 및 시행별 정답률 계산, `beh_all.csv` 생성 |
| `cluster_load_threshold.py` | 정답률 분포에 K-means(k=2)를 적용해 정상/과부하 경계값(40.5%)을 데이터 기반으로 도출 (1회성 분석) |
| `pupil_baseline_pipeline.py` | 전체 파이프라인 메인 스크립트. feature 추출 → 라벨 생성 → 모델 학습 → 최종 결과 산출 |
| `evaluate_model_performance.py` | 참가자 단위 교차검증(GroupKFold) 기준 Accuracy/Precision/Recall/R²/상관계수 산출 |
| `generate_report_figures.py` | condition별 부하%, Feature 중요도, Feature 상관관계 heatmap 그래프 생성 |

## 사용 데이터셋

### OpenNeuro ds003838 (Pavlov et al., 2022)

- 참가자 84명 (전체 86명 중 동공 데이터 결측 2명 제외)
- Task: Digit Span (숫자 암기), memory(암기)/control(단순 청취) 조건 교차 수행
- 측정 데이터: Pupil(diameter_3d), EEG, ECG/PPG, Behavioral Data (현재 모듈에서는 Pupil + Behavioral만 사용)

### Cognitive Load 관련 조건

| Task | Digit 수 |
|---|---|
| Memory (Easy) | 5 |
| Memory (Medium) | 9 |
| Memory (Hard) | 13 |

※ 데이터셋에는 공식 Cognitive Load Label이 없어, Task Difficulty를 그대로 쓰는 대신 **참가자 개인의 쉬운 조건(5, 9 digit) 정답률 대비 상대적 저하 정도**를 인지 부하 라벨로 자체 산출하였음 (아래 "인지 부하 라벨링" 참고).

## 사용한 Feature

동공 크기가 자극 제시에 따라 "증가 → 정체 → 감소"의 3단계 패턴을 보인다는 선행 연구(Kosachenko et al., 2023)에 근거해, 반응의 크기·시점·변화 패턴을 중심으로 7개 Feature를 구성하였다.

| Feature | 의미 |
|---|---|
| `peak_val` | 시행 내 baseline 대비 최대 반응 크기 |
| `peak_pos` | 최대 반응이 나타난 자극 순번 |
| `early_mean` | 초반(1~5번째 자극) 평균 반응 |
| `late_val` | 마지막 자극 시점의 반응 |
| `decline` | 최대 반응 대비 마지막 시점까지의 감소량 |
| `plateau_len` | 반응 정체 구간의 길이 |
| `volatility` | 자극 간 반응 변화량의 표준편차 |

Feature 간 상관관계를 분석한 결과 절대값 0.7 이상의 심각한 다중공선성은 없었으며(최댓값 0.68), 핵심 판별 신호인 `peak_pos`-`decline` 간 상관은 -0.19로 낮아 서로 독립적인 정보를 제공함을 확인하였다.

## 전처리

- 참가자별 confidence(추적 신뢰도) 평균이 더 높은 눈(좌/우 중 1개) 선택
- diameter_3d 생리학적 범위(2~8mm) 벗어난 이상치 제거
- 직전 값 대비 급격한 변화(1.5mm 초과) 이상치로 판단, 결측 처리 후 선형 보간
- 각 시행 시작 직전 2초 구간 평균을 baseline으로 삼아 정규화
- Event 타임스탬프 간격으로 trial 경계 및 18-trial 블록 내 remember(4~15번째)/control(1~3, 16~18번째) 구간 구분

## 인지 부하 라벨링 (Personalized Z-score)

단일 절대 기준(예: "13 digit=과부하")으로 전체 참가자를 평가하지 않고, **참가자 개개인의 쉬운 조건(5, 9 digit) 정답률 분포**를 기준으로 상대적 부하 정도를 산출하였다.

1. 참가자별 baseline(5, 9 digit 정답률의 평균·표준편차) 계산
2. 전체 시행의 정답률을 이 baseline 기준 Z-score로 환산 (낮을수록 평소 대비 저하 = 부하 큼)
3. Z-score를 백분위 변환하여 0~100% "부하 정도 점수"로 산출
4. Z-score에 K-means(k=2)를 적용해 정상/과부하 경계값을 데이터 기반으로 도출 → **부하 정도 40.5%**를 경계로 채택
5. 이 경계값이 실제 난이도(5/9/13 digit)와 자연스럽게 대응하는지 교차검증 (5: 7.1% 과부하 / 9: 73.3% / 13: 98.2%)

정답률 자체를 부하의 절대적 정의로 사용하지 않고 개인차 보정 기준으로만 활용한 이유는, 선행 연구(Fuhl et al., 2023)에서 성과 지표가 동기부여·각성 수준 등 다른 요인에도 영향받아 인지 부하만을 독립적으로 반영하기 어렵다고 지적한 점을 반영한 것이다.

## 폴더 구조

```
pupil/
├── src/                                    ← 스크립트 6개
│   ├── download_and_reduce.py
│   ├── download_beh_all.py
│   ├── cluster_load_threshold.py
│   ├── pupil_baseline_pipeline.py
│   ├── evaluate_model_performance.py
│   └── generate_report_figures.py
├── data/                                   ← 원본/중간 데이터 (csv)
├── outputs/                                ← 최종 결과, 그래프(png)
└── README.md
```

## 실행 및 사용 방법

**1. 환경 설정 및 필수 라이브러리 설치**
```bash
pip install pandas numpy scikit-learn scipy matplotlib awscli
```

**2. 원본 데이터 다운로드 및 전처리 (최초 1회, 시간 소요)**

`pupil/` 폴더에서 실행 기준:
```bash
python src/download_and_reduce.py   # data/combined_pupil_positions.csv 생성
python src/download_beh_all.py      # data/beh_all.csv 생성
```

**3. 메인 파이프라인 실행 (feature 추출 → 라벨 생성 → 모델 학습)**
```bash
python src/pupil_baseline_pipeline.py
```
- 결과물: `outputs/pupil_load_score_labeled.csv` (시행별 Feature, 부하 정도 점수, 정상/과부하 라벨)

**4. 성능 평가 및 그래프 생성**
```bash
python src/evaluate_model_performance.py    # 참가자 단위 교차검증 성능 지표 출력
python src/generate_report_figures.py       # 그래프 3종(PNG) 생성 → outputs/
```

※ 모든 스크립트는 **`pupil/` 폴더에서 실행하는 것을 기준**으로 상대경로가 설정되어 있습니다. `src/` 안에서 직접 실행할 경우, 각 스크립트 상단의 경로 변수(`INPUT_PATH`, `OUTPUT_PATH` 등)를 `../data/...`, `../outputs/...` 형태로 맞춰야 합니다.

## 현재 결과

| 지표 | 값 |
|---|---|
| Accuracy (이진분류, 참가자 단위 교차검증) | 78.7% |
| R² (회귀, 참가자 단위 교차검증) | 0.42 |
| 상관계수 | 0.65 |
| Feature 중요도 1위 | peak_pos (53%) |
| Feature 중요도 2위 | decline (18%) |

condition별 평균 부하 정도는 5 digit 19.3% → 9 digit 54.7% → 13 digit 76.3%로 난이도 순서에 따라 매끄럽게 증가하여, 라벨링 방법론의 타당성을 뒷받침한다.

## 현재 한계

- **정답률 기반 라벨의 간접성**: 인지 부하를 직접 측정한 것이 아니라 수행 결과(정답률)로부터 역산한 라벨이라, 동기부여 등 다른 요인의 영향을 완전히 배제하지 못함
- **중간 난이도(9 digit) 판별 어려움**: 5, 13 digit 대비 경계 사례로 혼동이 상대적으로 잦음
- **실험실 데이터 기반**: 전용 아이트래커(Pupil Labs)로 측정된 데이터이며, 실제 웹캠 환경에서의 성능은 별도 검증 필요
- **과제 특성의 한계**: 숫자 암기라는 실험실 과제에서 확인된 패턴이, 최종 목표 상황(강압 면담)에서도 동일하게 나타날지는 미검증

## 향후 계획

- 중간 난이도 판별력 개선을 위한 Feature 추가 검토
- 다른 파트(Gaze, Blink, rPPG)와 Feature-level 또는 Late Fusion 결합
