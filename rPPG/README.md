# rPPG 기반 인지 부하 분석 파이프라인 (UBFC-rPPG)

시선/동공/눈깜빡임 팀과 함께 진행하는 "인지 과부하 → 거짓말 판별" 프로젝트의
rPPG 파트 코드입니다.
작성: hyeonjirhee

## 전체 흐름

```
얼굴 영상 (vid.avi)
    │  rppg_pos.py
    ▼
rPPG 파형 (POS 알고리즘, bandpass filtered)
    │  hrv_features.py (FFT 기반 사전 HR 추정 + prominence 필터로 피크 검출)
    ▼
윈도우별 HRV 특징 (mean_hr, SDNN, RMSSD, pNN50, LF/HF ratio, ...)
    │
    ├─► cognitive_load_score.py   → 인지 부하 연속 점수(0~100) [라벨 불필요, 추천]
    │
    └─► cognitive_load_pipeline.py → RandomForest 이진 분류기 (LOSO 평가) [라벨 있을 때]

검증 & 시각화
    validate_rppg_accuracy.py  → ground_truth 대비 윈도우 단위 HR 정확도 검증
    generate_result_figures.py → 산점도/Bland-Altman/점수분포/subject별 점수 그래프
```

## 파일 구성

| 파일 | 역할 |
|---|---|
| `rppg_pos.py` | 얼굴 ROI 검출(mediapipe FaceMesh 또는 Haar cascade) + POS 알고리즘으로 rPPG 신호 추출 |
| `hrv_features.py` | FFT 기반 사전 HR 추정 + prominence 필터로 개선된 피크 검출, HRV 시간/주파수영역 특징 계산, 슬라이딩 윈도우 특징 테이블 생성 |
| `cognitive_load_pipeline.py` | 전체 subject 순회 → 특징 테이블 생성 → (라벨 있을 시) 라벨 매칭 → RandomForest 이진 분류기 LOSO 학습/평가 |
| `cognitive_load_score.py` | 라벨 없이 HRV 특징을 조합해 인지 부하 연속 점수(0~100) 산출, reference 통계 fit/apply |
| `validate_rppg_accuracy.py` | `ground_truth.txt` 파싱, 윈도우 단위로 rPPG 추정 HR vs 실측 HR 비교 (MAE/RMSE/Pearson r/Bland-Altman) |
| `generate_result_figures.py` | 위 결과 csv들을 이용한 그래프(PNG) 생성 (HR 산점도, Bland-Altman, 점수 분포, subject별 점수, 파형+피크 예시) |

## 설치

```bash
pip install opencv-python mediapipe numpy scipy pandas scikit-learn joblib matplotlib
```
(mediapipe 설치가 어려운 환경이면 `--no_mediapipe` 옵션으로 Haar cascade fallback 사용 가능. 정확도는 다소 떨어짐)

**환경 관련 주의사항 (실제로 겪었던 문제들):**
- `opencv-python`과 `opencv-contrib-python`을 동시에 설치하면 mediapipe와 ABI 충돌로 **segfault**가 날 수 있음 → 하나만 설치
- mediapipe는 `opencv-contrib-python`을 의존성으로 요구하므로, 정합성 있게 맞추려면 `pip install "opencv-contrib-python==4.9.0.80"` 하나만 쓰는 걸 권장
- numpy 2.x와 opencv-python 4.9.x(numpy 1.x로 컴파일됨)는 호환 안 됨 → `pip install "numpy<2"` 필요
- mediapipe 0.10.30+ 일부 빌드에서 `import mediapipe as mp; mp.solutions...`가 `AttributeError`를 낼 수 있음 → 이 코드는 `from mediapipe.python.solutions import face_mesh`로 명시적 import해서 우회함 (이미 반영됨)

## 데이터셋 폴더 구조 (UBFC-rPPG DATASET_2 가정)

```
UBFC_ROOT/
    subject1/
        vid.avi
        ground_truth.txt   (1행: PPG 파형, 2행: HR bpm, 3행: timestamp 초)
    subject2/
        vid.avi
        ground_truth.txt
    ...
```

`UBFC_ROOT` 바로 밑에 `subject*` 폴더가 나란히 있어야 함 (한 단계만 탐색).

## 라벨(label)에 대해 — 반드시 읽어주세요

**UBFC-rPPG DATASET_2 전체 42명은 영상 시작부터 끝(약 2분)까지 계속 시간제한
수학 게임을 하는 단일 조건**이라, 영상 내에 안정/과부하 구간이 나뉘어 있지 않고
공식적인 인지 부하 라벨도 없습니다. 그래서 기본적으로는 아래 "인지 부하 점수"
방식(라벨 불필요)을 사용합니다.

만약 시선·동공·눈깜빡임 팀에서 별도로 세션 단위 과부하 라벨을 만들어준다면,
아래 포맷의 `labels.csv`로 분류기(`cognitive_load_pipeline.py`) 경로도 쓸 수 있습니다:

```csv
subject,start_sec,end_sec,label
subject1,0,60,0
subject1,60,180,1
```
- `label = 0`: baseline(저부하/안정), `label = 1`: task(고부하)

## 1) 인지 부하 "점수" 방식 (라벨 없이, 기본 추천)

```bash
python cognitive_load_pipeline.py --data_root ./UBFC_ROOT --out_dir ./results
```
→ `results/hrv_features_all_subjects.csv` 생성

```bash
python cognitive_load_score.py \
    --features ./results/hrv_features_all_subjects.csv \
    --out_dir ./results \
    --mode fit
```
생성물:
- `reference_stats.json` — 각 특징의 population mean/std, 방향(+/-), 가중치. **팀 공용 영상에 나중에 적용할 때 그대로 재사용해야** 점수 기준이 흔들리지 않음
- `cognitive_load_scores.csv` — 윈도우별 `cognitive_load_score`(0~100)와 subject별 평균

나중에 팀원과 같은 영상으로 실전 적용할 때 (같은 기준으로 점수만 매기고 싶을 때):
```bash
python cognitive_load_score.py \
    --features ./results/team_video_features.csv \
    --out_dir ./results \
    --mode apply \
    --reference ./results/reference_stats.json
```

### 점수 계산에 쓰는 특징과 방향

| 특징 | 방향 | 가중치 | 근거 |
|---|---|---|---|
| mean_hr | ↑ = 과부하 | 1.0 | 교감신경 활성화 시 심박 증가 |
| sdnn | ↓ = 과부하 | 1.0 | 전반적 HRV 감소 |
| rmssd | ↓ = 과부하 | 1.5 | 부교감 활성도, 급성 스트레스에 민감 |
| pnn50 | ↓ = 과부하 | 1.0 | 부교감 활성도 |
| lf_hf_ratio | ↑ = 과부하 | 1.5 | 교감/부교감 균형, 가장 직접적 지표 |
| hf_power | ↓ = 과부하 | 1.0 | 부교감 활성도 |

population 기준 z-score 표준화 → 부호 있는 가중합 → sigmoid로 0~100 스케일 매핑.
가중치는 `cognitive_load_score.py`의 `FEATURE_DIRECTIONS` 딕셔너리에서 직접 조정 가능.

## 2) 분류기(이진) 방식 — 라벨이 생기면

```bash
python cognitive_load_pipeline.py \
    --data_root ./UBFC_ROOT \
    --labels ./labels.csv \
    --out_dir ./results
```
Subject 간 개인차(피부톤, 얼굴 각도, baseline 심박수 등)가 커서, 랜덤 K-fold 대신
**Leave-One-Subject-Out (LOSO)**로 평가 (한 명씩 빼서 테스트).

## 3) rPPG 신호 정확도 검증

`ground_truth.txt`의 실측 HR과, 우리가 rPPG로 추정한 HR을 **윈도우 단위(기본 10초,
5초 슬라이딩)로 짝지어** 비교합니다 (영상 전체 평균 1개끼리 비교하면 시간에 따른
변화 패턴을 놓칠 수 있어 윈도우 단위로 검증).

```bash
# 먼저 ground_truth.txt 구조 확인 (최초 1회)
python validate_rppg_accuracy.py --data_root ./UBFC_ROOT --preview subject1

# 전체 subject 윈도우 단위 검증
python validate_rppg_accuracy.py --data_root ./UBFC_ROOT --out_dir ./results
```
출력: MAE, RMSE, Pearson 상관계수, Bland-Altman bias/95% 일치 한계, subject별 MAE 순위.
결과는 `results/rppg_accuracy_validation_windowed.csv`에 저장.

**알려진 이슈 (발견 및 수정 완료):** 심박수가 낮은 subject일수록 dicrotic notch(이완기
반사파)가 별도 피크로 이중 카운팅되어 실제 HR의 최대 2배까지 과대 추정되는 문제가
있었음 (예: 실제 60.2bpm → 추정 126.4bpm). `hrv_features.py`의 `detect_peaks()`에
FFT(Welch) 기반 사전 HR 추정으로 피크 탐색 범위를 제한하고, `prominence` 기준을
추가해 작은 notch 피크를 걸러내도록 수정함.

## 4) 결과 그래프 생성

```bash
python generate_result_figures.py \
    --validation_csv ./results/rppg_accuracy_validation_windowed.csv \
    --scores_csv ./results/cognitive_load_scores.csv \
    --out_dir ./results/figures \
    --example_video ./UBFC_ROOT/subject1/vid.avi
```
생성되는 그래프: HR 산점도, Bland-Altman plot, 인지 부하 점수 분포 히스토그램,
subject별 평균 점수 막대그래프, (선택) rPPG 파형+검출된 피크 예시. 그래프 텍스트는
서버 환경 한글 폰트 미설치 문제를 피하기 위해 전부 영어로 작성됨.

## 사용한 특징 (feature) 목록과 근거

| 특징 | 의미 | 인지 부하 시 경향 |
|---|---|---|
| mean_hr | 평균 심박수 | 증가 |
| sdnn | RR interval 표준편차 (전반적 HRV) | 감소 |
| rmssd | 연속 RR interval 차이의 RMS (부교감 활성도) | 감소 |
| pnn50 | 연속 RR 차이가 50ms 넘는 비율 | 감소 |
| lf_power / hf_power | 저주파/고주파 HRV 파워 | - |
| lf_hf_ratio | 교감/부교감 균형 지표 | 증가 |
| signal_std/skew/kurtosis | rPPG 파형 형태(모션·노이즈 proxy) | 참고용 |

## 알려진 한계 / 다음 단계로 고려할 점

- 영상 압축, 조명 변화, 머리 움직임이 있으면 rPPG 품질이 크게 떨어짐 → 필요시
  CHROM, PBV 등 다른 rPPG 알고리즘과 앙상블하거나 신호 품질 지표로 저품질 윈도우 제외
- 인지 부하 점수의 Feature별 가중치가 데이터 기반 최적화가 아닌 문헌 기반 임의
  설정값 → 추후 PCA 등으로 데이터 기반 가중치 산출 고려
- 현재는 특징 기반(HRV) 분석. 시퀀스 정보(파형 자체의 시계열 패턴)를 더 살리고
  싶다면 1D-CNN/LSTM으로 raw rPPG 파형을 직접 입력하는 방식도 고려 가능
- 최종적으로 시선/동공/눈깜빡임 특징과 late fusion(또는 feature-level fusion)해서
  "거짓말 여부" 2차 분류로 확장하는 구조를 염두에 두고, 이 코드의 출력(윈도우별 특징
  csv, 0~100 점수 스케일)을 다른 모달리티와 같은 타임스탬프 기준으로 맞춰두면 나중에
  합치기 쉬움