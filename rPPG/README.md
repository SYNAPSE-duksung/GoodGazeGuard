# rPPG 기반 인지 과부하 분류 파이프라인 (UBFC-rPPG)

시선/동공/눈깜빡임 팀과 함께 진행하는 "인지 과부하 → 거짓말 판별" 프로젝트의
rPPG 파트 코드입니다.

## 전체 흐름

```
얼굴 영상 (vid.avi)
    │  rppg_pos.py
    ▼
rPPG 파형 (POS 알고리즘, bandpass filtered)
    │  hrv_features.py
    ▼
윈도우별 HRV 특징 (SDNN, RMSSD, pNN50, LF/HF ratio, ...)
    │  cognitive_load_pipeline.py
    ▼
RandomForest 분류기 (Leave-One-Subject-Out 평가)
```

## 파일 구성

| 파일 | 역할 |
|---|---|
| `rppg_pos.py` | 얼굴 ROI 검출(mediapipe FaceMesh 또는 Haar cascade) + POS 알고리즘으로 rPPG 신호 추출 |
| `hrv_features.py` | rPPG 신호에서 피크 검출 → RR interval → HRV 시간/주파수 영역 특징 계산 |
| `cognitive_load_pipeline.py` | 전체 subject 순회, 특징 테이블 생성, 라벨 매칭, 분류기 학습/평가 |

## 설치

```bash
pip install opencv-python mediapipe numpy scipy pandas scikit-learn joblib
```
(mediapipe 설치가 어려운 환경이면 `--no_mediapipe` 옵션으로 Haar cascade fallback 사용 가능. 정확도는 다소 떨어짐)

## 데이터셋 폴더 구조 (UBFC-rPPG DATASET_2 가정)

```
UBFC_ROOT/
    subject1/
        vid.avi
        ground_truth.txt
    subject2/
        vid.avi
        ground_truth.txt
    ...
```

## 라벨(label)에 대해 — 반드시 읽어주세요

**UBFC-rPPG 자체에는 "인지 과부하" 라벨이 없습니다.** DATASET_2는 촬영 프로토콜상
"안정 상태 → 시간제한 수학 게임(스트레스 유발)" 구조이므로, 이 시간 구간을 근거로
직접 라벨을 만들어야 합니다.

`labels.csv` 형식:

```csv
subject,start_sec,end_sec,label
subject1,0,60,0
subject1,60,180,1
subject2,0,55,0
subject2,55,190,1
```

- `label = 0`: baseline(저부하/안정)
- `label = 1`: task(고부하/스트레스 유발 구간)

시선·동공·눈깜빡임 팀에서 별도로 세션 단위 과부하 라벨을 만들어준다면, 같은 포맷으로
맞춰서 사용하면 됩니다.

## 인지 부하 "점수" 방식 (라벨 없이, 추천)

UBFC-rPPG엔 과부하/정상 이진 라벨이 원래 없기 때문에, 이진 분류기를 억지로
학습시키기보다 **HRV 특징을 조합한 연속 점수(0~100)**를 매기는 방식을 추천합니다.
이 방식은 라벨이 전혀 필요 없고, 나중에 시선/동공/눈깜빡임 팀의 점수와
fusion하기도 쉽습니다 (다들 0~100 스케일로 맞춰두면 됨).

`cognitive_load_score.py`는 `cognitive_load_pipeline.py`가 만든
`hrv_features_all_subjects.csv`를 입력받아 동작합니다.

```bash
# 1) UBFC-rPPG로 reference 통계 산출 + 점수 계산 (최초 1회)
python cognitive_load_score.py \
    --features ./results/hrv_features_all_subjects.csv \
    --out_dir ./results \
    --mode fit
```

이 명령이 만드는 것:
- `reference_stats.json` — 각 특징의 population mean/std, 방향(+/-), 가중치. **팀 공용 영상에 나중에 적용할 때 그대로 재사용해야** 점수 기준이 흔들리지 않습니다.
- `cognitive_load_scores.csv` — 윈도우별 `cognitive_load_score`(0~100)와 subject별 평균 요약

나중에 팀원과 같은 영상으로 실전 적용할 때 (같은 기준으로 점수만 매기고 싶을 때):
```bash
python cognitive_load_score.py \
    --features ./results/team_video_features.csv \
    --out_dir ./results \
    --mode apply \
    --reference ./results/reference_stats.json
```

### 점수 계산 방식

각 특징을 population 기준 z-score로 표준화한 뒤, 생리학적으로 알려진 방향(+1/-1)과
가중치로 합산 → sigmoid로 0~100 스케일 매핑:

| 특징 | 방향 | 가중치 | 근거 |
|---|---|---|---|
| mean_hr | ↑ = 과부하 | 1.0 | 교감신경 활성화 시 심박 증가 |
| sdnn | ↓ = 과부하 | 1.0 | 전반적 HRV 감소 |
| rmssd | ↓ = 과부하 | 1.5 | 부교감 활성도, 급성 스트레스에 민감 |
| pnn50 | ↓ = 과부하 | 1.0 | 부교감 활성도 |
| lf_hf_ratio | ↑ = 과부하 | 1.5 | 교감/부교감 균형, 가장 직접적 지표 |
| hf_power | ↓ = 과부하 | 1.0 | 부교감 활성도 |

가중치는 HRV 문헌에서 급성 스트레스 반응성이 큰 지표(RMSSD, LF/HF ratio)에
더 높게 준 값이며, 필요하면 `cognitive_load_score.py`의 `FEATURE_DIRECTIONS`
딕셔너리에서 직접 조정 가능합니다.

## 분류기(이진) 방식 — 라벨이 생기면

 (라벨 없이 탐색적으로 보고 싶을 때):
```bash
python cognitive_load_pipeline.py --data_root /path/to/UBFC_ROOT --out_dir ./results
```

라벨까지 넣어서 분류기 학습:
```bash
python cognitive_load_pipeline.py \
    --data_root /path/to/UBFC_ROOT \
    --labels /path/to/labels.csv \
    --out_dir ./results
```

## 사용한 특징 (feature) 목록과 근거

| 특징 | 의미 | 인지 과부하 시 경향 |
|---|---|---|
| mean_hr | 평균 심박수 | 증가 |
| sdnn | RR interval 표준편차 (전반적 HRV) | 감소 |
| rmssd | 연속 RR interval 차이의 RMS (부교감 활성도) | 감소 |
| pnn50 | 연속 RR 차이가 50ms 넘는 비율 | 감소 |
| lf_power / hf_power | 저주파/고주파 HRV 파워 | - |
| lf_hf_ratio | 교감/부교감 균형 지표 | 증가 |
| signal_std/skew/kurtosis | rPPG 파형 형태(모션·노이즈 proxy) | 참고용 |

## 평가 방식

Subject 간 개인차(피부톤, 얼굴 각도, baseline 심박수 등)가 커서, 랜덤 K-fold 대신
**Leave-One-Subject-Out (LOSO)** 방식으로 평가합니다. 한 명씩 빼서 테스트하고
나머지로 학습하는 방식이라, 실제로 "본 적 없는 사람"에게 일반화되는지를 봅니다.

## 알려진 한계 / 다음 단계로 고려할 점

- 영상 압축, 조명 변화, 머리 움직임이 있으면 rPPG 품질이 크게 떨어짐 → 필요시
  CHROM, PBV 등 다른 rPPG 알고리즘과 앙상블하거나 신호 품질 지표로 저품질 윈도우 제외
- 현재는 RandomForest 기반 특징 분류. 시퀀스 정보(파형 자체의 시계열 패턴)를 더
  살리고 싶다면 1D-CNN/LSTM으로 raw rPPG 파형을 직접 입력하는 방식도 고려 가능
- 최종적으로 시선/동공/눈깜빡임 특징과 late fusion(또는 feature-level fusion)해서
  "거짓말 여부" 2차 분류로 확장하는 구조를 염두에 두고, 이 코드의 출력(윈도우별 특징
  csv)을 다른 모달리티와 같은 타임스탬프 기준으로 맞춰두면 나중에 합치기 쉬움