# GoodGazeGuard

웹캠으로 수집한 **동공·시선·눈 깜빡임** 신호를 바탕으로 사용자의 인지 부하(Cognitive Load)를 추정하는 프로젝트입니다. 숫자 기억 과제(Digit Span)를 수행하는 동안 얻은 생체·행동 신호를 분석하고, 이를 통합한 LightGBM 모델로 `Low` / `Medium` / `High` 수준을 예측합니다.

> 이 프로젝트의 예측값은 연구·시연 목적의 보조 지표이며, 의학적 진단이나 개인의 능력 판단 용도로 사용할 수 없습니다.

## 프로젝트 목표

- 시선(Gaze), 동공(Pupil), 눈 깜빡임(Blink), rPPG 기반 심박변이도(HRV)에서 인지 부하 관련 특징을 추출합니다.
- OpenNeuro `ds003838` 데이터셋을 이용해 특징별 분석과 멀티모달 결합 모델을 검토합니다.
- 사용자별 동공 기준값을 보정하고, 웹캠 기반 Streamlit 데모에서 과제 수행 중 인지 부하 예측 결과를 확인합니다.

## 주요 기능

- 웹캠·MediaPipe Face Landmarker를 이용한 얼굴 랜드마크 및 동공/시선/눈 깜빡임 신호 수집
- 5·9·13 digit 과제를 기준으로 한 개인별 동공 보정
- gaze + pupil + blink 특징을 결합한 3단계 LightGBM 분류
- Streamlit 기반 과제 진행, 간이 NASA-TLX 설문, 결과 시각화
- rPPG 영상에서 POS 알고리즘과 HRV 특징을 이용한 인지 부하 점수 산출 실험

## 저장소 구조

```text
GoodGazeGuard/
├── streamlit_demo/  # 실행 가능한 웹 시연 앱(기본 실행 대상)
├── webcamTest/      # 웹캠·개인별 pupil calibration 단독 검증 도구
├── branch1/         # pupil + gaze + blink 결합 데이터셋 및 LightGBM 학습 코드
├── Gaze/            # 시선 특징 추출·전처리·분석 코드
├── pupil/           # 동공 반응 특징 추출 및 부하 점수/모델 분석 코드
├── Blink/           # 눈 깜빡임 특징 추출 파이프라인
├── rPPG/            # 얼굴 영상 기반 rPPG·HRV 분석 실험 코드
├── .venv/           # 로컬 Python 가상환경(공유/배포 대상 아님)
└── .venv-webcam/    # 웹캠 테스트용 로컬 가상환경(공유/배포 대상 아님)
```

| 폴더 | 설명 |
| --- | --- |
| `streamlit_demo/` | 프로젝트의 통합 시연 화면입니다. 웹캠 측정, 숫자 과제, NASA-TLX 응답, LightGBM 예측 결과 표시를 담당합니다. |
| `webcamTest/` | Streamlit UI 없이 카메라, 얼굴 랜드마크, trial 기록, 동공 보정이 정상 동작하는지 확인하는 도구입니다. |
| `branch1/` | pupil·gaze·blink trial 특징을 병합하고 LightGBM 분류 모델을 학습/평가/시각화합니다. 현재 데모에서 사용하는 모델 계열의 학습 소스와 데이터가 있습니다. |
| `Gaze/` | eye-tracking 원본 데이터를 trial 단위로 나누고, fixation·속도·분산·scanpath 등 시선 특징을 추출합니다. |
| `pupil/` | 동공 크기 반응을 전처리하고, 개인 기준 정규화·Random Forest 기반 분석·결과 그래프를 생성합니다. |
| `Blink/` | pupil 데이터 안의 blink 플래그를 이용해 blink 횟수, 간격, 엔트로피 등의 특징을 계산합니다. |
| `rPPG/` | 얼굴 영상에서 rPPG 신호를 뽑아 심박과 HRV 특징을 계산하고 인지 부하 점수를 실험합니다. 별도 UBFC-rPPG 데이터셋이 필요합니다. |

## 빠른 시작 — 웹 시연 실행

### 1. 요구 사항

- Python 3.10 이상 권장
- 웹캠
- 카메라 접근 권한이 허용된 브라우저

저장소 최상위 폴더에서 가상환경을 만들고 의존성을 설치합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\streamlit_demo\requirements.txt
```

### 2. 모델 파일 확인

`streamlit_demo/model/`에 다음 파일이 있어야 웹캠 기반 기능을 사용할 수 있습니다.

```text
streamlit_demo/model/
├── branch1_lightgbm_personalized.txt
└── face_landmarker.task
```

현재 작업 폴더에는 포함되어 있을 수 있지만, 대용량/모델 파일 정책에 따라 Git에 항상 추적되지는 않습니다. 파일이 없다면 프로젝트 관리자에게 받거나, 학습 모델을 `branch1/`에서 생성해 해당 위치에 복사해야 합니다.

### 3. 앱 실행

```powershell
cd .\streamlit_demo
streamlit run app.py
```

브라우저가 열리면 시연 모드 또는 정식 모드를 선택하고 안내에 따라 과제를 진행합니다. 카메라가 열리지 않으면 브라우저의 카메라 권한과 다른 앱의 카메라 사용 여부를 확인하세요.

### 4. 최초 사용 전 동공 보정(권장)

새 사용자에게는 개인별 pupil 기준값이 필요합니다. 별도 PowerShell 창에서 아래를 실행합니다.

```powershell
cd ..
.\.venv\Scripts\python.exe .\streamlit_demo\webcam_trial.py
```

화면 안내에 따라 `c`로 보정을 시작한 뒤, 5·9·13 digit trial을 수행합니다. 완료되면 `streamlit_demo/output/pupil_reference.json`이 생성됩니다. 보정 파일이 없으면 앱은 제한적인 대체 처리로 동작할 수 있으므로, 실제 시연 전에는 보정을 완료하는 것을 권장합니다.

## 개발·분석 모듈 실행

각 모듈은 독립적인 실험 코드입니다. 원본 데이터 경로 및 출력 경로가 코드/설정에 포함된 경우가 있으므로, 실행 전 해당 모듈의 `README.md`와 스크립트 상단 설정을 확인하세요.

### 결합 모델 학습 (`branch1`)

```powershell
python -m pip install lightgbm pandas numpy scikit-learn
python .\branch1\src\train_branch1_lightgbm.py `
  --merged .\branch1\dataset\gaze_pupil_blink_merged.csv `
  --out .\branch1\output\branch1_merged_test `
  --clean-signal
```

`--clean-signal`은 과제 길이와 직접적으로 연결될 수 있는 특징을 제외해 데이터 누수를 줄이는 옵션입니다.

### 동공 분석 (`pupil`)

```powershell
cd .\pupil
python -m pip install -r requirements.txt
python .\src\pupil_baseline_pipeline.py
python .\src\evaluate_model_performance.py
python .\src\generate_report_figures.py
```

원본 OpenNeuro 데이터를 새로 받는 경우에는 먼저 `download_and_reduce.py`와 `download_beh_all.py`를 실행합니다.

### 시선 분석 (`Gaze`)

```powershell
cd .\Gaze
python -m pip install -r requirements.txt
# 데이터 경로를 설정한 뒤 src/, experiments/의 분석 스크립트를 실행합니다.
```

### 눈 깜빡임 특징 추출 (`Blink`)

```powershell
cd .\Blink
python -m pip install -r requirements.txt
python .\run_pipeline.py
```

OpenNeuro 데이터셋의 로컬 경로는 실행 전 설정해야 합니다.

### rPPG 실험 (`rPPG`)

```powershell
cd .\rPPG
python -m pip install opencv-python mediapipe numpy scipy pandas scikit-learn joblib matplotlib
python .\cognitive_load_pipeline.py --data_root <UBFC_ROOT> --out_dir .\results
python .\cognitive_load_score.py --features .\results\hrv_features_all_subjects.csv --out_dir .\results --mode fit
```

`<UBFC_ROOT>`에는 `subject*/vid.avi` 구조의 UBFC-rPPG DATASET_2 경로를 넣습니다.

## 데이터와 산출물

- 분석 데이터의 주요 출처는 [OpenNeuro ds003838](https://openneuro.org/datasets/ds003838)입니다.
- `branch1/dataset/`에는 결합 학습에 사용하는 CSV가, `branch1/output/`에는 학습 결과와 특징 중요도가 있습니다.
- 각 모듈의 `output/`, `outputs/`, `results/`에는 실행 결과가 생성됩니다. 개인별 보정값, 카메라 기록, 원본 대용량 데이터는 공유 저장소에 올리지 않는 것을 권장합니다.

## 기술 스택

Python, Streamlit, LightGBM, pandas, NumPy, SciPy, scikit-learn, OpenCV, MediaPipe, streamlit-webrtc, Matplotlib

## 알려진 제한 사항

- 웹캠에서 얻는 동공·시선·blink 값은 연구용 eye tracker 데이터와 측정 방식이 다르므로, 실제 환경에서 별도 검증이 필요합니다.
- 조명, 얼굴 각도, 안경, 움직임, 카메라 품질은 신호 품질과 예측 결과에 영향을 줍니다.
- rPPG 모듈은 독립 실험 단계이며, 현재 웹 데모의 실시간 예측 파이프라인에는 통합되어 있지 않습니다.
- 모델의 인지 부하 라벨은 과제 난이도와 행동 결과를 바탕으로 구성한 연구용 지표입니다.

## 참고

- OpenNeuro ds003838: <https://openneuro.org/datasets/ds003838>
- UBFC-rPPG Dataset: rPPG 모듈 실행 시 별도로 준비해야 합니다.
