# 웹캠 calibration 진행 상황

## 현재 목적

이 폴더는 웹캠 영상, MediaPipe 얼굴 랜드마크, 수동 trial 기록, pupil calibration이
정상 동작하는지 확인하기 위한 코드입니다. 실제 테스트 문제와 실시간 Branch1 예측은
아직 연결하지 않았습니다.

임시 숫자 과제 화면과 자동 예측은 실제 과제 규격을 받은 뒤 연결하기 위해 현재 실행
흐름에서 제외했습니다. 대신 실제 과제 코드가 digit 표시 시점을 전달할 수 있도록
기록 인터페이스를 유지합니다.

## 실행 방법

기존 `.venv`가 아닌 프로젝트 전용 `.venv-webcam` 환경으로 실행합니다.

```powershell
.\.venv-webcam\Scripts\python.exe .\webcamTest\webcam_trial.py
```

필요한 패키지는 `requirements.txt`에 고정되어 있습니다. 환경을 새로 만들 때는 다음을
사용합니다.

```powershell
python -m venv .venv-webcam
.\.venv-webcam\Scripts\python.exe -m pip install -r .\webcamTest\requirements.txt
```

`webcamTest/model/face_landmarker.task`는 MediaPipe 모델 파일이며 Git에 포함하지
않습니다. 실행 전에 팀 공유 경로에서 받아 해당 위치에 넣어야 합니다.

## 수동 pupil calibration 절차

키 입력은 반드시 웹캠 미리보기 창에 포커스를 둔 상태에서 합니다.

1. 얼굴이 웹캠에 잘 보이도록 한 뒤 약 2초간 정면을 바라봅니다.
2. `c`를 한 번 눌러 calibration을 시작합니다.
3. 5-digit trial: `s`를 누르고, 각 digit이 표시되는 순간마다 `d`를 총 5번 누른 뒤 `e`를 한 번 누릅니다.
4. 같은 방식으로 9-digit (`d` 9번), 13-digit (`d` 13번) trial을 진행합니다.

`e`는 프로그램 종료가 아니라 현재 trial만 끝냅니다. 프로그램 종료는 `q`입니다.
각 `d` 이후 1초 이내에 기록된 유효 pupil 표본 중 가장 가까운 값을 해당 digit 표본으로
사용합니다. 따라서 `d`는 연속해서 누르지 말고 digit 표시 간격에 맞춰 눌러야 합니다.

## 생성 파일

실행 결과는 모두 `webcamTest/output/`에 저장됩니다.

- `trial_raw.csv`: 가장 최근 trial의 프레임별 원시 기록
- `pupil_reference.json`: 개인 pupil z-score용 평균, 표준편차, 표본 수
- `gaze_screen_calibration.json`, `gaze_blink_reference.json`: 이후 확장 기능에서 생성될 수 있는 보정 파일

`pupil_reference.json`은 `n_digit_samples >= 27`, 유한한 `mean`, 0보다 큰 `std`이면 정상입니다.
예를 들어 표본이 32개이고 `std`가 1.68이면 calibration이 성공한 것입니다.

## 현재 검증된 항목

- MediaPipe Face Landmarker로 웹캠과 얼굴 랜드마크를 읽음
- VIDEO 모드 timestamp를 항상 증가시켜 `Input timestamp must be monotonically increasing` 오류 방지
- trial 원시 CSV 저장
- digit onset pupil 표본 수집 및 `pupil_reference.json` 저장
- Branch1 개인화 모델의 41개 feature 이름과 webcam feature 순서 일치 확인
- LightGBM 모델의 CRLF 줄바꿈 문제 방지 및 모델 로드/테스트 예측 확인

## 주의 및 다음 단계

- 웹캠 pupil 값은 MediaPipe 홍채 랜드마크의 px 거리 기반 대리값이며, 학습 데이터의 `diameter_3d`와 다릅니다.
- 웹캠 gaze는 카메라 영상 속 홍채 위치이고, 학습 gaze는 화면 정규화 좌표입니다.
- webcam blink는 EAR 기준이며 학습 데이터의 blink 신호와 다릅니다.

따라서 현재 코드는 기록과 연결 흐름을 검증하는 단계입니다. 실제 과제가 도착하면 과제 코드는
아래 호출만 정확한 표시 시점에 연결하면 됩니다.

```python
recorder.start_trial()                 # trial 시작
recorder.mark_digit(index, onset)      # 각 digit 표시 직후
raw_df = recorder.end_trial()          # trial 종료
```

그다음 `extract_features.py`가 `pupil_reference.json`을 읽어 digit pupil 값을 z-score로
변환하고 Branch1 입력 feature를 만들 수 있습니다.
