# GazeGuard Streamlit 데모

웹캠으로 pupil/gaze/blink를 실시간 측정하면서 질문에 답변 → 난이도(블록)별로 Branch1 모델 예측을 보여주는 데모 앱.

## 실행 방법

```
cd streamlit_demo
pip install -r requirements.txt
python -m streamlit run app.py
```

## 처음 실행 전에 꼭 해야 하는 것 (사람마다 한 번씩)

1. **개인 pupil calibration 하기**
   ```
   python webcam_trial.py
   ```
   실행 후 안내(`c`→calibration 시작, `s`→trial 시작, `d`→답변마다 기록, `e`→trial 종료)에 따라 5자리/9자리/13자리 총 3번 진행. 끝나면 `output/pupil_reference.json`이 생성됨 (이게 있어야 실제 모델 예측이 나옴, 없으면 예측 실패하고 더미 결과로 대체됨).

## 폴더 구조 / 필요한 파일

`streamlit_demo` 폴더 하나만 있으면 돌아감 (branch1 모델 파일 복사본을
`model/` 안에 같이 넣어뒀음 -- `model_runtime.py`가 이 안에서 먼저 찾고,
없을 때만 예전처럼 `../branch1/dataset/`를 찾음):

```
streamlit_demo/
├── app.py                  # 메인 화면 (시작 → 측정 → 결과)
├── task_data.py            # 질문 30개(난이도별 10개) + 순서 그룹
│                           #   ※ 질문 내용 바꿀 땐 여기만 수정하면 됨
├── task_component.py       # 질문 진행 로직 (준비/답변/휴식,
│                           #   블록별 NASA-TLX, 블록별 trial/예측)
├── webcam_component.py     # 웹캠 실시간 연동 (streamlit-webrtc)
├── model_runtime.py        # Branch1 모델 로딩/예측
├── config.py                )
├── signal_processing.py     )
├── calibration.py           )  웹캠 담당자 원본 코드
├── trial_recorder.py        )  (수정 없이 그대로 사용)
├── extract_features.py      )
├── webcam_trial.py          )  ← 개인 calibration용 CLI 도구
├── requirements.txt
│
├── model/
│   ├── branch1_lightgbm_personalized.txt   # Branch1 모델 (레포에 포함됨)
│   └── face_landmarker.task                # MediaPipe 얼굴 인식 모델 (레포에 포함됨)
└── output/
    └── pupil_reference.json   # [각자 calibration 후 자동 생성 -- 레포엔 없음]
```

대괄호(`[ ]`) 표시된 파일만 레포에 없는 게 정상 — "처음 실행 전에 꼭 해야
하는 것" 단계(calibration)를 각자 밟으면 로컬에 자동으로 생김.

## 업로드/공유 시 제외할 것

- `output/` 폴더 (개인 calibration 파일, 실행 로그 — 각자 본인 걸로 새로 생성해야 함)
- `__pycache__/`, `.model_cache/`
- `model/` 안의 두 파일(`branch1_lightgbm_personalized.txt`, `face_landmarker.task`)은
  반대로 **꼭 포함**해야 함 — 둘 다 용량이 작아서(각각 1.4MB/3.6MB) 레포에 같이
  올려도 되고, 그래야 다운로드하는 사람이 별도 링크 없이 한 번에 다 받을 수 있음
