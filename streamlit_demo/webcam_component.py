"""
webcam_component.py

웹캠 담당자가 만든 코드(config/signal_processing/calibration/trial_recorder/
extract_features/model_runtime)를 그대로 재사용해서, 원래 cv2 창(cv2.imshow)
방식이었던 부분만 Streamlit 안에서 실시간으로 보이게 streamlit-webrtc로 새로
짠 파일. 웹캠 담당자의 원본 5개 파일(config.py, signal_processing.py,
calibration.py, trial_recorder.py, extract_features.py, model_runtime.py)은
전혀 수정하지 않고 그대로 import해서 씀.

이 파일이 새로 하는 일
------------------------------------------------
- MediaPipe Face Landmarker + TrialRecorder + BlinkDetector를 웹캠 프레임마다
  실행하는 걸 streamlit-webrtc의 VideoProcessorBase 콜백 안에서 수행
  (원래 webcam_trial.py의 while 루프 안 로직과 거의 동일, cv2.imshow 대신
  Streamlit 컴포넌트가 화면에 그려줌)
- 버튼 클릭(메인 스레드)으로 trial 시작/종료를 제어할 수 있게 스레드 안전
  (thread-safe) 메서드 제공 -- 영상 처리는 별도 스레드에서 돌기 때문에,
  self.lock으로 감싸서 두 스레드가 동시에 같은 데이터를 건드리지 않게 함

전제 조건
------------------------------------------------
- pupil 개인화 calibration은 이 파일이 아니라 webcam_trial.py(원본 cv2 도구)로
  미리 한 번 끝내둬야 함 (output/pupil_reference.json 파일이 있어야 pupil
  feature가 정상적으로 나옴). 이 파일은 "이미 calibration이 끝났다"고 가정하고
  측정(measuring) 단계만 담당함.
- 실제 과제(문제) 파트가 아직 없어서, "다음 숫자 기록" 버튼으로 digit onset을
  임시로 흉내냄 -- 실제 과제 코드가 오면 mark_digit() 호출을 그 쪽 타이밍으로
  교체하면 됨.
"""

import threading
import time

import av
import cv2
import mediapipe as mp
from streamlit_webrtc import VideoProcessorBase

from config import MODEL_PATH as MP_LANDMARKER_PATH
from signal_processing import BlinkDetector, extract_frame_metrics
from trial_recorder import TrialRecorder


def _clamp_point(point, width, height):
    """cv2.circle 등에 넘기기 전에 좌표를 화면 범위 안으로 눌러 담음.
    빠른 머리 움직임으로 landmark가 화면 밖으로 크게 벗어나면 (NaN 포함) 이후
    .astype(int) 값이 비정상적으로 커지거나 깨져서 cv2 그리기 함수가 예외를
    던질 수 있어서, 그리기 직전에 안전한 범위로 clamp함."""
    x, y = float(point[0]), float(point[1])
    if not (x == x) or not (y == y):  # NaN 체크 (NaN != NaN)
        x, y = 0.0, 0.0
    x = int(min(max(x, 0), width - 1))
    y = int(min(max(y, 0), height - 1))
    return x, y


class GazeGuardVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.recorder = TrialRecorder()
        self.blink_detector = BlinkDetector()
        self.session_start = time.monotonic()
        self.last_video_timestamp_ms = -1
        self.latest_metrics = None  # 화면 오버레이/디버깅용 최근 프레임 지표
        self.landmarker = self._create_landmarker()

    @staticmethod
    def _create_landmarker():
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MP_LANDMARKER_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------
    # streamlit-webrtc가 프레임마다 자동으로 호출하는 콜백 (별도 스레드에서 실행됨)
    # ------------------------------------------------------------
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # 거울처럼 좌우반전
        height, width, _ = img.shape

        # 이 아래는 원래 try/except 없이 쭉 돌던 부분인데, 참가자가 고개를
        # 빨리/크게 움직일 때 홍채 landmark 좌표가 화면 밖으로 크게 벗어나면서
        # cv2.circle에 비정상적인 좌표가 들어가 예외가 나고, 그게 recv() 밖으로
        # 새 나가면서 웹캠 트랙 자체가 죽는(화면이 꺼지는) 문제가 있었음.
        # 프레임 하나 처리에 실패해도 웹캠 스트림 자체는 절대 안 죽게, 전체를
        # try/except로 감싸고 실패하면 오버레이 없는 원본 프레임만 반환함.
        try:
            timestamp = time.monotonic()
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            video_timestamp_ms = max(
                int((timestamp - self.session_start) * 1000), self.last_video_timestamp_ms + 1
            )
            self.last_video_timestamp_ms = video_timestamp_ms
            result = self.landmarker.detect_for_video(mp_image, video_timestamp_ms)

            with self.lock:
                self.recorder.note_frame(bool(result.face_landmarks))
                if result.face_landmarks:
                    metrics = extract_frame_metrics(result.face_landmarks[0], width, height)
                    self.recorder.add_baseline_sample(timestamp, metrics["pupil_diameter_px"])

                    if self.recorder.recording:
                        metrics["blink_flag"] = self.blink_detector.update(timestamp, metrics["blink_ratio"])
                    else:
                        self.blink_detector.add_open_eye_sample(timestamp, metrics["blink_ratio"])
                        metrics["blink_flag"] = 0

                    rel_pupil = self.recorder.record_frame(timestamp, metrics)
                    self.latest_metrics = {**metrics, "rel_pupil": rel_pupil}

                    # 빠른 움직임 등으로 landmark가 화면 밖으로 크게 벗어나면
                    # (음수거나 width/height를 훌쩍 넘는 좌표) cv2.circle이
                    # 터질 수 있어서, 그리기 전에 화면 범위 안으로 clamp함.
                    left = _clamp_point(metrics["left_iris"], width, height)
                    right = _clamp_point(metrics["right_iris"], width, height)
                    cv2.circle(img, left, 4, (0, 255, 255), -1)
                    cv2.circle(img, right, 4, (0, 255, 255), -1)
                    cv2.putText(img, f"blink EAR: {metrics['blink_ratio']:.3f}", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    if self.recorder.recording:
                        cv2.putText(img, "RECORDING", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)
                        if rel_pupil is not None:
                            cv2.putText(img, f"rel pupil: {rel_pupil:.2f}px", (20, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    self.latest_metrics = None
                    cv2.putText(img, "얼굴이 안 보여요", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)
        except Exception as e:
            # 프레임 하나 처리 실패 -- 로그만 남기고 이번 프레임은 오버레이
            # 없이 그냥 흘려보냄 (여기서 예외가 새 나가면 웹캠 스트림 전체가 죽음)
            print(f"[GazeGuardVideoProcessor] frame 처리 실패, 이번 프레임은 건너뜀: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # ------------------------------------------------------------
    # 메인 스레드(Streamlit 버튼 클릭)에서 호출하는 제어 메서드들
    # 전부 self.lock으로 감싸서 recv()랑 동시에 데이터 안 건드리게 함
    # ------------------------------------------------------------
    def start_trial(self):
        with self.lock:
            baseline = self.recorder.start_trial()
            blink_threshold = self.blink_detector.freeze_threshold()
        return baseline, blink_threshold

    def mark_digit(self, digit_index: int):
        with self.lock:
            self.recorder.mark_digit(digit_index, time.monotonic())

    def end_trial(self):
        with self.lock:
            raw_df = self.recorder.end_trial()
            quality = self.recorder.quality_summary()
        return raw_df, quality

    def is_recording(self) -> bool:
        with self.lock:
            return self.recorder.recording

    def current_digit_index(self):
        with self.lock:
            return self.recorder.current_digit_index
