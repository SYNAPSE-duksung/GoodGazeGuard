import time

import cv2
import mediapipe as mp
from config import MODEL_PATH, OUTPUT_DIR, PUPIL_CALIBRATION_DIGIT_TARGET
from signal_processing import BlinkDetector, extract_frame_metrics
from calibration import PupilCalibrator
from trial_recorder import TrialRecorder


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam을 연결하지 못했습니다.")

    print("Webcam started")
    print("c: calibration 시작 | s: trial 시작 | d: 다음 digit onset 기록 | e: trial 종료 | q: 종료")

    # 객체 생성
    recorder = TrialRecorder()  # trail 관리
    calibrator = PupilCalibrator()  # calibration 관리
    blink_detector = BlinkDetector()    # 시간에 따른 blink event 처리
    session_start = time.monotonic()    # 세션 시간 기준
    last_video_timestamp_ms = -1

    try:
        # Face Landmarker 객체 생성 및 사용
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True: # 웹캠 프레임 반복
                ok, frame = cap.read()
                if not ok:
                    print("Webcam 프레임을 읽을 수 없습니다.")
                    break

                frame = cv2.flip(frame, 1)  # 좌우반전(거울처럼)
                height, width, _ = frame.shape
                timestamp = time.monotonic()    # 현재 프레임의 시간
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # RGB 프레임을 MediaPipe가 처리할 수 있는 이미지 객체로 변환
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                # MediaPipe Face Landmarker를 사용하여 얼굴 랜드마크 감지
                # VIDEO mode requires strictly increasing millisecond timestamps.
                # A fast webcam can produce multiple frames within the same millisecond.
                video_timestamp_ms = max(int((timestamp - session_start) * 1000), last_video_timestamp_ms + 1)
                last_video_timestamp_ms = video_timestamp_ms
                result = landmarker.detect_for_video(mp_image, video_timestamp_ms)
                recorder.note_frame(bool(result.face_landmarks))    # 기록

                if result.face_landmarks:   # 얼굴이 검출된 경우
                    # 한 프레임의 pupil, gaze, blink 관련 metrics 추출
                    metrics = extract_frame_metrics(result.face_landmarks[0], width, height)
                    recorder.add_baseline_sample(timestamp, metrics["pupil_diameter_px"])   # 현재 pupil diameter를 baseline sample로 추가

                    if recorder.recording:
                        metrics["blink_flag"] = blink_detector.update(timestamp, metrics["blink_ratio"])
                    else:
                        blink_detector.add_open_eye_sample(timestamp, metrics["blink_ratio"])
                        metrics["blink_flag"] = 0   # blink event가 발생하지 않음
                    rel_pupil = recorder.record_frame(timestamp, metrics)   # 현재 프레임의 metrics를 trial 기록에 추가

                    cv2.circle(frame, tuple(metrics["left_iris"].astype(int)), 4, (0, 255, 255), -1)
                    cv2.circle(frame, tuple(metrics["right_iris"].astype(int)), 4, (0, 255, 255), -1)
                    cv2.putText(frame, f"gaze: ({metrics['gaze_x']:.3f}, {metrics['gaze_y']:.3f})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"blink EAR: {metrics['blink_ratio']:.3f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"pupil diameter: {metrics['pupil_diameter_px']:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    if recorder.recording:
                        cv2.putText(frame, "RECORDING", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, f"rel pupil: {rel_pupil:.2f}px", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Webcam Trial", frame)
                key = cv2.waitKey(1) & 0xFF # 키 입력 처리

                if key == ord("s"): # Trial 시작
                    try:
                        baseline = recorder.start_trial()
                        blink_threshold = blink_detector.freeze_threshold() # trial 시작 시점에 BlinkDetector가 calibration 과정에서 얻은 threshold를 고정(freeze)
                        print(f"Trial started | baseline_px={baseline:.3f}, blink_threshold={blink_threshold:.3f}")
                    except ValueError as error:
                        print(f"Error starting trial: {error}")
                elif key == ord("c"):   # pupil calibration 시작
                    calibrator.start()
                    print(f"Calibration started. Complete 5-, 9-, and 13-digit trials ({PUPIL_CALIBRATION_DIGIT_TARGET} samples).")
                elif key == ord("d"):   # 다음 digit 시작
                    next_digit = 1 if recorder.current_digit_index is None else recorder.current_digit_index + 1
                    try:
                        recorder.mark_digit(next_digit, time.monotonic())
                        print(f"Digit {next_digit} onset recorded")
                    except ValueError as error:
                        print(error)
                elif key == ord("e"):   # Trial 종료
                    raw_df = recorder.end_trial()
                    quality = recorder.quality_summary()
                    print(
                        f"Trial ended - {len(raw_df)} valid frames "
                        f"({quality['valid_frame_ratio']:.1%} face coverage)"
                    )
                    if not raw_df.empty:
                        raw_path = OUTPUT_DIR / "trial_raw.csv"
                        raw_df.to_csv(raw_path, index=False)
                        print(f"{raw_path} saved")
                        if not quality["quality_accepted"]:
                            print("Trial quality is below the minimum threshold; calibration sample is skipped.")
                        try:    # 현재 trial의 raw_df를 calibrator에 전달하여 calibration 진행
                            progress = calibrator.add_trial(raw_df)
                            if progress:
                                if progress.get("skipped"):
                                    print(f"Calibration trial skipped: {progress['skipped']}")
                                    continue
                                print(f"Calibration samples: trial={progress['trial_samples']}, total={progress['total_samples']}")
                                if progress["reference"]:
                                    print("Calibration complete: pupil_reference.json saved")
                        except ValueError as error:
                            print(f"Calibration collection failed: {error}")
                elif key == ord("q"):   # 프로그램 종료
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
