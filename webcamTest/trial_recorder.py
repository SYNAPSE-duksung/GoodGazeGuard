from collections import deque

import numpy as np
import pandas as pd

from config import MIN_VALID_FRAME_RATIO, MIN_VALID_FRAMES, PRETRIAL_BASELINE_SECONDS


class TrialRecorder:
    """시행(trial) 상태를 관리하고 extract_features.py에서 쓸 원본 스키마를 생성한다."""

    def __init__(self):
        self.pretrial_buffer = deque()
        self.recording = False
        self.records = []
        self.baseline_px = None
        self.trial_position = 0
        self.current_digit_index = None
        self.current_digit_shown_at = None
        self.total_trial_frames = 0
        self.valid_trial_frames = 0

    # 대기 중 샘플 추가 -> 최근 PRETRIAL_BASELINE_SECONDS 동안의 샘플만 유지
    def add_baseline_sample(self, timestamp, pupil_diameter_px):
        self.pretrial_buffer.append({"timestamp": timestamp, "pupil_diameter_px": pupil_diameter_px})
        while self.pretrial_buffer and timestamp - self.pretrial_buffer[0]["timestamp"] > PRETRIAL_BASELINE_SECONDS:
            self.pretrial_buffer.popleft()

    # trial 시작 시점에 baseline 계산 및 trial 기록 초기화
    def start_trial(self):
        if not self.pretrial_buffer:
            raise ValueError("baseline 계산에 사용할 사전 시행(pretrial) 동공 샘플이 없습니다.")
        self.baseline_px = float(np.mean([sample["pupil_diameter_px"] for sample in self.pretrial_buffer]))
        self.recording = True
        self.records = []
        self.trial_position = 0
        self.current_digit_index = None
        self.current_digit_shown_at = None
        self.total_trial_frames = 0
        self.valid_trial_frames = 0
        return self.baseline_px

    # trial 기록 중 face landmark coverage 추적
    def note_frame(self, face_detected):
        """시행이 기록되고 있는 동안에만 얼굴 랜드마크 인식률(coverage)을 추적한다."""
        if self.recording:
            self.total_trial_frames += 1
            self.valid_trial_frames += int(face_detected)

    # 현제 제시한 digit의 index와 onset timestamp 기록
    def mark_digit(self, digit_index, shown_at):
        if not self.recording:
            raise ValueError("시행이 시작되기 전에는 digit를 표시(mark)할 수 없습니다.")
        self.current_digit_index = int(digit_index)
        self.current_digit_shown_at = shown_at

    # trial 기록에 현재 프레임의 metrics를 추가하고, baseline 대비 상대 pupil diameter 계산
    def record_frame(self, timestamp, frame_metrics):
        if not self.recording:
            return None
        self.trial_position += 1
        rel_pupil = frame_metrics["pupil_diameter_px"] - self.baseline_px
        self.records.append({"timestamp": timestamp, "position": self.trial_position, "digit_index": self.current_digit_index, "digit_shown_at": self.current_digit_shown_at, "gaze_x": frame_metrics["gaze_x"], "gaze_y": frame_metrics["gaze_y"], "pupil_diameter_px": frame_metrics["pupil_diameter_px"], "baseline_px": self.baseline_px, "rel_pupil": rel_pupil, "blink_ratio": frame_metrics["blink_ratio"], "blink_flag": frame_metrics["blink_flag"]})
        return rel_pupil

    # trial 종료 시점에 trial 기록을 DataFrame으로 반환
    def end_trial(self):
        self.recording = False
        raw_df = pd.DataFrame(self.records)
        quality = self.quality_summary()
        if not raw_df.empty:
            for name, value in quality.items():
                raw_df[name] = value
        return raw_df

    # trial 기록의 quality summary 반환
    def quality_summary(self):
        ratio = self.valid_trial_frames / self.total_trial_frames if self.total_trial_frames else 0.0
        accepted = self.valid_trial_frames >= MIN_VALID_FRAMES and ratio >= MIN_VALID_FRAME_RATIO
        return {
            "total_frames": self.total_trial_frames,
            "valid_frames": self.valid_trial_frames,
            "valid_frame_ratio": ratio,
            "quality_accepted": accepted,
        }
