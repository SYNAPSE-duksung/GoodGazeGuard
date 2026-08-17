from collections import deque

import numpy as np

from config import (
    BLINK_THRESHOLD,
    BLINK_THRESHOLD_RATIO,
    LEFT_EYE,
    LEFT_IRIS,
    MIN_BLINK_INTERVAL_SECONDS,
    PRETRIAL_BASELINE_SECONDS,
    RIGHT_EYE,
    RIGHT_IRIS,
)

# 두 점 사이의 유클리드 거리. np.linalg.norm으로 벡터 차의 크기(norm)를 계산
def distance(point_a, point_b):
    return np.linalg.norm(np.asarray(point_a) - np.asarray(point_b))

def get_eye_ratio(landmarks, indices, width, height):
    # 눈 주변 6개 landmark index를 받아서 픽셀 좌표로 변환
    points = [np.array([landmarks[index].x * width, landmarks[index].y * height]) for index in indices]
    horizontal = distance(points[0], points[3]) # points[0]과 points[3]은 눈의 좌우 끝(수평 거리)
    if horizontal == 0:
        return 0.0
    return (distance(points[1], points[5]) + distance(points[2], points[4])) / (2 * horizontal)

# 홍채(iris) landmark들의 평균 좌표 -> gaze 추정에 사용
def get_iris_center(landmarks, indices, width, height):
    points = [[landmarks[index].x * width, landmarks[index].y * height] for index in indices]
    return np.mean(points, axis=0)

# 한 프레임의 랜드마크로부터 최종 지표를 뽑아내는 메인 함수
def extract_frame_metrics(landmarks, width, height):
    left_ear = get_eye_ratio(landmarks, LEFT_EYE, width, height)
    right_ear = get_eye_ratio(landmarks, RIGHT_EYE, width, height)
    blink_ratio = (left_ear + right_ear) / 2
    left_iris = get_iris_center(landmarks, LEFT_IRIS, width, height)
    right_iris = get_iris_center(landmarks, RIGHT_IRIS, width, height)
    left_diameter = distance([landmarks[469].x * width, landmarks[469].y * height], [landmarks[471].x * width, landmarks[471].y * height])
    right_diameter = distance([landmarks[474].x * width, landmarks[474].y * height], [landmarks[476].x * width, landmarks[476].y * height])
    return {
        "left_iris": left_iris,
        "right_iris": right_iris,
        "gaze_x": ((left_iris[0] / width) + (right_iris[0] / width)) / 2,
        "gaze_y": ((left_iris[1] / height) + (right_iris[1] / height)) / 2,
        # landmark 469/471 (왼쪽 홍채 상하 또는 좌우 경계점 추정), 474/476 (오른쪽)의 거리를 각각 구해 평균 → 동공 지름 추정치(픽셀 단위)
        "pupil_diameter_px": (left_diameter + right_diameter) / 2,
        "blink_ratio": blink_ratio,
        "blink_flag": 0,  # Placeholder for blink flag
    }


class BlinkDetector:
    """눈 깜빡임(blink) 이벤트를 감지하고, 시행(trial) 시작 시점에 threshold를 고정(freeze)한다."""

    def __init__(self):
        self.open_eye_samples = deque()
        self.threshold = BLINK_THRESHOLD
        self.in_blink = False
        self.last_blink_end = float("-inf")

    # 최근 PRETRIAL_BASELINE_SECONDS 동안의 EAR 샘플만 유지하는 슬라이딩 윈도우
    def add_open_eye_sample(self, timestamp, ear):
        self.open_eye_samples.append((timestamp, ear))
        while self.open_eye_samples and timestamp - self.open_eye_samples[0][0] > PRETRIAL_BASELINE_SECONDS:
            self.open_eye_samples.popleft()

    # 지금까지 모은 open_eye_samples의 EAR 값들 중 90th percentile을 baseline으로 잡음 (눈을 크게 뜬 편의 값을 기준으로 삼아 노이즈에 강건하게)
    def freeze_threshold(self):
        if self.open_eye_samples:
            baseline_ear = np.percentile([ear for _, ear in self.open_eye_samples], 90)
            self.threshold = float(baseline_ear * BLINK_THRESHOLD_RATIO)
        self.in_blink = False
        self.last_blink_end = float("-inf")
        return self.threshold

    # 매 프레임 호출되는 실시간 판정 로직
    def update(self, timestamp, ear):
        if ear < self.threshold:    # (눈이 임계값보다 많이 감긴 상태)인 경우
            if not self.in_blink and timestamp - self.last_blink_end >= MIN_BLINK_INTERVAL_SECONDS:
                self.in_blink = True
            return int(self.in_blink)
        if self.in_blink:
            self.in_blink = False
            self.last_blink_end = timestamp
        return 0
