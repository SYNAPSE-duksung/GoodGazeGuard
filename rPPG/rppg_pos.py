"""
rppg_pos.py
-----------
UBFC-rPPG 영상(vid.avi)에서 얼굴 ROI를 검출하고,
POS(Plane-Orthogonal-to-Skin) 알고리즘으로 rPPG 파형을 추출하는 모듈.

의존 라이브러리:
    pip install opencv-python mediapipe numpy scipy

사용 예:
    from rppg_pos import extract_rppg_from_video
    rppg_signal, fps, timestamps = extract_rppg_from_video("subject1/vid.avi")
"""

import cv2
import numpy as np
from scipy.signal import butter, filtfilt

try:
    # mediapipe 0.10.30+ 일부 빌드에서 `import mediapipe as mp` 후
    # `mp.solutions`로 바로 접근하면 AttributeError가 나는 경우가 있어
    # legacy solutions 서브모듈을 명시적으로 import해서 우회.
    import mediapipe as mp
    from mediapipe.python.solutions import face_mesh as mp_face_mesh_module
    _MP_AVAILABLE = True
except (ImportError, AttributeError, ModuleNotFoundError):
    try:
        # 위 경로도 안 되면 일반적인 solutions 경로로 한 번 더 시도
        from mediapipe import solutions as mp_solutions_module
        mp_face_mesh_module = mp_solutions_module.face_mesh
        _MP_AVAILABLE = True
    except (ImportError, AttributeError, ModuleNotFoundError):
        _MP_AVAILABLE = False


# -----------------------------------------------------------------
# 1. 얼굴 ROI 추출
# -----------------------------------------------------------------
class FaceROIExtractor:
    """
    mediapipe가 있으면 FaceMesh 랜드마크로 이마+양볼 ROI를 잡고,
    없으면 OpenCV Haar cascade로 얼굴 바운딩박스 내부 중앙 영역을 ROI로 사용.
    """

    # FaceMesh 기준 이마/볼 랜드마크 인덱스 (근사치)
    FOREHEAD_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                    361, 288, 397, 365, 379, 378, 400, 377, 152, 148]
    LEFT_CHEEK_IDX = [116, 117, 118, 119, 100, 126, 209, 49, 129]
    RIGHT_CHEEK_IDX = [345, 346, 347, 348, 329, 355, 429, 279, 358]

    def __init__(self, use_mediapipe: bool = True):
        self.use_mediapipe = use_mediapipe and _MP_AVAILABLE
        if self.use_mediapipe:
            self.mp_face_mesh = mp_face_mesh_module.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def get_roi_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """frame과 같은 크기의 0/1 마스크(ROI=1)를 반환. 검출 실패 시 전부 0."""
        h, w = frame_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if self.use_mediapipe:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return mask
            landmarks = results.multi_face_landmarks[0].landmark

            for idx_group in (self.FOREHEAD_IDX, self.LEFT_CHEEK_IDX, self.RIGHT_CHEEK_IDX):
                pts = np.array(
                    [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in idx_group],
                    dtype=np.int32,
                )
                if len(pts) >= 3:
                    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
            return mask
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) == 0:
                return mask
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            # 얼굴 박스 중앙 60% 영역(이마~볼 부근)만 ROI로 사용
            rx1, ry1 = x + int(fw * 0.2), y + int(fh * 0.1)
            rx2, ry2 = x + int(fw * 0.8), y + int(fh * 0.65)
            mask[ry1:ry2, rx1:rx2] = 1
            return mask


# -----------------------------------------------------------------
# 2. 프레임별 평균 RGB 신호 추출
# -----------------------------------------------------------------
def extract_rgb_trace(video_path: str, roi_extractor: FaceROIExtractor):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없음: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # UBFC-rPPG 기본값 fallback

    rgb_trace = []
    timestamps = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = roi_extractor.get_roi_mask(frame)
        if mask.sum() > 0:
            b = frame[:, :, 0][mask == 1].mean()
            g = frame[:, :, 1][mask == 1].mean()
            r = frame[:, :, 2][mask == 1].mean()
        else:
            # 검출 실패 프레임은 이전 값 유지(보간용으로 NaN 처리)
            r, g, b = np.nan, np.nan, np.nan

        rgb_trace.append([r, g, b])
        timestamps.append(frame_idx / fps)
        frame_idx += 1

    cap.release()

    rgb_trace = np.array(rgb_trace, dtype=np.float64)
    # NaN(검출 실패 프레임) 선형 보간
    for ch in range(3):
        col = rgb_trace[:, ch]
        nans = np.isnan(col)
        if nans.any() and (~nans).sum() > 1:
            col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), col[~nans])
        rgb_trace[:, ch] = col

    return rgb_trace, fps, np.array(timestamps)


# -----------------------------------------------------------------
# 3. POS 알고리즘 (Wang et al., 2017)
# -----------------------------------------------------------------
def pos_algorithm(rgb_trace: np.ndarray, fps: float, window_sec: float = 1.6) -> np.ndarray:
    """
    rgb_trace: (N, 3) array, 각 행이 [R, G, B] 프레임 평균값
    반환: 길이 N의 rPPG pulse 신호 (정규화되지 않은 원신호)
    """
    N = rgb_trace.shape[0]
    win_len = max(2, int(window_sec * fps))
    H = np.zeros(N)

    for start in range(0, N - win_len + 1):
        window = rgb_trace[start:start + win_len].T  # shape (3, win_len)
        mean_rgb = window.mean(axis=1, keepdims=True)
        mean_rgb[mean_rgb == 0] = 1e-6
        Cn = window / mean_rgb  # temporal normalization

        # 투영 행렬
        S1 = Cn[1] - Cn[2]                 # G - B
        S2 = Cn[1] + Cn[2] - 2 * Cn[0]     # G + B - 2R

        std_S1 = S1.std() if S1.std() > 1e-8 else 1e-8
        std_S2 = S2.std() if S2.std() > 1e-8 else 1e-8
        alpha = std_S1 / std_S2
        h = S1 + alpha * S2

        h = h - h.mean()
        H[start:start + win_len] += h

    return H


def bandpass_filter(signal: np.ndarray, fps: float, low=0.7, high=3.5, order=3) -> np.ndarray:
    """심박 대역(0.7~3.5Hz ≈ 42~210 bpm)만 통과시키는 버터워스 대역통과 필터."""
    nyq = fps / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


# -----------------------------------------------------------------
# 4. 통합 함수
# -----------------------------------------------------------------
def extract_rppg_from_video(video_path: str, use_mediapipe: bool = True):
    """
    반환: (filtered_rppg_signal, fps, timestamps)
    """
    roi_extractor = FaceROIExtractor(use_mediapipe=use_mediapipe)
    rgb_trace, fps, timestamps = extract_rgb_trace(video_path, roi_extractor)
    raw_signal = pos_algorithm(rgb_trace, fps)
    filtered_signal = bandpass_filter(raw_signal, fps)
    return filtered_signal, fps, timestamps


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python rppg_pos.py <vid.avi 경로>")
        sys.exit(1)

    sig, fps, ts = extract_rppg_from_video(sys.argv[1])
    print(f"추출 완료: {len(sig)} samples @ {fps:.2f} fps ({ts[-1]:.1f}s)")