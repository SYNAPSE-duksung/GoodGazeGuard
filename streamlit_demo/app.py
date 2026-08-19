"""
app.py - GazeGuard 실시간 인지부하 측정 시연 앱 (Streamlit)

화면 흐름: 시작 -> 측정 -> 결과 (st.session_state로 어느 화면인지 기억함 --
Streamlit은 위젯 누를 때마다 코드 전체를 다시 실행하는 구조라 이게 없으면
화면 전환이 유지가 안 됨)

담당 파트별 연결 지점
------------------------------------------------
- 웹캠(실시간 신호 추출) 파트  -> webcam_component.py(GazeGuardVideoProcessor)로
  연동 완료. 웹캠 담당자의 원본 파일(config/signal_processing/calibration/
  trial_recorder/extract_features/model_runtime)은 그대로 재사용, cv2 창
  대신 streamlit-webrtc로 화면에 실시간으로 보이게만 새로 짬 (_render_webcam_panel
  참고). pupil calibration(webcam_trial.py로 미리 한 번 해둬야 함)은 아직 이
  화면에 없음 -- 원본 cv2 도구를 따로 실행해서 끝내둬야 함.
- 문제(과제 로직) 파트         -> task_data.py(취조식 질문 30개+순서 그룹) +
  task_component.py(준비 5초 -> 답변 -> 휴식 10초 상태머신, 블록마다 간이
  NASA-TLX)로 연동 완료. 질문이 뜰 때마다 자동으로 vp.mark_digit()을 호출하고,
  trial 시작/종료도 블록(난이도)마다 과제 흐름이 알아서 vp.start_trial()/
  vp.end_trial()을 호출함 (수동 버튼 없앰). 2026-08-19 회의 반영: (1) 정답이
  있는 사칙연산 -> 정답 없는 취조식 질문으로 교체, 정답 체크 로직 제거하고
  '답변 완료' 버튼으로만 종료, (2) 시작 화면에서 시연 모드(블록당 3문항)/
  정식 모드(블록당 10문항)를 매번 선택 가능, (3) 결과 화면에 난이도(블록)별
  예측뿐 아니라 pupil/gaze/blink 원시 지표 비교도 같이 보여줌.
  답변 시간(목표 20~30초)은 자동 강제 종료되지 않음 -- Streamlit은 사용자
  입력(버튼 클릭) 없이는 스크립트가 안 돌아서, 대신 반응시간을 그대로 기록해
  두고 나중에 30초 넘는 것만 데이터에서 걸러내는 방식으로 처리함
  (task_component.py 상단 주석 참고).
- 웹캠이 준비 안 됐거나(패키지 미설치, 카메라 권한 없음 등) trial을 안 돌렸으면
  화면 하단 더미 버튼으로 화면 흐름만 계속 테스트할 수 있음.

실행 방법:
    pip install -r requirements.txt
    streamlit run app.py

주의: mediapipe/opencv 설치 용량이 커서 설치가 몇 분 걸릴 수 있음. 웹캠
연동 부분은 실제 카메라가 있는 환경에서만 테스트 가능함.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb

MODEL_PATH = Path(__file__).parent / "branch1_lightgbm_personalized.txt"
LABELS = ["Low", "Medium", "High"]

# 학습된 모델이 기대하는 feature 순서 그대로 (train_branch1_lightgbm.py의
# build_feature_table(clean_signal=True) 결과와 동일해야 함)
FEATURE_COLS = [
    "pupil_mean", "pupil_std", "pupil_min", "pupil_max", "pupil_first", "pupil_last", "pupil_slope",
    "movement_mean", "movement_std", "movement_max", "movement_min", "movement_median",
    "movement_p95", "movement_p99", "movement_iqr", "movement_cv", "movement_skew", "movement_kurtosis",
    "gaze_dispersion", "dispersion_x", "dispersion_y",
    "center_distance_mean", "center_distance_std", "center_distance_max",
    "gaze_velocity_mean", "gaze_velocity_std", "gaze_velocity_max",
    "acceleration_mean", "acceleration_std", "acceleration_max",
    "fixation_mean_duration", "fixation_max_duration", "hull_area",
    "mean_ibi_ours", "std_ibi_ours", "blink_entropy_trial_ours",
    "blink_ratio", "blink_duration_mean", "blink_duration_std", "blink_duration_max", "blink_duration_min",
]


@st.cache_resource
def load_model():
    return lgb.Booster(model_file=str(MODEL_PATH))


def inject_css():
    """화면 중앙 정렬 + 화면 크기에 맞춰 너비가 알아서 조절되게 하는 CSS.
    Streamlit 기본 layout="centered"는 고정폭(약 730px)이라 큰 모니터에서
    너무 좁아 보임 -- 대신 wide로 두고 CSS로 비율(%) 기반 폭을 직접 지정함."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 70%;
            margin: 0 auto;
            padding-top: 3rem;
        }
        h1, h2, h3 {
            text-align: center;
        }
        div[data-testid="stMetric"] {
            text-align: center;
        }
        @media (max-width: 900px) {
            .block-container { max-width: 92%; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def centered_button(label: str, **kwargs) -> bool:
    """버튼을 가운데 칸에만 넣어서 시각적으로 중앙 정렬되게 하는 헬퍼.
    st.columns로 좌우에 빈 칸을 두고 가운데 칸(비율 1:1:1)에만 버튼을 그림."""
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        return st.button(label, use_container_width=True, **kwargs)


def init_state():
    if "stage" not in st.session_state:
        st.session_state.stage = "start"
    if "last_result" not in st.session_state:
        st.session_state.last_result = None  # (pred_label, proba) 튜플 -- 더미 테스트 버튼 전용


def go_to(stage: str):
    st.session_state.stage = stage
    st.rerun()


# ------------------------------------------------------------------
# TODO: 실제 연동 지점 (웹캠/문제 담당자 코드 완성되면 여기를 교체)
# ------------------------------------------------------------------
def get_dummy_features() -> pd.DataFrame:
    """화면 테스트용 더미 feature. 실제로는 웹캠 파트가 넘겨주는 pupil/gaze/blink
    집계값으로 채워져야 함. (지금은 아래 get_dummy_prediction()을 대신 쓰고 있어서
    실제로 호출되진 않음 -- 나중에 진짜 모델 연동할 때 다시 사용)"""
    rng = np.random.default_rng()
    values = rng.normal(loc=0.0, scale=1.0, size=len(FEATURE_COLS))
    return pd.DataFrame([values], columns=FEATURE_COLS)


def get_dummy_prediction():
    """화면 테스트용 더미 '결과'. 지금 붙어있는 모델은 최종 배포용이 아니라서,
    모델을 거치지 않고 Low/Medium/High 중 하나를 그냥 랜덤으로 골라 그럴듯한
    확률과 함께 돌려줌 (독립적인 랜덤 feature 41개를 실제 모델에 넣으면 거의
    항상 Low로 쏠리는 현상이 있어서, 화면 테스트 목적에는 이 방식이 더 적합함).

    나중에 진짜 모델이 준비되면: 이 함수 호출을 지우고
        model = load_model(); proba = model.predict(X)[0]
    로 교체하면 됨.
    """
    rng = np.random.default_rng()
    pred_idx = int(rng.integers(0, 3))

    top = rng.uniform(0.4, 0.7)          # 1등 클래스 확률
    split = rng.uniform(0.3, 0.7)        # 나머지를 두 클래스에 나누는 비율
    proba = np.zeros(3)
    proba[pred_idx] = top
    others = [i for i in range(3) if i != pred_idx]
    proba[others[0]] = (1 - top) * split
    proba[others[1]] = (1 - top) * (1 - split)

    return LABELS[pred_idx], proba


# ------------------------------------------------------------------
# 화면 1: 시작
# ------------------------------------------------------------------
def view_start():
    st.title("GazeGuard 인지부하 측정 시연")
    st.markdown(
        "<p style='text-align:center;'>웹캠으로 눈 움직임/깜빡임을 측정하면서 "
        "질문에 답변합니다.<br>시작을 누르면 측정 화면으로 이동합니다.</p>",
        unsafe_allow_html=True,
    )
    st.write("")  # 버튼 위 여백
    if centered_button("시작하기", type="primary"):
        go_to("measuring")


# ------------------------------------------------------------------
# 화면 2: 측정 (웹캠 + 과제가 여기서 같이 돌아가는 화면)
# ------------------------------------------------------------------
def view_measuring():
    st.title("측정 중")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("웹캠")
        vp = _render_webcam_panel()

    with col2:
        st.subheader("과제")
        from task_component import render_task_panel
        render_task_panel(vp)

    st.divider()
    st.caption("과제를 다 마치면 자동으로 결과 화면으로 넘어가요. 웹캠 없이 화면 흐름만 보고 싶으면 아래 버튼을 써도 돼요.")
    if centered_button("측정 종료 (더미, 화면 테스트용)"):
        st.session_state.last_result = get_dummy_prediction()
        go_to("result")


def _render_webcam_panel():
    """웹캠 패널: streamlit-webrtc로 실시간 영상을 보여줌. trial 시작/mark_digit/
    종료는 이제 오른쪽 '과제' 패널(task_component.render_task_panel)이 문제
    흐름에 맞춰 알아서 호출하므로, 여기서는 영상 + 상태 표시만 담당함.
    streamlit-webrtc나 mediapipe가 설치 안 돼있으면(또는 카메라 권한이 없으면)
    에러 대신 안내 메시지만 보여주고 None을 반환함 -- 이 패널이 실패해도 화면
    하단의 더미 버튼으로 흐름 테스트는 계속할 수 있게 하기 위함.

    반환값: GazeGuardVideoProcessor 인스턴스 (연결 안 됐으면 None)
    """
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        from webcam_component import GazeGuardVideoProcessor
        from config import PUPIL_REFERENCE_PATH
    except Exception as e:
        st.warning(f"웹캠 컴포넌트를 불러오지 못했어요 (패키지 설치 확인 필요): {e}")
        return None

    if not PUPIL_REFERENCE_PATH.exists():
        st.warning(
            "아직 개인 pupil calibration 파일(pupil_reference.json)이 없어요. "
            "`python webcam_trial.py`로 먼저 calibration을 한 번 끝내야 pupil "
            "feature가 제대로 나와요 (calibration 안 해도 화면은 뜨지만, 결과는 부정확할 수 있음)."
        )

    ctx = webrtc_streamer(
        key="gazeguard-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=GazeGuardVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    if not (ctx and ctx.video_processor):
        st.caption("웹캠 연결 대기 중... (브라우저가 카메라 권한을 물어보면 허용해주세요)")
        return None

    vp = ctx.video_processor
    recording = vp.is_recording()
    st.caption(f"상태: {'🔴 측정 중' if recording else '⚪ 대기 중'}")
    return vp


# ------------------------------------------------------------------
# 화면 3: 결과
# ------------------------------------------------------------------
def _render_one_result(label_prefix: str, pred, proba):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # st.metric은 라벨/값 크기 차이가 커서 보기 불편해 커스텀 HTML로 교체.
        st.markdown(
            f"""
            <div style='text-align:center; margin-bottom:0.5rem;'>
                <span style='font-size:1.3rem; font-weight:600;'>{label_prefix}: </span>
                <span style='font-size:1.3rem; font-weight:700; color:#2f855a;'>{pred}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for lab, p in zip(LABELS, proba):
            st.progress(float(p), text=f"{lab}: {p * 100:.1f}%")


def _render_indicator_comparison(block_results):
    """난이도(블록)별로 계산된 원시 feature 값을 표+막대그래프로 비교.
    회의에서 나온 '각 지표별로 부하가 얼마나 나타났는지 보고 싶다'는
    요청 반영 -- 최종 라벨만으로는 안 보이는 세부 신호 변화를 보여줌."""
    from task_component import DISPLAY_FEATURES
    from task_data import DIFFICULTY_LABELS

    rows = []
    for r in block_results:
        if not r.get("features"):
            continue
        row = {"난이도": DIFFICULTY_LABELS.get(r["difficulty"], r["difficulty"])}
        for col, label in DISPLAY_FEATURES:
            row[label] = r["features"].get(col)
        rows.append(row)

    if not rows:
        return

    st.divider()
    st.subheader("지표별 비교")
    df = pd.DataFrame(rows).set_index("난이도")
    st.dataframe(df.style.format("{:.3f}"), use_container_width=True)

    # 그래프를 세로로 쭉 나열하면 스크롤이 너무 길어져서, 한 줄에 3개씩
    # 격자로 배치함 (st.columns로 줄마다 새로 나눔)
    CHARTS_PER_ROW = 3
    chart_cols = [(col, label) for col, label in DISPLAY_FEATURES if label in df.columns]
    for i in range(0, len(chart_cols), CHARTS_PER_ROW):
        row_items = chart_cols[i:i + CHARTS_PER_ROW]
        cols = st.columns(len(row_items))
        for c, (col, label) in zip(cols, row_items):
            with c:
                st.caption(label)
                st.bar_chart(df[[label]], height=220)


def view_result():
    st.title("결과")

    block_results = st.session_state.get("task_block_results") or []

    if block_results:
        # 난이도(블록)마다 따로 돌린 예측 결과를 각각 보여줌 (task_component._finalize_block 참고)
        from task_data import DIFFICULTY_LABELS
        for i, r in enumerate(block_results):
            diff_label = DIFFICULTY_LABELS.get(r["difficulty"], r["difficulty"])
            if r["error"]:
                st.warning(f"[{diff_label}] 예측 실패: {r['error']}")
            else:
                _render_one_result(f"{diff_label} 블록 -> 예측", r["label"], r["proba"])
            if i < len(block_results) - 1:
                st.write("")
        _render_indicator_comparison(block_results)
    elif st.session_state.last_result is not None:
        pred, proba = st.session_state.last_result
        _render_one_result("예측된 인지부하 수준", pred, proba)
    else:
        st.warning("측정 데이터가 없습니다. 처음부터 다시 시작해주세요.")

    st.write("")
    if centered_button("첫 화면으로"):
        st.session_state.last_result = None
        from task_component import reset_task_state
        reset_task_state()
        go_to("start")


def main():
    st.set_page_config(page_title="GazeGuard 데모", layout="wide")
    inject_css()
    init_state()

    stage = st.session_state.stage
    if stage == "start":
        view_start()
    elif stage == "measuring":
        view_measuring()
    elif stage == "result":
        view_result()


if __name__ == "__main__":
    main()
