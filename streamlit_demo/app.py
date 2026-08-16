"""
app.py - GazeGuard 실시간 인지부하 측정 시연 앱 (Streamlit)

화면 흐름: 시작 -> 측정 -> 결과 (st.session_state로 어느 화면인지 기억함 --
Streamlit은 위젯 누를 때마다 코드 전체를 다시 실행하는 구조라 이게 없으면
화면 전환이 유지가 안 됨)

담당 파트별 연결 지점 (다른 두 명이 만드는 부분)
------------------------------------------------
- 웹캠(실시간 신호 추출) 파트  -> get_webcam_features() 자리에 실제 로직 연결
- 문제(과제 로직) 파트         -> run_task() 자리에 실제 숫자 스팬 과제 로직 연결
- 이 둘이 최종적으로 넘겨주는 값을 합쳐서 Branch1 모델 feature 41개를 채우면
  예측(Low/Medium/High)이 나옴

지금 상태
------------------------------------------------
아직 웹캠/과제 파트가 없어서, 더미(랜덤) feature로 "화면 흐름 + 모델 연동"까지만
동작하는 뼈대만 만들어둠. 실제 데이터가 오면 get_dummy_features() 자리를
실제 함수 호출로 바꾸면 됨.

실행 방법:
    pip install streamlit lightgbm pandas numpy
    streamlit run app.py
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
        st.session_state.last_result = None  # (pred_label, proba) 튜플


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
        "숫자 기억 과제를 수행합니다.<br>시작을 누르면 측정 화면으로 이동합니다.</p>",
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
        st.info("여기에 웹캠 담당자의 화면(streamlit-webrtc 컴포넌트 등)을 붙일 예정")
    with col2:
        st.subheader("과제")
        st.info("여기에 문제 담당자의 숫자 스팬 과제 화면을 붙일 예정")

    st.divider()
    st.caption("지금은 두 파트가 아직 없어서, 버튼으로 '측정 종료'를 흉내내는 상태입니다.")

    if centered_button("측정 종료 (임시)"):
        # 실제로는: 과제가 끝나는 시점에 웹캠 파트가 집계한 feature를 모델에 넣어서
        # 예측해야 함. 지금은 모델이 최종본이 아니라서 더미 결과로 대체.
        st.session_state.last_result = get_dummy_prediction()
        go_to("result")


# ------------------------------------------------------------------
# 화면 3: 결과
# ------------------------------------------------------------------
def view_result():
    st.title("결과")

    result = st.session_state.last_result
    if result is None:
        st.warning("측정 데이터가 없습니다. 처음부터 다시 시작해주세요.")
    else:
        pred, proba = result

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # st.metric은 라벨/값 크기 차이가 커서 보기 불편해 커스텀 HTML로 교체.
            # 라벨+값을 한 줄에 같은 크기로 표시함 (span 두 개를 나란히 놓고 폰트 크기 동일하게 지정)
            st.markdown(
                f"""
                <div style='text-align:center; margin-bottom:1rem;'>
                    <span style='font-size:1.6rem; font-weight:600;'>예측된 인지부하 수준: </span>
                    <span style='font-size:1.6rem; font-weight:700; color:#2f855a;'>{pred}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for lab, p in zip(LABELS, proba):
                st.progress(float(p), text=f"{lab}: {p * 100:.1f}%")

    st.write("")
    if centered_button("첫 화면으로"):
        st.session_state.last_result = None
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
