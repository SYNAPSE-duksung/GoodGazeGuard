"""
task_component.py

측정 화면의 '과제(문제)' 파트. task_data.py의 30문항(Low/Medium/High 각 10개,
취조식/면담 질문)을 순서대로 보여주고, 웹캠 파트(webcam_component.
GazeGuardVideoProcessor)와 연동해서 질문 등장 시점마다 vp.mark_digit()을
호출한다. 난이도(블록)마다 trial을 따로 끊어서 각각 예측을 돌린다.

2026-08-19 회의 반영 내용
------------------------------------------------
- 정답 체크 로직 제거: 취조식 질문은 정답이 없어서, "제출"(정답 입력) 대신
  "답변 완료" 버튼만 누르면 반응시간만 기록하고 다음 단계로 넘어감.
- 시연 모드: 시작 화면에서 "시연 모드(블록당 3문항)" / "정식 모드(블록당
  10문항)"를 매 세션마다 고를 수 있음 (_render_intro 참고). 고른 문제 목록은
  st.session_state.task_active_problems에 저장해서 세션 내내 씀.
- 지표별 비교: 블록(난이도)이 끝날 때마다 예측뿐 아니라 그 블록에서 계산된
  원시 feature 값(pupil/gaze/blink 지표)도 같이 저장해서, 결과 화면에서
  난이도별로 지표가 어떻게 달라졌는지 비교할 수 있게 함 (_finalize_block 참고).

Streamlit 구조상 알아둬야 할 점
------------------------------------------------
- 준비(5초)/휴식(10초) 대기는 처음에 time.sleep(5)/time.sleep(10)처럼 한 번에
  길게 블로킹하는 방식으로 짰었는데, 실사용 중 블록이 하나씩 통째로 건너뛰는
  버그가 발견됨 (Streamlit은 긴 time.sleep()을 스크립트 실행 중간에 걸어두는
  걸 권장하지 않음 -- 연결이 불안정해지면서 위젯이 의도치 않게 재제출되는
  것으로 추정). 그래서 0.3~1초 단위로 짧게 쉬었다 다시 그리는(rerun) 방식의
  "폴링形 대기"로 바꿈: _start_wait()/_render_waiting() 참고. 화면에는 결과적
  으로 카운트다운처럼 보임(오히려 더 자연스러움).
- st.fragment로 대기 화면만 따로 감쌈: 예전엔 대기 중 rerun이 전체 페이지를
  다시 그려서 웹캠(streamlit-webrtc) 연결이 자꾸 끊기는 문제가 있었음.
- 답변 시간(질문당 목표 20~30초)은 자동 강제 종료되지 않음 -- Streamlit은
  사용자 입력(버튼 클릭) 없이는 스크립트가 아예 안 돌아서, "답변 완료"를
  누르는 순간까지 화면이 그대로 떠있고 그때 반응시간이 계산됨. 목표 시간을
  넘겨도 반응시간 값 자체는 그대로 기록되니 나중에 데이터에서 걸러내면 됨.
"""

import random
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from config import OUTPUT_DIR
from task_data import (
    ALL_PROBLEMS,
    DEMO_PROBLEMS_PER_BLOCK,
    DIFFICULTY_LABELS,
    NASA_TLX_ITEMS,
    ORDER_GROUPS,
    READY_SECONDS,
    REST_SECONDS,
    SOLVE_SECONDS,
)

# 결과 화면 '지표별 비교'에 보여줄 대표 feature 몇 개 (41개 다 보여주면 너무
# 많아서, 해석하기 쉬운 것 위주로 추림)
DISPLAY_FEATURES = [
    ("pupil_mean", "동공 크기 평균(z-score)"),
    ("pupil_std", "동공 크기 변동성"),
    ("blink_ratio", "눈 깜빡임 비율"),
    ("gaze_dispersion", "시선 분산도"),
    ("movement_mean", "시선 이동량 평균"),
    ("fixation_mean_duration", "평균 응시 지속시간"),
]


def init_task_state():
    defaults = {
        "task_started": False,
        "task_done": False,
        "task_demo_mode": True,   # 시연 모드 기본값 (블록당 3문항)
        "task_active_problems": None,  # 이번 세션에서 실제로 쓸 문제 dict (시연/정식 모드에 따라 결정)
        "task_group": None,       # ("low","medium","high") 같은 순서 튜플
        "task_block_idx": 0,      # 0,1,2
        "task_problem_idx": 0,    # 블록 안에서 0~N-1
        "task_global_number": 0,  # 화면 표시/위젯 key용 전체 진행 번호
        "task_stage": "intro",    # intro | waiting | solving | block_break | done
        "task_onset_time": None,
        "task_log": [],           # 문항별 기록
        "task_nasa_log": [],      # 블록별 NASA-TLX 기록
        "task_participant_id": "",
        "task_wait_kind": None,       # "ready" | "rest"
        "task_wait_start": None,
        "task_wait_duration": 0.0,
        "task_block_results": [],  # 블록(난이도)별 예측+지표 결과
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _current_problem():
    difficulty = st.session_state.task_group[st.session_state.task_block_idx]
    problem = st.session_state.task_active_problems[difficulty][st.session_state.task_problem_idx]
    return difficulty, problem


def _is_last_problem_in_block():
    difficulty = st.session_state.task_group[st.session_state.task_block_idx]
    return st.session_state.task_problem_idx == len(st.session_state.task_active_problems[difficulty]) - 1


def _is_last_block():
    return st.session_state.task_block_idx == len(st.session_state.task_group) - 1


def _start_wait(kind: str, duration: float):
    """'ready'(질문 준비) 또는 'rest'(질문 사이 휴식) 대기를 시작함.
    실제 대기 진행/종료 처리는 _render_waiting()이 폴링(짧게 자고 다시 그리기)
    방식으로 담당함 -- 긴 time.sleep() 한 방으로 처리하다가 블록이 통째로
    건너뛰는 버그가 있어서 이렇게 잘게 쪼갬."""
    st.session_state.task_wait_kind = kind
    st.session_state.task_wait_start = time.monotonic()
    st.session_state.task_wait_duration = duration
    st.session_state.task_stage = "waiting"


def _finish_wait(vp):
    """대기 시간이 다 찼을 때 상태 전이 처리 (한 번만 실행, 그 다음 전체 페이지 rerun)."""
    if st.session_state.task_wait_kind == "ready":
        st.session_state.task_global_number += 1
        if vp is not None:
            # 블록마다 trial을 따로 끊으므로, mark_digit 번호도 블록 안에서
            # 1부터 다시 시작해야 함 (전체 번호가 아니라). task_global_number는
            # 화면 표시/위젯 key 용으로만 계속 늘어남.
            #
            # 웹캠이 중간에 끊겼다 재연결되면 streamlit-webrtc가 video_processor를
            # 새로 만드는데, 그 새 인스턴스는 start_trial()이 아직 안 불린 상태라
            # recorder.recording이 False라서 mark_digit()이 ValueError를 던짐 --
            # 이걸 안 잡으면 과제 화면 전체가 에러로 멈춰버려서, 방어적으로
            # start_trial()을 한 번 다시 걸어보고 그래도 안 되면 이번 문항의
            # digit 기록만 포기하고 흐름은 계속 진행시킴.
            try:
                vp.mark_digit(st.session_state.task_problem_idx + 1)
            except ValueError:
                try:
                    vp.start_trial()
                    vp.mark_digit(st.session_state.task_problem_idx + 1)
                except Exception:
                    pass
        st.session_state.task_onset_time = time.monotonic()
        st.session_state.task_stage = "solving"
    else:  # rest 종료 -> 블록이 끝났으면 이 블록 trial을 마감하고 설문으로, 아니면 다음 질문 준비로
        if _is_last_problem_in_block():
            _finalize_block(vp)
            st.session_state.task_stage = "block_break"
        else:
            st.session_state.task_problem_idx += 1
            _start_wait("ready", READY_SECONDS)
    st.rerun()


def _render_waiting_content(vp):
    elapsed = time.monotonic() - st.session_state.task_wait_start
    remaining = st.session_state.task_wait_duration - elapsed

    if remaining <= 0:
        _finish_wait(vp)
        return

    if st.session_state.task_wait_kind == "ready":
        st.markdown(
            "<div style='text-align:center; font-size:3rem; padding:2rem;'>+</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='text-align:center; padding:2rem;'>잠깐 휴식... ({remaining:0.0f}초)</div>",
            unsafe_allow_html=True,
        )


if hasattr(st, "fragment"):
    # st.fragment로 감싸면 이 부분만 주기적으로 다시 그려지고, 웹캠이 있는
    # 나머지 화면(col1)은 건드리지 않음 -- 예전에 매번 전체 페이지를
    # st.rerun()으로 다시 그렸더니 streamlit-webrtc 연결이 자꾸 끊기는
    # 문제가 있어서 이렇게 바꿈.
    @st.fragment(run_every=1)
    def _waiting_fragment(vp):
        _render_waiting_content(vp)

    def _render_waiting(vp):
        _waiting_fragment(vp)
else:
    # 혹시 st.fragment를 지원하지 않는 옛날 streamlit 버전이면 예전 방식(전체
    # rerun)으로 대체. 1초 간격이라 예전 0.3초보다는 웹캠에 부담이 덜함.
    def _render_waiting(vp):
        _render_waiting_content(vp)
        if st.session_state.task_stage == "waiting":
            time.sleep(1.0)
            st.rerun()


def render_task_panel(vp):
    """view_measuring()의 '과제' 칸에서 호출. vp는 GazeGuardVideoProcessor 또는 None."""
    init_task_state()

    if st.session_state.task_done:
        st.success("과제가 모두 끝났어요. 결과 화면으로 이동합니다...")
        return

    if st.session_state.task_stage == "intro":
        _render_intro(vp)
        return

    if st.session_state.task_stage == "waiting":
        _render_waiting(vp)
        return

    if st.session_state.task_stage == "solving":
        _render_solving(vp)
        return

    if st.session_state.task_stage == "block_break":
        _render_block_break(vp)
        return


def _render_intro(vp):
    st.session_state.task_participant_id = st.text_input(
        "참가자 ID (선택, 비워두면 'demo')", value=st.session_state.task_participant_id
    )

    mode_label = st.radio(
        "진행 모드",
        options=["시연 모드 (블록당 3문항, 빠른 확인용)", "정식 모드 (블록당 10문항)"],
        index=0 if st.session_state.task_demo_mode else 1,
    )
    st.session_state.task_demo_mode = mode_label.startswith("시연")

    if vp is None:
        st.warning("웹캠이 아직 준비되지 않아서 과제를 시작할 수 없어요. 웹캠 연결 후 다시 시도해주세요.")
        return

    n_per_block = DEMO_PROBLEMS_PER_BLOCK if st.session_state.task_demo_mode else len(ALL_PROBLEMS["low"])
    st.caption(f"질문은 총 {n_per_block * 3}개(난이도별 {n_per_block}개)이고, 순서는 시작 시 6가지 중 하나로 무작위 배정돼요.")

    if st.button("과제 시작", type="primary", key="task_start_btn"):
        if st.session_state.task_demo_mode:
            st.session_state.task_active_problems = {
                diff: probs[:DEMO_PROBLEMS_PER_BLOCK] for diff, probs in ALL_PROBLEMS.items()
            }
        else:
            st.session_state.task_active_problems = ALL_PROBLEMS

        st.session_state.task_group = random.choice(ORDER_GROUPS)
        st.session_state.task_block_idx = 0
        st.session_state.task_problem_idx = 0
        st.session_state.task_global_number = 0
        st.session_state.task_log = []
        st.session_state.task_nasa_log = []
        st.session_state.task_block_results = []
        vp.start_trial()
        _start_wait("ready", READY_SECONDS)
        st.rerun()


def _render_solving(vp):
    difficulty, problem = _current_problem()
    n_total = len(st.session_state.task_group) * len(st.session_state.task_active_problems[difficulty])
    st.markdown(f"**난이도: {DIFFICULTY_LABELS[difficulty]}**")
    st.markdown(f"### {problem['question']}")
    st.caption(f"목표 답변 시간 20~30초 이내 (문항 {st.session_state.task_global_number}/{n_total}) -- 정답을 채점하지 않으니 편하게 답변하시면 돼요.")

    note = st.text_area("답변 메모 (선택, 면담자가 참고용으로 기록)", key=f"note_{st.session_state.task_global_number}")
    done_clicked = st.button("답변 완료", type="primary", key=f"done_btn_{st.session_state.task_global_number}")

    if not done_clicked:
        return

    response_time = time.monotonic() - st.session_state.task_onset_time
    st.session_state.task_log.append({
        "participant_id": st.session_state.task_participant_id or "demo",
        "problem_id": problem["id"],
        "difficulty": difficulty,
        "question": problem["question"],
        "response_time_sec": round(response_time, 3),
        "note": note,
        "timed_out": response_time > SOLVE_SECONDS,
    })

    _start_wait("rest", REST_SECONDS)
    st.rerun()


def _finalize_block(vp):
    """방금 끝난 블록(한 난이도)의 trial을 마감하고 그 블록만의 예측 + 원시
    feature 값을 저장함. 난이도 3개를 하나로 뭉쳐서 예측하면 신호가 섞여서
    난이도별 차이를 볼 수 없어서, 블록(=난이도)마다 trial을 따로 끊어 따로
    예측함 (실제 서비스도 긴 세션 하나를 통째로 보는 게 아니라 최근 구간
    단위로 계속 갱신되는 구조일 거라 이 편이 더 현실에 가까움).
    feature 값은 결과 화면의 '지표별 비교'에서 씀 (DISPLAY_FEATURES 참고)."""
    difficulty = st.session_state.task_group[st.session_state.task_block_idx]
    result = {"difficulty": difficulty, "label": None, "proba": None, "error": None, "features": None}

    if vp is None:
        result["error"] = "웹캠이 연결되지 않아 예측을 건너뜀."
        st.session_state.task_block_results.append(result)
        return

    raw_df, quality = vp.end_trial()
    if raw_df.empty or not quality["quality_accepted"]:
        result["error"] = f"trial 품질이 기준 미달이라 예측을 건너뜀 (얼굴 인식률 {quality['valid_frame_ratio']:.1%})."
    else:
        try:
            from extract_features import extract_features
            import model_runtime
            features = extract_features(raw_df)
            label, proba = model_runtime.predict_features(features)
            result["label"] = label
            result["proba"] = list(proba)
            result["features"] = features.iloc[0].to_dict()
        except Exception as e:
            result["error"] = str(e)

    st.session_state.task_block_results.append(result)


def _render_block_break(vp):
    difficulty = st.session_state.task_group[st.session_state.task_block_idx]
    is_last = _is_last_block()

    st.info(f"'{DIFFICULTY_LABELS[difficulty]}' 블록 완료! 잠깐 쉬었다가 아래 설문에 답해주세요.")

    with st.form(key=f"nasa_tlx_form_{st.session_state.task_block_idx}"):
        responses = {}
        for key, label in NASA_TLX_ITEMS:
            responses[key] = st.slider(label, min_value=0, max_value=10, value=5, key=f"nasa_{st.session_state.task_block_idx}_{key}")
        button_label = "측정 종료" if is_last else "다음 블록 시작"
        submitted = st.form_submit_button(button_label, type="primary")

    if not submitted:
        return

    record = {
        "participant_id": st.session_state.task_participant_id or "demo",
        "block_index": st.session_state.task_block_idx,
        "difficulty": difficulty,
    }
    record.update(responses)
    st.session_state.task_nasa_log.append(record)

    if is_last:
        _finalize_session()
    else:
        st.session_state.task_block_idx += 1
        st.session_state.task_problem_idx = 0
        if vp is not None:
            vp.start_trial()  # 다음 블록(난이도)을 위한 새 trial 시작
        _start_wait("ready", READY_SECONDS)

    st.rerun()


def _finalize_session():
    """블록별 예측은 이미 _finalize_block()에서 다 끝나있음 -- 여기선 로그 저장 +
    결과 화면으로 자동 이동만 처리."""
    _save_logs()
    st.session_state.task_done = True
    st.session_state.stage = "result"  # app.py의 화면 전환용 키


def reset_task_state():
    """'첫 화면으로' 눌렀을 때 과제 진행 상태를 전부 초기화."""
    for key in list(st.session_state.keys()):
        if key.startswith("task_"):
            del st.session_state[key]
    init_task_state()


def _save_logs():
    """문항별 로그 + NASA-TLX 로그를 output/에 CSV로 저장."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    pid = st.session_state.task_participant_id or "demo"

    if st.session_state.task_log:
        task_path = OUTPUT_DIR / f"task_log_{pid}_{ts}.csv"
        pd.DataFrame(st.session_state.task_log).to_csv(task_path, index=False)

    if st.session_state.task_nasa_log:
        nasa_path = OUTPUT_DIR / f"nasa_tlx_{pid}_{ts}.csv"
        pd.DataFrame(st.session_state.task_nasa_log).to_csv(nasa_path, index=False)
