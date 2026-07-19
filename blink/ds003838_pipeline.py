"""
ds003838_pipeline.py
=============================================================
OpenNeuro ds003838 데이터를 S3(HTTPS 공개 엔드포인트)에서 스트리밍으로
읽어오는 다운로드 전용 모듈. 설정값은 전부 config.py에서 가져옴
"""

import io
import os
import socket
import threading
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import urllib3.util.connection as _urllib3_conn

import config

try:
    from tqdm import tqdm
    _log = tqdm.write  # tqdm 진행바와 동시에 출력해도 화면이 깨지지 않도록
except ImportError:
    _log = print


# =============================================================
# IPv4 강제 + 강제 워치독(hard timeout)
# =============================================================
def _force_ipv4_only():
    def _allowed_gai_family():
        return socket.AF_INET
    _urllib3_conn.allowed_gai_family = _allowed_gai_family

_force_ipv4_only()

_TIMEOUT = (config.CONNECT_TIMEOUT, config.READ_TIMEOUT)
_session = requests.Session()  # 커넥션 재사용 (스레드마다 매번 새로 여는 것보다 빠름, thread-safe)


def _with_hard_timeout(func, *args, **kwargs):
    """
    실제 요청을 daemon 스레드에서 실행하고, HARD_TIMEOUT_SEC 안에 안 끝나면
    강제로 실패 처리. daemon=True라서, 이 요청이 끝내 안 끝나더라도 메인
    스크립트가 다 끝나면 프로세스 자체는 정상 종료됨 (아니면 좀비 스레드
    때문에 스크립트가 마지막에 또 멈춰있는 것처럼 보임).
    """
    last_error = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        box = {}

        def _runner():
            try:
                box["value"] = func(*args, **kwargs)
            except Exception as e:
                box["error"] = e

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join(timeout=config.HARD_TIMEOUT_SEC)

        if t.is_alive():
            last_error = TimeoutError(
                f"{config.HARD_TIMEOUT_SEC}초 안에 응답이 없어 강제 중단 "
                f"(시도 {attempt}/{config.MAX_RETRIES})"
            )
            print(f"    ⏱️ {last_error}", flush=True)
            continue

        if "error" in box:
            last_error = box["error"]
            print(f"    ⚠️ 요청 실패 (시도 {attempt}/{config.MAX_RETRIES}): {last_error}", flush=True)
            continue

        return box.get("value")

    raise last_error


# =============================================================
# 저수준 HTTP 유틸
# =============================================================
def _get_text(url: str) -> str:
    print(f"GET(TEXT): {url}", flush=True)

    def _do_request():
        resp = requests.get(url, timeout=(5, 20), headers={"Connection": "close"})
        resp.raise_for_status()
        return resp.text

    return _with_hard_timeout(_do_request)


def _list_prefix(prefix: str):
    """S3 REST API(ListObjectsV2)를 HTTPS GET으로 직접 호출해 해당 prefix 아래
    파일 목록(Key 전체)을 반환."""
    url = f"https://s3.amazonaws.com/{config.BUCKET}?list-type=2&prefix={prefix}"
    xml_text = _get_text(url)
    root = ET.fromstring(xml_text)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    return [el.text for el in root.findall(".//s3:Contents/s3:Key", ns)]


# =============================================================
# 참가자 목록
# =============================================================
def get_subject_list():
    """participants.tsv를 읽어 참가자 ID 목록(예: ['sub-013', ...])을 반환.
    pupil 데이터 결측으로 알려진 참가자(config.SUBJECTS_MISSING_PUPIL)는
    시도해봐야 어차피 실패하므로 여기서 미리 제외합니다."""
    text = _get_text(f"{config.HTTPS_ROOT}/participants.tsv")
    participants = pd.read_csv(io.StringIO(text), sep="\t")
    id_col = "participant_id" if "participant_id" in participants.columns else participants.columns[0]
    subs = sorted(participants[id_col].astype(str).tolist())

    def _subject_num(sub_id: str):
        try:
            return int(sub_id.split("-")[-1])
        except ValueError:
            return None

    before = len(subs)
    subs = [s for s in subs if _subject_num(s) not in config.SUBJECTS_MISSING_PUPIL]
    excluded = before - len(subs)
    if excluded:
        print(f"ℹ️ pupil 데이터 결측으로 알려진 참가자 {excluded}명 제외 "
              f"({sorted(config.SUBJECTS_MISSING_PUPIL)})", flush=True)
    return subs


# =============================================================
# 파일 경로 탐색
# =============================================================
def _resolve_pupil_url(sub_id: str) -> str:
    """
    기본적으로 확인된 패턴으로 바로 URL을 만들어 시도하고(HEAD 요청으로
    존재 확인, 빠름), 없으면 해당 서브젝트의 pupil/ 폴더를 실제로 나열해서
    "_pupil.tsv"로 끝나는 파일을 찾습니다.
    """
    fast_path = f"{sub_id}/pupil/{sub_id}_task-{config.DEFAULT_TASK}_pupil.tsv"
    print(f"{sub_id} HEAD 요청 시작", flush=True)
    head = _with_hard_timeout(
        _session.head,
        f"https://s3.amazonaws.com/{config.BUCKET}/{config.DATASET_ID}/{fast_path}",
        timeout=_TIMEOUT,
    )
    print(f"{sub_id} HEAD 상태코드 : {head.status_code}", flush=True)
    if head.status_code == 200:
        return f"{config.HTTPS_ROOT}/{fast_path}"

    # 기본 경로가 아니면, 실제 폴더를 나열해서 찾기
    prefix = f"{config.DATASET_ID}/{sub_id}/pupil/"
    keys = _list_prefix(prefix)
    pupil_keys = [k for k in keys if k.endswith("_pupil.tsv") or k.endswith("_pupil.tsv.gz")]
    if pupil_keys:
        key = sorted(pupil_keys)[0]
        return f"https://s3.amazonaws.com/{config.BUCKET}/{key}"

    raise FileNotFoundError(
        f"{sub_id}: '_pupil.tsv' 파일을 찾지 못했습니다. "
        f"sanity_check('{sub_id}')를 실행해서 실제 폴더 구조를 확인해보세요."
    )


# =============================================================
# tsv 로딩 (+ 로컬 캐시)
# =============================================================
def load_tsv_from_s3(sub_id: str, kind: str = "pupil") -> pd.DataFrame:
    """
    지정한 kind의 tsv를 HTTPS로 스트리밍 다운로드해 DataFrame으로 반환.
    USE_CACHE=True이면 최초 1회만 받고, 이후엔 로컬 parquet 캐시를 사용.
    """
    if kind != "pupil":
        raise NotImplementedError(f"kind='{kind}'는 아직 구현되어 있지 않습니다.")

    cache_path = None
    if config.USE_CACHE:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(config.CACHE_DIR, f"{sub_id}_{kind}.parquet")
        if os.path.exists(cache_path):
            _log(f"    [캐시 사용] {cache_path}")
            return pd.read_parquet(cache_path)

    # ---- 수동 다운로드 우선 사용 ----
    # 자동 다운로드가 계속 실패하는 참가자는, 브라우저나 `aws s3 cp`로 직접
    # 받아서 config.MANUAL_DOWNLOAD_DIR 폴더에 원래 파일명 그대로 넣어두면
    # (예: sub-014_task-memory_pupil.tsv), 네트워크 시도 자체를 건너뛰고
    # 이 파일을 바로 씁니다.
    manual_path = os.path.join(
        config.MANUAL_DOWNLOAD_DIR, f"{sub_id}_task-{config.DEFAULT_TASK}_pupil.tsv"
    )
    if os.path.exists(manual_path):
        print(f"📁 [{sub_id}] 수동 다운로드 파일 사용: {manual_path}", flush=True)
        df = pd.read_csv(manual_path, sep="\t")
        if config.USE_CACHE:
            df.to_parquet(cache_path)
        return df

    url = _resolve_pupil_url(sub_id)

    print(f"\n========== {sub_id} ==========", flush=True)
    print("[1] URL 결정 완료", flush=True)
    print(url, flush=True)
    print("[2] GET 시작", flush=True)

    def _do_download():
        # stream=True + 바이트 단위 진행 바: 다운로드가 "느린 것"과 "완전히
        # 멈춘 것"을 구분할 수 있게 해줌 (막대가 조금씩이라도 움직이면 살아있는 것)
        resp = requests.get(url, timeout=(5, 20), headers={"Connection": "close"}, stream=True)
        resp.raise_for_status()

        total = int(resp.headers.get("Content-Length", 0)) or None
        chunks = bytearray()
        with tqdm(total=total, unit="B", unit_scale=True,
                  desc=f"   ㄴ {sub_id} 다운로드", leave=False) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                if chunk:
                    chunks.extend(chunk)
                    pbar.update(len(chunk))
        return bytes(chunks)

    content = _with_hard_timeout(_do_download)
    print("[3] GET 완료", flush=True)

    if url.endswith(".gz"):
        df = pd.read_csv(io.BytesIO(content), sep="\t", compression="gzip")
    else:
        df = pd.read_csv(io.BytesIO(content), sep="\t")
    print("[4] pandas 파싱 완료", flush=True)

    if config.USE_CACHE:
        df.to_parquet(cache_path)
        _log(f"    [캐시 저장] {cache_path}")

    return df


# =============================================================
# 샘플링레이트
# =============================================================
def get_sampling_rate(sub_id: str = None) -> int:
    """현재는 참가자 전원 동일한 고정 샘플링레이트(config.FS)를 반환합니다."""
    return config.FS


# =============================================================
# Sanity check — 단독 실행 시 폴더 구조/컬럼 확인용
# =============================================================
def sanity_check(sub_id: str = None):
    print("[1] participants.tsv 로딩 중...")
    subjects = get_subject_list()
    print(f"    → 총 {len(subjects)}명: {subjects[:5]} ...")

    target = sub_id or subjects[0]
    print(f"\n[2] '{target}' 폴더 구조 확인 중 (prefix: {config.DATASET_ID}/{target}/)...")
    try:
        keys = _list_prefix(f"{config.DATASET_ID}/{target}/")
        for k in keys:
            print("    -", k)
    except Exception as e:
        print(f"    ⚠️ 폴더 탐색 실패: {e}")
        return

    print(f"\n[3] '{target}' pupil tsv 파일 탐색 및 로딩 시도...")
    try:
        df = load_tsv_from_s3(target, kind="pupil")
        print(f"    ✅ 로딩 성공! shape={df.shape}")
        print(f"    컬럼: {list(df.columns)}")
        print(df.head())
        for expected_col in [config.TIME_COL_PUPIL, config.BLINK_COL]:
            status = "✅ 있음" if expected_col in df.columns else "❌ 없음 — config.py 컬럼명을 맞춰줘야 함"
            print(f"    '{expected_col}' 컬럼: {status}")
    except Exception as e:
        print(f"    ⚠️ 로딩 실패: {e}")


if __name__ == "__main__":
    sanity_check()


def load_beh_from_s3(sub_id: str) -> pd.DataFrame:
    """행동 데이터(_beh.tsv)를 S3에서 가져오는 함수"""
    # BIDS 구조상 beh 폴더 안에 있을 확률이 높습니다.
    prefix = f"{config.DATASET_ID}/{sub_id}/beh/"
    keys = _list_prefix(prefix)
    beh_keys = [k for k in keys if k.endswith("_beh.tsv")]

    if not beh_keys:
        raise FileNotFoundError(f"{sub_id}: '_beh.tsv' 행동 파일을 찾지 못했습니다.")

    key = sorted(beh_keys)[0]
    url = f"https://s3.amazonaws.com/{config.BUCKET}/{key}"

    # 기존 다운로드 로직 재사용
    def _do_download():
        resp = requests.get(url, timeout=(5, 20), headers={"Connection": "close"})
        resp.raise_for_status()
        return resp.text

    content = _with_hard_timeout(_do_download)
    return pd.read_csv(io.StringIO(content), sep="\t")
