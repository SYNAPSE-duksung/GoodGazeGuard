DIFFICULTY_MAP = {
    5: "Low",
    9: "Medium",
    13: "High",
}


def get_difficulty(seq_len):
    """
    자릿수(sequence_length)를 Low/Medium/High 난이도로 통일
    gaze / pupil / blink 세 모달리티 feature에서 동일하게 사용
    """
    if seq_len not in DIFFICULTY_MAP:
        raise ValueError(f"Unknown sequence length: {seq_len}")
    return DIFFICULTY_MAP[seq_len]