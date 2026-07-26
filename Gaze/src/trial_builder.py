import pandas as pd


def parse_label(label):
    """
    Event label parsing

    50xxxx : Listening
    60xxxxx: Memory
    """

    label = str(int(label))

    if label.startswith("50"):
        return {
            "task": "listening",
            "digit": int(label[2:4]),
            "seq": int(label[4:6])
        }

    elif label.startswith("60"):
        return {
            "task": "memory",
            "digit": int(label[2:4]),
            "seq": int(label[4:6]),
            "correct": int(label[6])
        }

    else:
        raise ValueError(f"Unknown label : {label}")


def make_trials(events):
    """
       Event들을 sequence(5/9/13) 단위로 묶어서 반환
    """

    trials = []

    i = 0

    while i < len(events):

        info = parse_label(events.iloc[i]["label"])
        seq = info["seq"]

        trials.append(events.iloc[i:i + seq].copy())

        i += seq

    return trials