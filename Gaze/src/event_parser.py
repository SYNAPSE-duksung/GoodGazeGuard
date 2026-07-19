import os
import pandas as pd

BASE_PATH = r"D:\OpenNeuro"

def load_events(subject):

    data_path = os.path.join(BASE_PATH, subject, "pupil")

    events = pd.read_csv(
        os.path.join(data_path, f"{subject}_task-memory_events.tsv"),
        sep="\t"
    )

    return events