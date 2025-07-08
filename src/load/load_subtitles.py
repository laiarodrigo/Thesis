import pysrt
import pandas as pd

def load_subtitles(path):
    subs = pysrt.open(path)
    return [
        {
            "start": pd.to_timedelta(s.start.ordinal, unit="ms"),  # milliseconds → Timedelta
            "end":   pd.to_timedelta(s.end.ordinal,   unit="ms"),
            "text":  s.text.replace("\n", " ").strip(),
        }
        for s in subs
    ]