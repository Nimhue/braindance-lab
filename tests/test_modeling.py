import numpy as np
import pandas as pd

from braindance.labels import SLEEP_STAGES
from braindance.modeling import emotion_cv, feature_columns, sleep_cv


def _toy_sleep(rng, n_subj=4, n=120):
    rows = []
    for s in range(n_subj):
        stages = np.repeat(rng.choice(SLEEP_STAGES, size=n // 10), 10)
        for i, st in enumerate(stages):
            k = SLEEP_STAGES.index(st)
            rows.append({"subject": f"S{s}", "session": f"n{s}", "epoch_idx": i, "stage": st,
                         "f1": k + rng.normal(0, 0.6), "f2": (k % 2) + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


def test_sleep_cv_smoothing_helps_on_blocky_sequences():
    df = _toy_sleep(np.random.default_rng(0))
    cv = sleep_cv(df, ["f1", "f2"], n_splits=4, max_iter=40)
    assert feature_columns(df) == ["f1", "f2"]
    assert cv.metrics_raw["kappa"] > 0.3
    assert cv.metrics_smooth["kappa"] >= cv.metrics_raw["kappa"]
    assert np.allclose(cv.transition.sum(1), 1)


def test_emotion_cv_song_level_and_shuffle():
    rng = np.random.default_rng(1)
    rows = []
    for s in range(5):
        for song in range(6):
            v = song % 2
            for w in range(10):
                rows.append({"subject": f"M{s}", "session": str(song), "epoch_idx": w, "valence": v,
                             "d__x": v + rng.normal(0, 0.8)})
    df = pd.DataFrame(rows)
    real = emotion_cv(df, ["d__x"], "valence", max_iter=30)
    fake = emotion_cv(df, ["d__x"], "valence", shuffle_labels=True, max_iter=30)
    assert real["song"]["n"] == 30
    assert real["song"]["balanced_accuracy"] > fake["song"]["balanced_accuracy"]
