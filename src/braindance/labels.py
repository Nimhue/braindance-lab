"""Label mapping for both sources.

Sleep-EDF hypnograms are scored in Rechtschaffen & Kales stages; we collapse
them to the modern AASM five (stages 3 and 4 merge into N3).
MET labels come from post-song self-ratings on 1–9 scales.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

SLEEP_STAGES = ["W", "N1", "N2", "N3", "REM"]
SLEEP_STAGE_INDEX = {s: i for i, s in enumerate(SLEEP_STAGES)}

_RK_TO_AASM = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}
MOVEMENT = "Movement time"


def rk_to_aasm(description: str) -> str | None:
    """None for 'Sleep stage ?', movement time and anything unknown."""
    return _RK_TO_AASM.get(description.strip())


def hypnogram_to_epochs(onsets: Sequence[float], durations: Sequence[float],
                        descriptions: Sequence[str], epoch_s: float = 30.0,
                        recording_s: float | None = None) -> list[dict]:
    """Expand hypnogram annotations into one record per scoring epoch.

    Sleep-EDF hypnograms list *runs* (e.g. 'Sleep stage W' lasting 2,400 s);
    each run becomes ``duration / epoch_s`` epochs. Epochs carry ``stage=None``
    when unscored, and ``movement=True`` when inside a movement-time run.
    """
    out: list[dict] = []
    for onset, dur, desc in zip(onsets, durations, descriptions):
        n = int(round(float(dur) / epoch_s))
        stage = rk_to_aasm(desc)
        for k in range(n):
            start = float(onset) + k * epoch_s
            if recording_s is not None and start + epoch_s > recording_s:
                break
            out.append({
                "epoch_idx": int(round(start / epoch_s)),
                "t_start": start,
                "stage": stage,
                "movement": desc.strip() == MOVEMENT,
            })
    return out


def trim_wake(epochs: list[dict], margin_s: float = 30 * 60) -> list[dict]:
    """Keep only ``margin_s`` of wake before the first and after the last sleep epoch.

    Sleep-EDF cassette recordings run ~20 h, mostly daytime wake. Without
    trimming, 'W' dominates and accuracy looks great for the wrong reason.
    """
    sleep_t = [e["t_start"] for e in epochs if e["stage"] not in (None, "W")]
    if not sleep_t:
        return []
    lo, hi = min(sleep_t) - margin_s, max(sleep_t) + margin_s
    return [e for e in epochs if lo <= e["t_start"] <= hi]


def rating_to_binary(rating: float | None, threshold: float = 5.0) -> int | None:
    """1–9 SAM rating -> 1 (high) / 0 (low). Exactly-on-threshold is ambiguous -> None."""
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return None
    if rating > threshold:
        return 1
    if rating < threshold:
        return 0
    return None


def _missing(v) -> bool:
    import pandas as pd

    return v is None or bool(pd.isna(v))


def quadrant(valence: int | None, arousal: int | None) -> str | None:
    """Binary valence/arousal -> 'HVHA' | 'HVLA' | 'LVHA' | 'LVLA'.

    Missing (None / NaN / pd.NA) on either axis -> None. Pandas silently turns
    None into NaN in numeric columns, and NaN is truthy, so this guard matters.
    """
    if _missing(valence) or _missing(arousal):
        return None
    return f"{'H' if valence else 'L'}V{'H' if arousal else 'L'}A"


def encode(labels: Iterable[str], classes: Sequence[str] = SLEEP_STAGES) -> np.ndarray:
    lut = {c: i for i, c in enumerate(classes)}
    return np.array([lut[x] for x in labels], dtype=int)
