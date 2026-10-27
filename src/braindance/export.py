"""Builders for the "BD tape" JSON files the web app plays.

Shared by notebook 07 (real data) and scripts/make_demo_data.py (synthetic),
so both produce exactly the same schema — see docs/tape_schema.md.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .bands import BANDS
from .signal import bandpass, resample_to

SNIPPET_BANDS = ["delta", "theta", "alpha", "beta"]


def r(x, nd: int = 3):
    """Round floats (recursively) so tapes stay small."""
    if isinstance(x, dict):
        return {k: r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return None
        return round(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    return x


def snippet(signal_uv: np.ndarray, fs: float, channels: Sequence[str], *,
            fs_out: float = 50.0, seconds: float | None = 10.0,
            band_channel: int = 0) -> dict:
    """Downsampled waveform for the scrolling display plus band-filtered layers.

    Band layers are computed at the source rate, then downsampled, for the
    first (or ``band_channel``) channel only — enough for the BAND overlay.
    """
    x = np.atleast_2d(np.asarray(signal_uv, dtype=np.float64))
    if seconds is not None:
        n = min(x.shape[-1], int(round(seconds * fs)))
        x = x[:, :n]
    out = {"fs": fs_out, "signals": {}, "bands": {}}
    for i, ch in enumerate(channels):
        out["signals"][ch] = r(resample_to(x[i], fs, fs_out).tolist(), 1)
    nyq = fs / 2
    for b in SNIPPET_BANDS:
        lo, hi = BANDS[b]
        if hi >= nyq * 0.95:
            continue
        filt = bandpass(x[band_channel], fs, lo, hi, order=3)
        out["bands"][b] = r(resample_to(filt, fs, fs_out).tolist(), 1)
    return out


def pick_snippet_indices(labels: Sequence, per_label: int, rng: np.random.Generator,
                         extra_every: int | None = None) -> list[int]:
    """A few representative windows per label, plus an even spread for scrubbing."""
    labels = list(labels)
    chosen: set[int] = set()
    for lab in sorted({l for l in labels if l is not None}, key=str):
        idx = [i for i, l in enumerate(labels) if l == lab]
        k = min(per_label, len(idx))
        chosen.update(rng.choice(idx, size=k, replace=False).tolist())
    if extra_every:
        chosen.update(range(0, len(labels), extra_every))
    return sorted(chosen)


def radar(rel_means: dict[str, float], coh_means: dict[str, float],
          cohort_max: dict[str, float] | None = None, bands: Sequence[str] = ()) -> dict:
    """Neural-sync radar: relative band power (cohort-normalised) and coherence, both 0–1."""
    bands = list(bands) or list(rel_means.keys())
    power = []
    for b in bands:
        v = rel_means.get(b, 0.0)
        denom = (cohort_max or {}).get(b) or 1.0
        power.append(min(1.0, v / denom))
    sync = [min(1.0, max(0.0, coh_means.get(b, 0.0))) for b in bands]
    return {"axes": bands, "power": r(power), "sync": r(sync)}
