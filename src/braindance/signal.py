"""Continuous-signal cleaning and windowing.

Filtering happens on the *continuous* recording before it is cut into
windows; filtering each 30 s window separately would add edge artefacts at
every boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps


def bandpass(x: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis."""
    nyq = fs / 2.0
    high = min(high, nyq * 0.95)
    sos = sps.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sps.sosfiltfilt(sos, x, axis=-1)


def notch(x: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Mains-hum notch (50 Hz in Europe / China). No-op if freq >= Nyquist."""
    if freq >= fs / 2.0:
        return x
    b, a = sps.iirnotch(freq, q, fs)
    return sps.filtfilt(b, a, x, axis=-1)


def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resampling (anti-aliased) along the last axis."""
    if fs_in == fs_out:
        return x
    from fractions import Fraction

    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return sps.resample_poly(x, frac.numerator, frac.denominator, axis=-1)


@dataclass(frozen=True)
class Window:
    idx: int
    start_sample: int
    stop_sample: int


def sliding_windows(n_samples: int, fs: float, win_s: float, step_s: float,
                    offset_s: float = 0.0, stop_s: float | None = None) -> list[Window]:
    """Fixed-length windows inside [offset_s, stop_s). Partial tails are dropped."""
    win = int(round(win_s * fs))
    step = int(round(step_s * fs))
    start = int(round(offset_s * fs))
    stop = n_samples if stop_s is None else min(n_samples, int(round(stop_s * fs)))
    out, i = [], 0
    while start + win <= stop:
        out.append(Window(i, start, start + win))
        start += step
        i += 1
    return out


def artifact_flags(epoch: np.ndarray, amp_uv: float = 200.0, flat_uv: float = 0.5,
                   max_flat_frac: float = 0.5) -> dict[str, bool]:
    """Cheap, explainable artefact checks on one window (channels x samples, µV).

    * ``clipped``: any absolute sample above ``amp_uv``
    * ``flatline``: a channel whose sample-to-sample change is below
      ``flat_uv`` for more than ``max_flat_frac`` of the window
    """
    epoch = np.atleast_2d(epoch)
    clipped = bool(np.any(np.abs(epoch) > amp_uv))
    diffs = np.abs(np.diff(epoch, axis=-1))
    flat_frac = (diffs < flat_uv).mean(axis=-1)
    flatline = bool(np.any(flat_frac > max_flat_frac))
    return {"clipped": clipped, "flatline": flatline, "is_artifact": clipped or flatline}


def unflatten(signal, n_samples: int) -> np.ndarray:
    """Flat channel-major array (as stored in silver.epochs) -> (channels, samples)."""
    a = np.asarray(signal, dtype=np.float64)
    return a.reshape(-1, int(n_samples))
