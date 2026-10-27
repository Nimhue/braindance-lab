"""Per-window spectral and time-domain features.

One call to :func:`epoch_features` turns a (channels x samples) window into a
flat ``dict[str, float]`` — the shape a pandas UDF can return as a row.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np
from scipy import signal as sps
from scipy.integrate import trapezoid

from .bands import BANDS, usable_bands

_EPS = 1e-12


def clean_channel(name: str) -> str:
    """'EEG Fpz-Cz' -> 'Fpz_Cz'; 'F3' -> 'F3'. Safe for column names."""
    name = re.sub(r"^(EEG|EOG|EMG)\s+", "", name.strip(), flags=re.I)
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


def welch_psd(x: np.ndarray, fs: float, seg_s: float) -> tuple[np.ndarray, np.ndarray]:
    nper = max(8, min(x.shape[-1], int(round(seg_s * fs))))
    return sps.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, axis=-1)


def _integrate(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs < hi)
    if mask.sum() < 2:
        return np.zeros(psd.shape[:-1])
    return trapezoid(psd[..., mask], freqs[mask], axis=-1)


def band_powers(freqs: np.ndarray, psd: np.ndarray, bands: Iterable[str],
                ceiling: float) -> dict[str, np.ndarray]:
    """Absolute band power, relative power and total power per channel."""
    total = _integrate(freqs, psd, 0.5, ceiling) + _EPS
    out: dict[str, np.ndarray] = {"total": total}
    for b in bands:
        lo, hi = BANDS[b]
        p = _integrate(freqs, psd, lo, hi)
        out[f"{b}_abs"] = p
        out[f"{b}_rel"] = p / total
    return out


def spectral_entropy(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normalised Shannon entropy of the PSD in [lo, hi): 0 = one peak, 1 = flat."""
    mask = (freqs >= lo) & (freqs < hi)
    p = psd[..., mask]
    p = p / (p.sum(axis=-1, keepdims=True) + _EPS)
    h = -(p * np.log(p + _EPS)).sum(axis=-1)
    return h / np.log(mask.sum())


def spectral_edge(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float,
                  q: float = 0.95) -> np.ndarray:
    """Frequency below which ``q`` of the power in [lo, hi) lies."""
    mask = (freqs >= lo) & (freqs < hi)
    f, p = freqs[mask], psd[..., mask]
    c = np.cumsum(p, axis=-1)
    c = c / (c[..., -1:] + _EPS)
    idx = np.argmax(c >= q, axis=-1)
    return f[idx]


def hjorth(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hjorth activity, mobility, complexity along the last axis."""
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)
    var_x, var_dx, var_ddx = x.var(axis=-1), dx.var(axis=-1), ddx.var(axis=-1)
    mobility = np.sqrt(var_dx / (var_x + _EPS))
    complexity = np.sqrt(var_ddx / (var_dx + _EPS)) / (mobility + _EPS)
    return var_x, mobility, complexity


def band_coherence(x: np.ndarray, y: np.ndarray, fs: float, seg_s: float,
                   bands: Iterable[str]) -> dict[str, float]:
    """Mean magnitude-squared coherence between two channels inside each band.

    This is the "neural sync" number: 1 = the two sites move in lock-step at
    that rhythm, 0 = unrelated.
    """
    nper = max(8, min(x.shape[-1], int(round(seg_s * fs))))
    f, cxy = sps.coherence(x, y, fs=fs, nperseg=nper, noverlap=nper // 2)
    out = {}
    for b in bands:
        lo, hi = BANDS[b]
        m = (f >= lo) & (f < hi)
        out[b] = float(cxy[m].mean()) if m.any() else float("nan")
    return out


def epoch_features(signals: np.ndarray, channels: Sequence[str], fs: float,
                   lowpass_hz: float, seg_s: float,
                   coherence_pairs: Sequence[tuple[str, str]] = (),
                   asymmetry_pairs: Sequence[tuple[str, str]] = ()) -> dict[str, float]:
    """All features for one window.

    ``asymmetry_pairs`` are (left, right) channel names. Asymmetry is
    ``ln(alpha_right) - ln(alpha_left)``: because alpha power is inversely
    related to cortical activity, a positive value means relatively *more*
    left-frontal activity — the classic approach-motivation / positive
    valence marker.
    """
    signals = np.atleast_2d(np.asarray(signals, dtype=np.float64))
    names = [clean_channel(c) for c in channels]
    bands = usable_bands(fs, lowpass_hz)
    ceiling = min(fs / 2.0, lowpass_hz)

    freqs, psd = welch_psd(signals, fs, seg_s)
    bp = band_powers(freqs, psd, bands, ceiling)
    ent = spectral_entropy(freqs, psd, 0.5, ceiling)
    sef = spectral_edge(freqs, psd, 0.5, ceiling)
    act, mob, comp = hjorth(signals)

    row: dict[str, float] = {}
    for i, ch in enumerate(names):
        row[f"{ch}__log_total"] = float(np.log10(bp["total"][i]))
        for b in bands:
            row[f"{ch}__{b}_rel"] = float(bp[f"{b}_rel"][i])
            row[f"{ch}__{b}_log"] = float(np.log10(bp[f"{b}_abs"][i] + _EPS))
        row[f"{ch}__entropy"] = float(ent[i])
        row[f"{ch}__sef95"] = float(sef[i])
        row[f"{ch}__hjorth_act"] = float(np.log10(act[i] + _EPS))
        row[f"{ch}__hjorth_mob"] = float(mob[i])
        row[f"{ch}__hjorth_comp"] = float(comp[i])
        if "theta" in bands and "beta" in bands:
            row[f"{ch}__theta_beta"] = float(bp["theta_abs"][i] / (bp["beta_abs"][i] + _EPS))
        if "beta" in bands and "alpha" in bands:
            row[f"{ch}__beta_alpha"] = float(bp["beta_abs"][i] / (bp["alpha_abs"][i] + _EPS))

    idx = {c: i for i, c in enumerate(names)}
    for a, b in coherence_pairs:
        a, b = clean_channel(a), clean_channel(b)
        if a in idx and b in idx:
            coh = band_coherence(signals[idx[a]], signals[idx[b]], fs, seg_s, bands)
            for band, v in coh.items():
                row[f"coh__{a}__{b}__{band}"] = v

    for left, right in asymmetry_pairs:
        left, right = clean_channel(left), clean_channel(right)
        if left in idx and right in idx and "alpha" in bands:
            l_alpha = bp["alpha_abs"][idx[left]] + _EPS
            r_alpha = bp["alpha_abs"][idx[right]] + _EPS
            row[f"asym__{left}__{right}__alpha"] = float(np.log(r_alpha) - np.log(l_alpha))
            if "beta" in bands:
                l_beta = bp["beta_abs"][idx[left]] + _EPS
                r_beta = bp["beta_abs"][idx[right]] + _EPS
                row[f"asym__{left}__{right}__beta"] = float(np.log(r_beta) - np.log(l_beta))
    return row
