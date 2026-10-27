"""Frequency band definitions.

Sigma (sleep spindles) overlaps alpha/beta on purpose: it is a separate
physiological marker, not a partition of the spectrum. Relative powers are
therefore computed against the total 0.5–(upper limit) power, not the sum of
bands.
"""

from __future__ import annotations

BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 15.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

BAND_ORDER = ["delta", "theta", "alpha", "sigma", "beta", "gamma"]


def usable_bands(fs: float, lowpass_hz: float) -> list[str]:
    """Bands whose upper edge sits safely below both Nyquist and the low-pass.

    Sleep-EDF EEG is 100 Hz and we low-pass at 40 Hz, so gamma is dropped
    there; MET resampled to 250 Hz keeps it.
    """
    ceiling = min(fs / 2.0, lowpass_hz)
    return [b for b in BAND_ORDER if BANDS[b][1] <= ceiling + 1e-9]
