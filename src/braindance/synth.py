"""Synthetic EEG for unit tests and the offline demo tapes.

NOT real brain data. Each sleep stage / emotion quadrant gets a caricature
of its textbook spectral signature over pink noise, so the pipeline and the
web app can be exercised before (or without) downloading the datasets.
Anything produced here is flagged ``demo: true`` downstream.
"""

from __future__ import annotations

import numpy as np


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spec = np.fft.rfft(white) / np.sqrt(f)
    x = np.fft.irfft(spec, n)
    return x / x.std()


def _osc(t, f, amp, rng, jitter=0.05):
    phase = rng.uniform(0, 2 * np.pi)
    drift = 1 + jitter * np.sin(2 * np.pi * rng.uniform(0.02, 0.1) * t)
    return amp * np.sin(2 * np.pi * f * drift * t + phase)


def sleep_epoch(stage: str, fs: float, seconds: float, rng: np.random.Generator,
                n_channels: int = 2) -> np.ndarray:
    n = int(fs * seconds)
    t = np.arange(n) / fs
    chans = []
    # Real stages overlap: eyes-open wake has little alpha, N1 keeps some, REM
    # looks like N1. Randomised gains keep the caricature from being trivially separable.
    g = rng.uniform(0.6, 1.4)
    eyes_open = stage == "W" and rng.random() < 0.35
    for c in range(n_channels):
        x = rng.uniform(8, 14) * pink_noise(n, rng)
        if stage == "W":
            a = 4 if eyes_open else (18 if c == 1 else 10)
            x += _osc(t, rng.uniform(9.5, 11), a * g, rng)
            x += _osc(t, rng.uniform(18, 24), 6 * g, rng)
        elif stage == "N1":
            x += _osc(t, rng.uniform(5, 7), 11 * g, rng)
            x += _osc(t, 10, rng.uniform(2, 9) * g, rng)
        elif stage == "N2":
            x += _osc(t, rng.uniform(5, 7), 12 * g, rng)
            for _ in range(rng.integers(0, 3)):        # spindles (sometimes none)
                c0 = rng.uniform(1, seconds - 1)
                env = np.exp(-((t - c0) ** 2) / (2 * 0.25 ** 2))
                x += 25 * env * np.sin(2 * np.pi * 13 * t)
            if rng.random() < 0.6:                     # K-complex
                c0 = rng.uniform(2, seconds - 2)
                x += -60 * np.exp(-((t - c0) ** 2) / (2 * 0.15 ** 2))
        elif stage == "N3":
            x += _osc(t, rng.uniform(0.8, 2.0), rng.uniform(30, 60) * g, rng)
            x += _osc(t, rng.uniform(2, 3), 20 * g, rng)
        elif stage == "REM":
            x += _osc(t, rng.uniform(4.5, 6.5), 11 * g, rng)
            x += _osc(t, rng.uniform(8.5, 10.5), rng.uniform(0, 6) * g, rng)
            x += _osc(t, rng.uniform(15, 22), 4 * g, rng)
        chans.append(x)
    return np.vstack(chans)


def hypnogram(n_epochs: int, rng: np.random.Generator) -> list[str]:
    """A plausible night: wake, descent, ~4–5 cycles with deeper N3 early, REM later."""
    seq = ["W"] * rng.integers(10, 25)
    cycles = 5
    for k in range(cycles):
        seq += ["N1"] * rng.integers(2, 6)
        seq += ["N2"] * rng.integers(15, 30)
        seq += ["N3"] * max(0, int(rng.integers(20, 40) * (1 - k / cycles)))
        seq += ["N2"] * rng.integers(5, 15)
        seq += ["REM"] * int(rng.integers(5, 12) * (1 + k / 2))
        if rng.random() < 0.4:
            seq += ["W"] * rng.integers(1, 4)
    seq += ["W"] * rng.integers(10, 25)
    if len(seq) >= n_epochs:
        return seq[:n_epochs]
    return seq + ["W"] * (n_epochs - len(seq))


def emotion_session(valence_high: int, arousal_high: int, fs: float, rng: np.random.Generator,
                    channels: list[str], pre_s: float = 5.0, music_s: float = 120.0,
                    post_s: float = 5.0) -> np.ndarray:
    """8-channel caricature: valence moves frontal alpha asymmetry, arousal moves beta/alpha."""
    n = int(fs * (pre_s + music_s + post_s))
    t = np.arange(n) / fs
    music = (t >= pre_s) & (t < pre_s + music_s)
    ramp = np.clip((t - pre_s) / 20.0, 0, 1) * music
    out = []
    for ch in channels:
        x = 6 * pink_noise(n, rng)
        alpha = 10.0
        is_left = ch.endswith(("1", "3", "7"))
        if valence_high:      # more left activity -> less left alpha
            alpha *= (1 - 0.35 * ramp) if is_left else (1 + 0.15 * ramp)
        else:
            alpha *= (1 + 0.15 * ramp) if is_left else (1 - 0.35 * ramp)
        beta = 3.0 * (1 + (0.9 if arousal_high else -0.3) * ramp)
        gamma = 1.2 * (1 + (0.7 if arousal_high else 0.0) * ramp)
        x += alpha * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6.3))
        x += beta * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 6.3))
        x += gamma * np.sin(2 * np.pi * 38 * t + rng.uniform(0, 6.3))
        x += 4 * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 6.3))
        out.append(x)
    return np.vstack(out)
