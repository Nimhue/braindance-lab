import numpy as np
import pytest

from braindance import features as F
from braindance.bands import usable_bands

FS = 100.0


def sine(f, seconds=30, fs=FS, amp=10.0, phase=0.0):
    t = np.arange(int(seconds * fs)) / fs
    return amp * np.sin(2 * np.pi * f * t + phase)


def test_usable_bands_drops_gamma_for_sleep_edf():
    assert "gamma" not in usable_bands(100, 40)
    assert "gamma" in usable_bands(250, 45)


@pytest.mark.parametrize("freq,band", [(2, "delta"), (6, "theta"), (10, "alpha"), (20, "beta")])
def test_pure_tone_lands_in_its_band(freq, band):
    rng = np.random.default_rng(0)
    x = sine(freq) + 0.5 * rng.standard_normal(3000)
    row = F.epoch_features(x[None], ["EEG Fpz-Cz"], FS, 40, seg_s=4)
    rels = {b: row[f"Fpz_Cz__{b}_rel"] for b in ["delta", "theta", "alpha", "beta"]}
    assert max(rels, key=rels.get) == band
    assert rels[band] > 0.6


def test_entropy_low_for_tone_high_for_noise():
    rng = np.random.default_rng(1)
    tone = F.epoch_features(sine(10)[None], ["A"], FS, 40, 4)["A__entropy"]
    noise = F.epoch_features(rng.standard_normal((1, 3000)), ["A"], FS, 40, 4)["A__entropy"]
    assert tone < 0.3 < 0.8 < noise


def test_coherence_identical_vs_independent():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(3000)
    same = F.epoch_features(np.vstack([a, a]), ["A", "B"], FS, 40, 4, coherence_pairs=[("A", "B")])
    indep = F.epoch_features(np.vstack([a, rng.standard_normal(3000)]), ["A", "B"], FS, 40, 4,
                             coherence_pairs=[("A", "B")])
    assert same["coh__A__B__alpha"] > 0.99
    assert indep["coh__A__B__alpha"] < 0.3


def test_alpha_asymmetry_sign():
    # Less alpha on the left (F3) => positive asymmetry (more left activity)
    left = sine(10, amp=3)
    right = sine(10, amp=12)
    row = F.epoch_features(np.vstack([left, right]), ["F3", "F4"], FS, 40, 4,
                           asymmetry_pairs=[("F3", "F4")])
    assert row["asym__F3__F4__alpha"] > 0


def test_hjorth_mobility_scales_with_frequency():
    _, m_lo, _ = F.hjorth(sine(2))
    _, m_hi, _ = F.hjorth(sine(20))
    assert m_hi > m_lo


def test_clean_channel():
    assert F.clean_channel("EEG Fpz-Cz") == "Fpz_Cz"
    assert F.clean_channel("F3") == "F3"
