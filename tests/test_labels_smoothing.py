import numpy as np

from braindance import labels as L
from braindance.smoothing import transition_matrix, viterbi


def test_rk_to_aasm_merges_3_and_4():
    assert L.rk_to_aasm("Sleep stage 3") == "N3"
    assert L.rk_to_aasm("Sleep stage 4") == "N3"
    assert L.rk_to_aasm("Sleep stage R") == "REM"
    assert L.rk_to_aasm("Sleep stage ?") is None
    assert L.rk_to_aasm("Movement time") is None


def test_hypnogram_expansion_and_bounds():
    ep = L.hypnogram_to_epochs([0, 90, 150], [90, 60, 300],
                               ["Sleep stage W", "Movement time", "Sleep stage 2"],
                               recording_s=240)
    assert [e["epoch_idx"] for e in ep] == list(range(8))
    assert [e["stage"] for e in ep][:3] == ["W"] * 3
    assert ep[3]["movement"] and ep[3]["stage"] is None
    assert ep[-1]["t_start"] + 30 <= 240


def test_trim_wake_keeps_margin_only():
    stages = ["W"] * 200 + ["N2"] * 10 + ["W"] * 200
    ep = [{"t_start": i * 30.0, "stage": s, "movement": False} for i, s in enumerate(stages)]
    kept = L.trim_wake(ep, margin_s=30 * 60)
    n_wake = sum(e["stage"] == "W" for e in kept)
    assert n_wake == 2 * 60 and len(kept) == 130


def test_rating_binarisation_and_quadrant():
    assert L.rating_to_binary(7) == 1
    assert L.rating_to_binary(2) == 0
    assert L.rating_to_binary(5) is None
    assert L.quadrant(1, 0) == "HVLA"
    assert L.quadrant(None, 1) is None


def test_viterbi_removes_single_epoch_flicker():
    true = np.array([2] * 20 + [3] * 20)
    trans = transition_matrix([true], n_states=5, alpha=0.1)
    prior = np.full(5, 0.2)
    probs = np.full((40, 5), 0.05)
    probs[np.arange(40), true] = 0.8
    probs[10] = [0.05, 0.05, 0.3, 0.05, 0.55]  # one noisy REM blip inside N2
    probs[10] /= probs[10].sum()
    raw = probs.argmax(1)
    smooth = viterbi(probs, trans, prior)
    assert raw[10] == 4 and smooth[10] == 2
    assert (smooth == true).all()
