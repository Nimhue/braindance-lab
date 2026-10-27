import numpy as np
import pandas as pd
import pytest

from braindance import synth
from braindance.export import pick_snippet_indices, radar, snippet
from braindance.signal import unflatten
from braindance.sources import met, sleep_edf


def test_physionet_listing_and_pairing():
    html = ('<a href="SC4001E0-PSG.edf">x</a><a href="SC4001EC-Hypnogram.edf">y</a>'
            '<a href="SC4002E0-PSG.edf">z</a><a href="SC4011EH-Hypnogram.edf">w</a>'
            '<a href="SC4011E0-PSG.edf">v</a>')
    names = sleep_edf.list_remote_files(html)
    pairs = sleep_edf.pair_files(names)
    assert [p["recording_id"] for p in pairs] == ["SC4001", "SC4011"]
    assert pairs[1]["subject"] == 1 and pairs[1]["night"] == 1
    assert pairs[0]["hypnogram"] == "SC4001EC-Hypnogram.edf"


def test_sleep_epochs_from_synthetic_night():
    rng = np.random.default_rng(3)
    stages = ["W"] * 80 + ["N2"] * 20 + ["N3"] * 20 + ["REM"] * 10 + ["W"] * 80
    rk = {"W": "Sleep stage W", "N2": "Sleep stage 2", "N3": "Sleep stage 4", "REM": "Sleep stage R"}
    data = np.hstack([synth.sleep_epoch(s, 100, 30, rng) for s in stages])
    rows = sleep_edf.epochs_from_arrays(
        data, 100, ["EEG Fpz-Cz", "EEG Pz-Oz"],
        [i * 30 for i in range(len(stages))], [30] * len(stages), [rk[s] for s in stages],
        recording_id="SC4001", subject=0, night=1, wake_margin_s=10 * 30)
    assert len(rows) == 10 + 50 + 10
    assert rows[0]["subject"] == "S00" and rows[0]["n_samples"] == 3000
    assert unflatten(rows[0]["signal"], rows[0]["n_samples"]).shape == (2, 3000)
    assert {r["stage"] for r in rows} == {"W", "N2", "N3", "REM"}


def test_edf_round_trip(tmp_path):
    """Write a synthetic PSG + hypnogram as EDF and read it back via the real reader."""
    mne = pytest.importorskip("mne")
    pytest.importorskip("edfio")
    rng = np.random.default_rng(4)
    stages = ["W"] * 4 + ["N2"] * 6 + ["W"] * 4
    data = np.hstack([synth.sleep_epoch(s, 100, 30, rng) for s in stages]) * 1e-6
    info = mne.create_info(["EEG Fpz-Cz", "EEG Pz-Oz"], 100, "eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    psg = tmp_path / "SC4001E0-PSG.edf"
    mne.export.export_raw(psg, raw, overwrite=True, verbose="error")
    hyp_raw = mne.io.RawArray(np.zeros((1, data.shape[1])), mne.create_info(["x"], 100, "misc"),
                              verbose="error")
    hyp_raw.set_annotations(mne.Annotations([0, 120, 300], [120, 180, 120],
                                            ["Sleep stage W", "Sleep stage 2", "Sleep stage W"]))
    hyp = tmp_path / "SC4001EC-Hypnogram.edf"
    mne.export.export_raw(hyp, hyp_raw, overwrite=True, verbose="error")
    rows = sleep_edf.read_recording(str(psg), str(hyp), recording_id="SC4001", subject=0,
                                    night=1, wake_margin_s=60)
    assert [r["stage"] for r in rows] == ["W", "W"] + ["N2"] * 6 + ["W", "W"]
    assert np.abs(np.array(rows[3]["signal"])).max() < 500  # read back in µV


def test_met_windows_count_and_baseline():
    rng = np.random.default_rng(5)
    x = synth.emotion_session(1, 1, 1000, rng, met.TARGET_CHANNELS)
    rows = met.windows_for_session(x, 1000, met.TARGET_CHANNELS, subject="3", session="07")
    base = [r for r in rows if r["phase"] == "baseline"]
    music = [r for r in rows if r["phase"] == "music"]
    assert len(base) == 1 and len(music) == 59
    assert rows[0]["fs"] == 250 and music[0]["n_samples"] == 1000
    assert len(music[0]["signal"]) == 8 * 1000
    assert music[0]["subject"] == "M03"


def test_resolve_channels_prefers_1020_then_hydrocel():
    assert met.resolve_channels(["fp1", "E9", "E24"])  == {"Fp1": "fp1", "Fp2": "E9", "F3": "E24"}


def test_met_events_layout():
    ev = pd.DataFrame({"onset": [3.0, 140.0, 300.0], "trial_type": ["music_01", "rating", "music_02"]})
    starts = met.session_starts_from_events(ev, met.MetLayout(mode="events"))
    assert starts == [(1, 0.0), (2, 295.0)]


def test_ratings_loading(tmp_path):
    p = tmp_path / "ratings.tsv"
    pd.DataFrame({"participant_id": ["sub-01", "sub-01", "sub-02"], "run": [1, 2, 1],
                  "valence": [8, 5, 2], "arousal": [7, 3, 9]}).to_csv(p, sep="\t", index=False)
    cfg = met.MetRatings(path_glob=str(p))
    lab = met.label_rows(met.load_ratings(cfg, str(tmp_path)))
    assert lab["subject"].tolist() == ["M01", "M01", "M02"]
    q = lab["quadrant"].tolist()
    assert q[0] == "HVHA" and pd.isna(q[1]) and q[2] == "LVHA"   # rating 5 = ambiguous
    assert pd.isna(lab["arousal_high"].iloc[1]) is False and pd.isna(lab["valence_high"].iloc[1])


def test_export_helpers():
    rng = np.random.default_rng(6)
    sig = synth.sleep_epoch("N2", 100, 30, rng)
    s = snippet(sig, 100, ["Fpz_Cz", "Pz_Oz"], fs_out=50, seconds=10)
    assert len(s["signals"]["Fpz_Cz"]) == 500 and set(s["bands"]) == {"delta", "theta", "alpha", "beta"}
    idx = pick_snippet_indices(["W", "W", "N2", "N2", "N3"], 1, rng)
    assert len(idx) == 3
    rd = radar({"alpha": 0.2, "delta": 0.6}, {"alpha": 0.5, "delta": 0.9},
               {"alpha": 0.4, "delta": 0.6}, ["delta", "alpha"])
    assert rd["power"] == [1.0, 0.5] and rd["sync"] == [0.9, 0.5]


def test_met_read_recording_from_bids_fif(tmp_path):
    """Exercise the real reader on a tiny fake BIDS file with HydroCel channel names."""
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(7)
    names = [met.HYDROCEL128_TO_1020[c] for c in met.TARGET_CHANNELS] + ["E1", "E2"]
    x = synth.emotion_session(1, 0, 1000, rng, met.TARGET_CHANNELS + ["F3", "F4"])
    raw = mne.io.RawArray(x * 1e-6, mne.create_info(names, 1000, "eeg"), verbose="error")
    d = tmp_path / "sub-02" / "eeg"
    d.mkdir(parents=True)
    path = d / "sub-02_task-music_run-03_eeg.fif"
    raw.save(path, verbose="error")
    recs = met.discover(str(tmp_path))
    assert recs[0]["subject"] == "02" and recs[0]["run"] == "03"
    rows = met.read_recording(recs[0], met.MetLayout(mode="per_file"))
    assert rows[0]["subject"] == "M02" and rows[0]["session"] == "3"
    assert rows[0]["channels"] == met.TARGET_CHANNELS and len(rows) == 60
