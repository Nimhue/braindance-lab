"""Sleep-EDF Expanded (PhysioNet, sleep-cassette subset).

File naming: ``SC4ssNE0-PSG.edf`` with hypnogram ``SC4ssNXX-Hypnogram.edf``
where ``ss`` = subject, ``N`` = night (1 or 2). The two letters before
``-Hypnogram`` are the scorer code and differ per file, so PSG and hypnogram
are paired on the first six characters.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from .. import labels as L
from ..signal import artifact_flags, bandpass

BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
DEFAULT_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]

_PSG_RE = re.compile(r"(SC4\d{3}E\d-PSG\.edf)")
_HYP_RE = re.compile(r"(SC4\d{3}[A-Z]{2}-Hypnogram\.edf)")


def list_remote_files(index_html: str) -> list[str]:
    """Extract PSG + hypnogram filenames from the PhysioNet directory listing."""
    names = set(_PSG_RE.findall(index_html)) | set(_HYP_RE.findall(index_html))
    return sorted(names)


def parse_name(filename: str) -> dict:
    stem = filename.rsplit("/", 1)[-1]
    return {
        "recording_id": stem[:6],          # SC4001
        "subject": int(stem[3:5]),
        "night": int(stem[5]),
        "kind": "hypnogram" if "Hypnogram" in stem else "psg",
    }


def pair_files(filenames: Iterable[str]) -> list[dict]:
    """[{recording_id, subject, night, psg, hypnogram}] — only complete pairs."""
    by_id: dict[str, dict] = {}
    for f in filenames:
        meta = parse_name(f)
        rec = by_id.setdefault(meta["recording_id"], {
            "recording_id": meta["recording_id"], "subject": meta["subject"],
            "night": meta["night"], "psg": None, "hypnogram": None})
        rec[meta["kind"]] = f
    return [r for r in sorted(by_id.values(), key=lambda r: r["recording_id"])
            if r["psg"] and r["hypnogram"]]


def epochs_from_arrays(data_uv: np.ndarray, fs: float, channels: list[str],
                       onsets, durations, descriptions, *, recording_id: str,
                       subject: int, night: int, epoch_s: float = 30.0,
                       low: float = 0.5, high: float = 40.0,
                       wake_margin_s: float = 30 * 60, amp_uv: float = 200.0) -> list[dict]:
    """Filter the continuous signal, cut 30 s epochs, attach stage labels.

    Separated from file reading so it can be tested with synthetic arrays.
    """
    n_samples = data_uv.shape[-1]
    filt = bandpass(data_uv, fs, low, high).astype(np.float32)
    hyp = L.hypnogram_to_epochs(onsets, durations, descriptions, epoch_s,
                                recording_s=n_samples / fs)
    hyp = L.trim_wake(hyp, wake_margin_s)
    win = int(round(epoch_s * fs))
    rows = []
    for e in hyp:
        start = int(round(e["t_start"] * fs))
        seg = filt[:, start:start + win]
        if seg.shape[-1] < win:
            continue
        flags = artifact_flags(seg, amp_uv=amp_uv)
        rows.append({
            "source": "sleep_edf",
            "subject": f"S{subject:02d}",
            "session": f"{recording_id}_n{night}",
            "epoch_idx": e["epoch_idx"],
            "t_start": e["t_start"],
            "phase": "sleep",
            "fs": float(fs),
            "channels": list(channels),
            "n_samples": int(seg.shape[-1]),
                "signal": seg.ravel().tolist(),   # channel-major, flat (Arrow-friendly)
            "stage": e["stage"],
            "movement": e["movement"],
            "clipped": flags["clipped"],
            "flatline": flags["flatline"],
            "is_artifact": flags["is_artifact"] or e["movement"],
        })
    return rows


def read_recording(psg_path: str, hyp_path: str, *, recording_id: str, subject: int,
                   night: int, channels: list[str] = DEFAULT_CHANNELS,
                   epoch_s: float = 30.0, **kw) -> list[dict]:
    """Read one PSG + hypnogram pair from disk (Volume path) into epoch rows."""
    import mne

    raw = mne.io.read_raw_edf(psg_path, include=channels, preload=True, verbose="error")
    fs = float(raw.info["sfreq"])
    data = raw.get_data(picks=channels, units="uV")
    ann = mne.read_annotations(hyp_path)
    return epochs_from_arrays(
        data, fs, channels, ann.onset, ann.duration, ann.description,
        recording_id=recording_id, subject=subject, night=night, epoch_s=epoch_s, **kw)
