"""MET — Music-Induced EEG Dataset (OpenNeuro, BIDS).

What the dataset description promises:
  * 20 participants x 20 music-listening sessions
  * 130 s per session: 5 s pre-music, 120 s music, 5 s post-music
  * post-session self-ratings: valence 1–9, arousal 1–9 (plus liking, familiarity)
  * EGI HydroCel GSN 128-channel net at 1000 Hz

What we do NOT know until the files are inspected: how sessions map to BIDS
runs, and where the ratings live (events.tsv, beh/, or a phenotype file).
So this adapter is deliberately configurable — run notebook 02's inspection
cell first, then set ``MET_LAYOUT`` and ``MET_RATINGS`` in 00_config.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import labels as L
from ..signal import artifact_flags, bandpass, notch, resample_to, sliding_windows

# 8 channels used by the dataset authors' reduced release.
TARGET_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T7", "T8"]

# Approximate 10-20 equivalents on the EGI HydroCel GSN-128 net.
# VERIFY against the dataset's channels.tsv / electrodes.tsv before trusting.
HYDROCEL128_TO_1020 = {
    "Fp1": "E22", "Fp2": "E9", "F3": "E24", "F4": "E124",
    "F7": "E33", "F8": "E122", "T7": "E45", "T8": "E108",
}

SYMMETRIC_PAIRS = [("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8"), ("T7", "T8")]

EEG_EXTENSIONS = ("_eeg.vhdr", "_eeg.set", "_eeg.edf", "_eeg.bdf", "_eeg.fif")


@dataclass
class MetLayout:
    """How sessions are laid out on disk.

    mode="per_file": every EEG file is one 130 s session.
    mode="events":   one continuous file per subject/run; music onsets are
                     rows of *_events.tsv whose ``trial_type`` matches
                     ``music_onset_regex``.
    """
    mode: str = "per_file"
    pre_s: float = 5.0
    music_s: float = 120.0
    music_onset_regex: str = r"music|song|stim"
    session_from: str = "run"   # BIDS entity naming the session in per_file mode


@dataclass
class MetRatings:
    """Where valence/arousal ratings live and what the columns are called."""
    path_glob: str = ""                 # e.g. ".../phenotype/ratings.tsv"; empty -> events.tsv
    subject_col: str = "participant_id"
    session_col: str = "run"
    valence_col: str = "valence"
    arousal_col: str = "arousal"
    class_col: str | None = None        # optional ready-made HVHA/... column
    extra: dict = field(default_factory=dict)


def norm_subject(s) -> str:
    """'sub-01' / '1' / 1 -> 'M01'. Both windows and ratings use this key."""
    s = str(s).replace("sub-", "")
    return f"M{int(s):02d}" if s.isdigit() else f"M{s}"


def norm_session(s) -> str:
    """'01' / 1 / '1.0' -> '1' so BIDS run entities and rating columns join."""
    s = str(s)
    try:
        return str(int(float(s)))
    except ValueError:
        return s


_BIDS_ENTITY = re.compile(r"(sub|ses|task|run|acq)-([A-Za-z0-9]+)")


def parse_bids_name(path: str) -> dict:
    return dict(_BIDS_ENTITY.findall(os.path.basename(path)))


def discover(root: str) -> list[dict]:
    """Every EEG recording under a BIDS root, with its events sidecar if present."""
    found = []
    for ext in EEG_EXTENSIONS:
        for p in glob.glob(os.path.join(root, "sub-*", "**", f"*{ext}"), recursive=True):
            ents = parse_bids_name(p)
            ev = p[: -len(ext)] + "_events.tsv"
            found.append({
                "path": p, "subject": ents.get("sub"), "ses": ents.get("ses"),
                "task": ents.get("task"), "run": ents.get("run"),
                "events_path": ev if os.path.exists(ev) else None,
            })
    return sorted(found, key=lambda r: r["path"])


def resolve_channels(available: list[str], wanted: list[str] = TARGET_CHANNELS,
                     fallback: dict[str, str] = HYDROCEL128_TO_1020) -> dict[str, str]:
    """Map target 10-20 names -> actual channel names in the file.

    Tries case-insensitive 10-20 names first (some releases relabel), then
    the HydroCel lookup. Missing channels are left out, not invented.
    """
    lower = {c.lower(): c for c in available}
    out = {}
    for w in wanted:
        if w.lower() in lower:
            out[w] = lower[w.lower()]
        elif fallback.get(w) and fallback[w].lower() in lower:
            out[w] = lower[fallback[w].lower()]
    return out


def windows_for_session(data_uv: np.ndarray, fs_in: float, channels: list[str], *,
                        subject: str, session: str, fs_out: float = 250.0,
                        win_s: float = 4.0, step_s: float = 2.0, pre_s: float = 5.0,
                        music_s: float = 120.0, mains_hz: float = 50.0,
                        low: float = 0.5, high: float = 45.0,
                        amp_uv: float = 150.0) -> list[dict]:
    """Clean one 130 s session and cut baseline + music windows.

    ``data_uv`` must start at the session start (pre-music segment first).
    The baseline is the 4 s window inside the 5 s pre-music segment; it is
    used later to normalise every music window of the same session.
    """
    x = resample_to(np.asarray(data_uv, dtype=np.float64), fs_in, fs_out)
    x = notch(x, fs_out, mains_hz)
    x = bandpass(x, fs_out, low, high).astype(np.float32)
    rows = []
    phases = [("baseline", 0.0, pre_s), ("music", pre_s, pre_s + music_s)]
    for phase, a, b in phases:
        for w in sliding_windows(x.shape[-1], fs_out, win_s, step_s, offset_s=a, stop_s=b):
            seg = x[:, w.start_sample:w.stop_sample]
            flags = artifact_flags(seg, amp_uv=amp_uv)
            rows.append({
                "source": "met",
                "subject": norm_subject(subject),
                "session": norm_session(session),
                "epoch_idx": w.idx if phase == "music" else -1 - w.idx,
                "t_start": w.start_sample / fs_out,
                "phase": phase,
                "fs": float(fs_out),
                "channels": list(channels),
                "n_samples": int(seg.shape[-1]),
                "signal": seg.ravel().tolist(),   # channel-major, flat (Arrow-friendly)
                "stage": None,
                "movement": False,
                "clipped": flags["clipped"],
                "flatline": flags["flatline"],
                "is_artifact": flags["is_artifact"],
            })
    return rows


def session_starts_from_events(events: pd.DataFrame, layout: MetLayout) -> list[tuple[int, float]]:
    """[(session_number, session_start_seconds)] from a BIDS events table."""
    col = "trial_type" if "trial_type" in events.columns else events.columns[-1]
    mask = events[col].astype(str).str.contains(layout.music_onset_regex, case=False, regex=True)
    onsets = events.loc[mask, "onset"].astype(float).tolist()
    return [(i + 1, max(0.0, t - layout.pre_s)) for i, t in enumerate(onsets)]


def read_recording(rec: dict, layout: MetLayout, **kw) -> list[dict]:
    """Read one BIDS EEG file into window rows (one or many sessions)."""
    import mne

    # Lazy open, pick the 8 target channels, *then* load: a continuous
    # 128-channel 1000 Hz file would otherwise be gigabytes in worker memory.
    raw = mne.io.read_raw(rec["path"], preload=False, verbose="error")
    mapping = resolve_channels(raw.ch_names)
    if len(mapping) < 2:
        raise ValueError(f"Could not resolve target channels in {rec['path']}; "
                         f"first names: {raw.ch_names[:10]}")
    names = list(mapping.keys())
    raw.pick([mapping[n] for n in names]).load_data(verbose="error")
    data = raw.get_data(picks=[mapping[n] for n in names], units="uV")  # explicit order
    fs = float(raw.info["sfreq"])
    total_s = layout.pre_s + layout.music_s

    rows: list[dict] = []
    if layout.mode == "per_file":
        session = rec.get(layout.session_from) or rec.get("run") or "1"
        rows += windows_for_session(data[:, : int(total_s * fs)], fs, names,
                                    subject=rec["subject"], session=session,
                                    pre_s=layout.pre_s, music_s=layout.music_s, **kw)
    elif layout.mode == "events":
        if not rec.get("events_path"):
            raise ValueError(f"events mode needs an events.tsv next to {rec['path']}")
        events = pd.read_csv(rec["events_path"], sep="\t")
        for n, start in session_starts_from_events(events, layout):
            a, b = int(start * fs), int((start + total_s) * fs)
            if b > data.shape[-1]:
                continue
            sess = f"{rec.get('run') or 'r'}_{n:02d}"
            rows += windows_for_session(data[:, a:b], fs, names, subject=rec["subject"],
                                        session=sess, pre_s=layout.pre_s,
                                        music_s=layout.music_s, **kw)
    else:
        raise ValueError(f"unknown layout mode {layout.mode!r}")
    return rows


def load_ratings(cfg: MetRatings, bids_root: str) -> pd.DataFrame:
    """Normalise ratings to columns: subject, session, valence, arousal, quadrant_given."""
    paths = glob.glob(cfg.path_glob, recursive=True) if cfg.path_glob else \
        glob.glob(os.path.join(bids_root, "sub-*", "**", "*_events.tsv"), recursive=True)
    frames = []
    for p in paths:
        sep = "\t" if p.endswith(".tsv") else ","
        df = pd.read_csv(p, sep=sep)
        if cfg.subject_col not in df.columns:
            df[cfg.subject_col] = parse_bids_name(p).get("sub")
        if cfg.session_col not in df.columns:
            df[cfg.session_col] = parse_bids_name(p).get("run")
        frames.append(df)
    if not frames:
        raise FileNotFoundError("no ratings files matched — check MET_RATINGS.path_glob")
    df = pd.concat(frames, ignore_index=True)
    needed = [cfg.valence_col, cfg.arousal_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"ratings columns {missing} not found; columns are {list(df.columns)}")
    out = pd.DataFrame({
        "subject": df[cfg.subject_col].astype(str).str.replace("sub-", "", regex=False),
        "session": df[cfg.session_col].map(norm_session),
        "valence": pd.to_numeric(df[cfg.valence_col], errors="coerce"),
        "arousal": pd.to_numeric(df[cfg.arousal_col], errors="coerce"),
    })
    out["quadrant_given"] = df[cfg.class_col].astype(str) if cfg.class_col else None
    out = out.dropna(subset=["valence", "arousal"])
    out["subject"] = out["subject"].map(norm_subject)
    return out.drop_duplicates(["subject", "session"])


def label_rows(ratings: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """Add binary valence/arousal and a quadrant per session."""
    r = ratings.copy()
    # Nullable Int64 keeps "ambiguous" as <NA> instead of a float NaN.
    r["valence_high"] = r["valence"].map(lambda v: L.rating_to_binary(v, threshold)).astype("Int64")
    r["arousal_high"] = r["arousal"].map(lambda v: L.rating_to_binary(v, threshold)).astype("Int64")
    r["quadrant"] = [L.quadrant(v, a) for v, a in zip(r["valence_high"], r["arousal_high"])]
    return r
