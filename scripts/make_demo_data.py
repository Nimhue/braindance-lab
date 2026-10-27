"""Build SYNTHETIC demo tapes for the web app — no downloads, no Databricks.

Runs the same package code as the notebooks (windowing -> features -> CV ->
smoothing -> tape export) on caricatured synthetic EEG from braindance.synth,
so the app has something to play before the real pipeline has run.

Every file it writes carries "demo": true and the app shows a banner.
The metrics are those of a toy problem and say nothing about real EEG.

    python scripts/make_demo_data.py --out app/data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from braindance import synth  # noqa: E402
from braindance.export import pick_snippet_indices, r, radar, snippet  # noqa: E402
from braindance.features import epoch_features  # noqa: E402
from braindance.labels import SLEEP_STAGES, quadrant  # noqa: E402
from braindance.modeling import emotion_cv, feature_columns, sleep_cv  # noqa: E402
from braindance.signal import bandpass, notch, resample_to, sliding_windows, artifact_flags  # noqa: E402
from braindance.sources.met import SYMMETRIC_PAIRS, TARGET_CHANNELS  # noqa: E402

SLEEP_CH = ["Fpz_Cz", "Pz_Oz"]
REL_BANDS = ["delta", "theta", "alpha", "sigma", "beta"]
EMO_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def sleep_part(rng, n_subjects: int):
    rows, signals = [], {}
    for s in range(n_subjects):
        stages = synth.hypnogram(int(rng.integers(560, 680)), rng)
        sess = f"SC4{s:02d}1_n1"
        for i, st in enumerate(stages):
            x = synth.sleep_epoch(st, 100, 30, rng)
            if rng.random() < 0.015:                      # sprinkle artefacts
                x[:, rng.integers(0, 2000):][:, :400] += 400
            x = bandpass(x, 100, 0.5, 40)
            flags = artifact_flags(x)
            f = epoch_features(x, SLEEP_CH, 100, 40, 4.0, coherence_pairs=[tuple(SLEEP_CH)])
            rows.append({"subject": f"S{s:02d}", "session": sess, "epoch_idx": i, "t_start": i * 30.0,
                         "stage": st, "is_artifact": flags["is_artifact"], **f})
            signals[(sess, i)] = x.astype(np.float32)
    df = pd.DataFrame(rows)
    ctx_cols = [c for c in df.columns if c.endswith(("_rel", "__entropy", "__sef95")) or c.startswith("coh__")]
    g = df.groupby("session")
    extra = {}
    for k in (1, 2):
        for c in ctx_cols:
            extra[f"{c}__lag{k}"] = g[c].shift(k)
            extra[f"{c}__lead{k}"] = g[c].shift(-k)
    df = pd.concat([df, pd.DataFrame(extra)], axis=1)
    start = g["t_start"].transform("min")
    df["hours_into_recording"] = (df["t_start"] - start) / 3600
    df["frac_of_recording"] = (df["t_start"] - start) / g["t_start"].transform(lambda t: t.max() - t.min())
    return df, signals


def emotion_part(rng, n_subjects: int, n_songs: int):
    rows, signals, songs = [], {}, []
    for s in range(n_subjects):
        for k in range(n_songs):
            v_hi, a_hi = int(rng.integers(0, 2)), int(rng.integers(0, 2))
            # self-report is noisy: 20% of ratings disagree with the induced state
            v_rep = v_hi if rng.random() > 0.2 else 1 - v_hi
            a_rep = a_hi if rng.random() > 0.2 else 1 - a_hi
            val = float(rng.integers(6, 10) if v_rep else rng.integers(1, 5))
            aro = float(rng.integers(6, 10) if a_rep else rng.integers(1, 5))
            raw = synth.emotion_session(v_hi, a_hi, 1000, rng, TARGET_CHANNELS)
            x = bandpass(notch(resample_to(raw, 1000, 250), 250, 50), 250, 0.5, 45).astype(np.float32)
            subj, sess = f"M{s + 1:02d}", str(k + 1)
            songs.append({"subject": subj, "session": sess, "valence": val, "arousal": aro,
                          "valence_high": v_rep, "arousal_high": a_rep})
            base_f = None
            for phase, a, b in (("baseline", 0, 5), ("music", 5, 125)):
                for w in sliding_windows(x.shape[-1], 250, 4, 2, a, b):
                    seg = x[:, w.start_sample:w.stop_sample]
                    f = epoch_features(seg, TARGET_CHANNELS, 250, 45, 2.0, coherence_pairs=SYMMETRIC_PAIRS,
                                       asymmetry_pairs=[("F3", "F4"), ("F7", "F8"), ("Fp1", "Fp2")])
                    if phase == "baseline":
                        base_f = f
                        continue
                    d = {f"d__{n}": f[n] - base_f[n] for n in f}
                    rows.append({"subject": subj, "session": sess, "epoch_idx": w.idx,
                                 "t_start": w.start_sample / 250, "is_artifact": False,
                                 "valence": v_rep, "arousal": a_rep, **f, **d})
                    signals[(subj, sess, w.idx)] = seg
    return pd.DataFrame(rows), signals, pd.DataFrame(songs)


def build(out: str, seed: int = 11, n_sleep: int = 5, n_emo: int = 5, n_songs: int = 6):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    print("synthesising sleep nights …")
    sdf, ssig = sleep_part(rng, n_sleep)
    clean = sdf[~sdf["is_artifact"]].reset_index(drop=True)
    cv = sleep_cv(clean, feature_columns(clean), n_splits=min(5, n_sleep), max_iter=60)
    clean["pred_raw"], clean["pred"] = cv.pred_raw, cv.pred_smooth
    clean["conf"] = cv.proba.max(1)

    print("synthesising music sessions …")
    edf, esig, songs = emotion_part(rng, n_emo, n_songs)
    feats = [c for c in feature_columns(edf) if c.startswith("d__") or c.startswith("asym__")]
    res = {t: emotion_cv(edf, feats, t, group_col="subject", max_iter=60, balanced=False) for t in ("valence", "arousal")}
    within = {t: emotion_cv(edf, feats, t, group_col="session", max_iter=60, balanced=False) for t in ("valence", "arousal")}
    for t in ("valence", "arousal"):
        edf[f"p_{t}"] = res[t]["window_proba"]

    tapes = []
    # ---- sleep tapes -------------------------------------------------------
    cohort_max = {b: max(clean[f"{ch}__{b}_rel"].mean() for ch in SLEEP_CH) * 1.6 for b in REL_BANDS}
    for sess, g in sdf.groupby("session"):
        g = g.sort_values("epoch_idx")
        pmap = clean[clean["session"] == sess].set_index("epoch_idx")
        idx = g["epoch_idx"].tolist()
        snip_idx = pick_snippet_indices(g["stage"].tolist(), 2, rng, extra_every=60)
        tape = {
            "id": f"sleep-{sess}", "kind": "sleep", "subject": g["subject"].iloc[0], "session": sess,
            "title": f"{g['subject'].iloc[0]} · night 1", "epoch_s": 30, "stages": SLEEP_STAGES,
            "channels": SLEEP_CH,
            "epochs": {
                "t": [i * 0.5 for i in idx],   # minutes
                "true": [SLEEP_STAGES.index(s) for s in g["stage"]],
                "pred": [int(pmap.at[i, "pred"]) if i in pmap.index else -1 for i in idx],
                "pred_raw": [int(pmap.at[i, "pred_raw"]) if i in pmap.index else -1 for i in idx],
                "conf": r([float(pmap.at[i, "conf"]) if i in pmap.index else 0.0 for i in idx], 2),
                "artifact": [int(a) for a in g["is_artifact"]],
                "rel": {b: r(g[f"Fpz_Cz__{b}_rel"].tolist(), 3) for b in REL_BANDS},
            },
            "snippets": [{"epoch": int(i), **snippet(ssig[(sess, idx[i])], 100, SLEEP_CH, fs_out=50, seconds=10)}
                         for i in snip_idx],
            "radar": radar({b: float(np.mean([g[f"{ch}__{b}_rel"].mean() for ch in SLEEP_CH])) for b in REL_BANDS},
                           {b: float(g[f"coh__Fpz_Cz__Pz_Oz__{b}"].mean()) for b in REL_BANDS},
                           cohort_max, REL_BANDS),
        }
        fn = f"{tape['id']}.json"
        json.dump(tape, open(os.path.join(out, fn), "w"), separators=(",", ":"))
        tapes.append({k: tape[k] for k in ("id", "kind", "subject", "session", "title")} | {"file": fn})

    # ---- emotion tapes -----------------------------------------------------
    pair_names = [f"{a}__{b}" for a, b in SYMMETRIC_PAIRS]
    emo_max = {b: max(edf[f"{ch}__{b}_rel"].mean() for ch in TARGET_CHANNELS) * 1.6 for b in EMO_BANDS}
    for (subj, sess), g in edf.groupby(["subject", "session"]):
        g = g.sort_values("epoch_idx")
        song = songs[(songs.subject == subj) & (songs.session == sess)].iloc[0]
        snip_idx = list(range(0, len(g), 6))
        pv, pa = float(g["p_valence"].mean()), float(g["p_arousal"].mean())
        tape = {
            "id": f"emo-{subj}-{sess}", "kind": "emotion", "subject": subj, "session": sess,
            "title": f"{subj} · song {sess}", "window_s": 4, "step_s": 2,
            "channels": ["F3", "F4"],
            "rating": {"valence": song.valence, "arousal": song.arousal,
                       "quadrant": quadrant(int(song.valence_high), int(song.arousal_high))},
            "windows": {
                "t": r(g["t_start"].tolist(), 1),
                "p_valence": r(g["p_valence"].tolist(), 3),
                "p_arousal": r(g["p_arousal"].tolist(), 3),
                "faa": r(g["d__asym__F3__F4__alpha"].tolist(), 3),
                "artifact": [0] * len(g),
                "rel": {b: r(g[[f"{c}__{b}_rel" for c in ("F3", "F4")]].mean(axis=1).tolist(), 3) for b in EMO_BANDS},
            },
            "prediction": {"p_valence": round(pv, 3), "p_arousal": round(pa, 3),
                           "quadrant": quadrant(int(pv >= 0.5), int(pa >= 0.5))},
            "snippets": [{"window": int(i), **snippet(esig[(subj, sess, int(g['epoch_idx'].iloc[i]))][[2, 3]], 250,
                                                       ["F3", "F4"], fs_out=62.5, seconds=4)}
                         for i in snip_idx],
            "radar": radar({b: float(np.mean([g[f"{c}__{b}_rel"].mean() for c in TARGET_CHANNELS])) for b in EMO_BANDS},
                           {b: float(np.mean([g[f"coh__{p}__{b}"].mean() for p in pair_names])) for b in EMO_BANDS},
                           emo_max, EMO_BANDS),
        }
        fn = f"{tape['id']}.json"
        json.dump(tape, open(os.path.join(out, fn), "w"), separators=(",", ":"))
        tapes.append({k: tape[k] for k in ("id", "kind", "subject", "session", "title")} | {"file": fn})

    def strip(m):
        return {k: v for k, v in m.items() if k not in ("window_proba", "frame")}

    metrics = {
        "demo": True,
        "sleep": {"model": "HistGradientBoosting + HMM/Viterbi", "classes": SLEEP_STAGES,
                  "n_subjects": int(clean.subject.nunique()), "n_epochs": int(len(clean)),
                  "raw": cv.metrics_raw, "smoothed": cv.metrics_smooth,
                  "transition": r(cv.transition.tolist(), 3)},
        "emotion": {"model": "HistGradientBoosting on baseline-normalised features",
                    "n_subjects": int(edf.subject.nunique()),
                    "n_songs": int(edf[["subject", "session"]].drop_duplicates().shape[0]),
                    "cross_subject": {t: strip(res[t]) for t in res},
                    "within_subject": {t: strip(within[t]) for t in within}},
    }
    json.dump(r(metrics, 4), open(os.path.join(out, "metrics.json"), "w"), indent=1)
    manifest = {"demo": True, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": "scripts/make_demo_data.py (synthetic)", "metrics": "metrics.json", "tapes": tapes}
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(f"wrote {len(tapes)} tapes to {out}")
    print("sleep κ raw/smoothed:", round(cv.metrics_raw["kappa"], 3), round(cv.metrics_smooth["kappa"], 3))
    for t in res:
        print(f"{t}: cross-subject song bal-acc {res[t]['song']['balanced_accuracy']:.3f} · "
              f"within {within[t]['song']['balanced_accuracy']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "app", "data"))
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    build(a.out, a.seed)
