# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Export BD tapes for the web app
# MAGIC Turns gold tables + out-of-fold predictions into the small JSON files the BD editor plays (`docs/tape_schema.md`).
# MAGIC Uses the **same builders** as `scripts/make_demo_data.py`, so real and demo tapes are interchangeable.
# MAGIC
# MAGIC Then copy the folder into the repo's `app/data/` and deploy:
# MAGIC ```bash
# MAGIC databricks fs cp -r dbfs:/Volumes/braindance/gold/tapes ./app/data --overwrite
# MAGIC ```

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json
import os
from datetime import datetime, timezone

import numpy as np
from pyspark.sql import functions as F

from braindance.export import pick_snippet_indices, r, radar, snippet
from braindance.labels import SLEEP_STAGES, quadrant
from braindance.signal import unflatten
from braindance.sources.met import SYMMETRIC_PAIRS, TARGET_CHANNELS

N_SLEEP_TAPES = 8        # nights
N_EMO_PER_QUADRANT = 6   # songs per quadrant
REL_BANDS = ["delta", "theta", "alpha", "sigma", "beta"]
EMO_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
rng = np.random.default_rng(42)
os.makedirs(EXPORT_VOLUME, exist_ok=True)

def signals_for(source, subject, keys):
    """Fetch flat signals for one subject's (session, epoch_idx) pairs from silver and reshape.

    Subject is part of the key: MET session numbers (1..20) repeat for every participant.
    Baseline windows have negative epoch_idx, so music windows never collide with them.
    """
    want = spark.createDataFrame(keys, "session string, epoch_idx int")
    rows = (spark.table(t(SCHEMA_SILVER, "epochs"))
            .where((F.col("source") == source) & (F.col("subject") == subject) & (F.col("phase") != "baseline"))
            .join(F.broadcast(want), ["session", "epoch_idx"])
            .select("session", "epoch_idx", "fs", "channels", "n_samples", "signal").collect())
    return {(x.session, x.epoch_idx): (unflatten(x.signal, x.n_samples), x.fs, list(x.channels)) for x in rows}

def write(obj, name):
    with open(f"{EXPORT_VOLUME}/{name}", "w") as f:
        json.dump(obj, f, separators=(",", ":"))

tapes = []

# COMMAND ----------

# MAGIC %md ### Sleep tapes

# COMMAND ----------

sf = spark.table(t(SCHEMA_GOLD, "sleep_features"))
rel_cols = [f"Fpz_Cz__{b}_rel" for b in REL_BANDS] + [f"Pz_Oz__{b}_rel" for b in REL_BANDS]
coh_cols = [f"coh__Fpz_Cz__Pz_Oz__{b}" for b in REL_BANDS]
sdf = (sf.select("subject", "session", "epoch_idx", "t_start", "stage", "is_artifact", *rel_cols, *coh_cols)
       .join(spark.table(t(SCHEMA_GOLD, "sleep_predictions"))
             .select("session", "epoch_idx", "pred", "pred_raw", "confidence"), ["session", "epoch_idx"], "left")
       .toPandas())
cohort_max = {b: max(sdf[f"Fpz_Cz__{b}_rel"].mean(), sdf[f"Pz_Oz__{b}_rel"].mean()) * 1.6 for b in REL_BANDS}
idx_of = {s: i for i, s in enumerate(SLEEP_STAGES)}

sessions = sorted(sdf.session.unique())
chosen = sorted(rng.choice(sessions, size=min(N_SLEEP_TAPES, len(sessions)), replace=False))
for sess in chosen:
    g = sdf[sdf.session == sess].sort_values("epoch_idx").reset_index(drop=True)
    g = g[g.stage.notna()].reset_index(drop=True)
    snip_pos = pick_snippet_indices(g.stage.tolist(), 2, rng, extra_every=60)
    sig = signals_for("sleep_edf", g.subject[0], [(sess, int(g.epoch_idx[i])) for i in snip_pos])
    t0 = g.t_start.min()
    tape = {
        "id": f"sleep-{sess}", "kind": "sleep", "subject": g.subject[0], "session": sess,
        "title": f"{g.subject[0]} · night {sess[-1]}", "epoch_s": SLEEP_EPOCH_S, "stages": SLEEP_STAGES,
        "channels": ["Fpz_Cz", "Pz_Oz"],
        "epochs": {
            "t": r(((g.t_start - t0) / 60).tolist(), 1),
            "true": [idx_of[s] for s in g.stage],
            "pred": [idx_of.get(p, -1) if isinstance(p, str) else -1 for p in g.pred],
            "pred_raw": [idx_of.get(p, -1) if isinstance(p, str) else -1 for p in g.pred_raw],
            "conf": r(g.confidence.fillna(0).tolist(), 2),
            "artifact": g.is_artifact.astype(int).tolist(),
            "rel": {b: r(g[f"Fpz_Cz__{b}_rel"].tolist(), 3) for b in REL_BANDS},
        },
        "snippets": [{"epoch": int(i), **snippet(*sig[(sess, int(g.epoch_idx[i]))][:2], ["Fpz_Cz", "Pz_Oz"],
                                                 fs_out=50, seconds=10)}
                     for i in snip_pos if (sess, int(g.epoch_idx[i])) in sig],
        "radar": radar({b: float((g[f"Fpz_Cz__{b}_rel"].mean() + g[f"Pz_Oz__{b}_rel"].mean()) / 2) for b in REL_BANDS},
                       {b: float(g[f"coh__Fpz_Cz__Pz_Oz__{b}"].mean()) for b in REL_BANDS}, cohort_max, REL_BANDS),
    }
    write(tape, f"{tape['id']}.json")
    tapes.append({k: tape[k] for k in ("id", "kind", "subject", "session", "title")} | {"file": f"{tape['id']}.json"})
print(f"{len(tapes)} sleep tapes")

# COMMAND ----------

# MAGIC %md ### Emotion tapes

# COMMAND ----------

if spark.catalog.tableExists(t(SCHEMA_GOLD, "emotion_predictions")):
    pair_names = [f"{a}__{b}" for a, b in SYMMETRIC_PAIRS]
    ef_cols = (["subject", "session", "epoch_idx", "t_start", "is_artifact", "quadrant", "d__asym__F3__F4__alpha"]
               + [f"{c}__{b}_rel" for c in TARGET_CHANNELS for b in EMO_BANDS]
               + [f"coh__{p}__{b}" for p in pair_names for b in EMO_BANDS])
    ef = spark.table(t(SCHEMA_GOLD, "emotion_features"))
    edf = (ef.select(*[c for c in ef_cols if c in ef.columns])
           .join(spark.table(t(SCHEMA_GOLD, "emotion_predictions"))
                 .select("subject", "session", "epoch_idx", "p_valence", "p_arousal"),
                 ["subject", "session", "epoch_idx"], "left").toPandas())
    ratings = spark.table(t(SCHEMA_BRONZE, "met_ratings")).toPandas()
    emo_max = {b: max(edf[f"{c}__{b}_rel"].mean() for c in TARGET_CHANNELS if f"{c}__{b}_rel" in edf) * 1.6
               for b in EMO_BANDS}
    songs = edf.dropna(subset=["quadrant"]).drop_duplicates(["subject", "session"])
    picked = (songs.groupby("quadrant", group_keys=False)
              .apply(lambda q: q.sample(min(N_EMO_PER_QUADRANT, len(q)), random_state=1)))
    n_emo = 0
    for s in picked.itertuples():
        g = edf[(edf.subject == s.subject) & (edf.session == s.session)].sort_values("epoch_idx").reset_index(drop=True)
        rt = ratings[(ratings.subject == s.subject) & (ratings.session == s.session)].iloc[0]
        snip_pos = list(range(0, len(g), 6))
        sig = signals_for("met", s.subject, [(s.session, int(g.epoch_idx[i])) for i in snip_pos])
        pv, pa = float(g.p_valence.mean()), float(g.p_arousal.mean())
        snippets = []
        for i in snip_pos:
            key = (s.session, int(g.epoch_idx[i]))
            if key not in sig:
                continue
            x, fs, chans = sig[key]
            pick = [chans.index("F3"), chans.index("F4")]
            snippets.append({"window": int(i), **snippet(x[pick], fs, ["F3", "F4"], fs_out=62.5, seconds=4)})
        tape = {
            "id": f"emo-{s.subject}-{s.session}", "kind": "emotion", "subject": s.subject, "session": s.session,
            "title": f"{s.subject} · song {s.session}", "window_s": MET_WIN_S, "step_s": MET_STEP_S,
            "channels": ["F3", "F4"],
            "rating": {"valence": float(rt.valence), "arousal": float(rt.arousal), "quadrant": s.quadrant},
            "windows": {
                "t": r(g.t_start.tolist(), 1), "p_valence": r(g.p_valence.tolist(), 3),
                "p_arousal": r(g.p_arousal.tolist(), 3), "faa": r(g["d__asym__F3__F4__alpha"].tolist(), 3),
                "artifact": g.is_artifact.astype(int).tolist(),
                "rel": {b: r(g[[f"F3__{b}_rel", f"F4__{b}_rel"]].mean(axis=1).tolist(), 3) for b in EMO_BANDS},
            },
            "prediction": {"p_valence": round(pv, 3), "p_arousal": round(pa, 3),
                           "quadrant": quadrant(int(pv >= 0.5), int(pa >= 0.5))},
            "snippets": snippets,
            "radar": radar({b: float(np.mean([g[f"{c}__{b}_rel"].mean() for c in TARGET_CHANNELS if f"{c}__{b}_rel" in g]))
                            for b in EMO_BANDS},
                           {b: float(np.mean([g[f"coh__{p}__{b}"].mean() for p in pair_names if f"coh__{p}__{b}" in g]))
                            for b in EMO_BANDS}, emo_max, EMO_BANDS),
        }
        write(tape, f"{tape['id']}.json")
        tapes.append({k: tape[k] for k in ("id", "kind", "subject", "session", "title")} | {"file": f"{tape['id']}.json"})
        n_emo += 1
    print(f"{n_emo} emotion tapes")

# COMMAND ----------

metrics = {"demo": False}
for part in ("sleep", "emotion"):
    p = f"{EXPORT_VOLUME}/_metrics_{part}.json"
    if os.path.exists(p):
        metrics[part] = json.load(open(p))
with open(f"{EXPORT_VOLUME}/metrics.json", "w") as f:
    json.dump(r(metrics, 4), f, indent=1)

manifest = {"demo": False, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "Sleep-EDF Expanded (PhysioNet) + MET (OpenNeuro)", "metrics": "metrics.json", "tapes": tapes}
with open(f"{EXPORT_VOLUME}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=1)

total = sum(os.path.getsize(f"{EXPORT_VOLUME}/{x['file']}") for x in tapes)
print(f"{len(tapes)} tapes · {total / 1e6:.1f} MB → {EXPORT_VOLUME}")
