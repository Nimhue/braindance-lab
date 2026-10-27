# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Emotion model — and why it's harder
# MAGIC Two binary targets from the post-song self-ratings (1–9 SAM scale, >5 = high, <5 = low, exactly 5 dropped):
# MAGIC **valence** (pleasant ↔ unpleasant) and **arousal** (calm ↔ intense). Their combination is the quadrant (HVHA, HVLA, LVHA, LVLA).
# MAGIC
# MAGIC Why this is a different game from sleep:
# MAGIC * **One label per song**, not per window — all 59 windows inherit the same rating, even if the feeling came and went
# MAGIC * **Self-report** — a feeling, remembered and rounded onto a scale
# MAGIC * **People differ** — the same song, the same rating, very different EEG
# MAGIC
# MAGIC So we report three things side by side: **cross-subject** (can it read a stranger?), **within-subject** (can it read a new song from someone it knows?),
# MAGIC and a **permutation null** (what does a model score when the labels are shuffled?). A result only counts if it clears the null.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from braindance.modeling import emotion_cv, feature_columns, permutation_null

df = spark.table(t(SCHEMA_GOLD, "emotion_features")).where("NOT is_artifact").toPandas()
# Use baseline-normalised deltas + raw asymmetry; raw absolute power mostly encodes "who", not "how they feel".
features = [c for c in feature_columns(df) if c.startswith("d__") or c.startswith("asym__")]
songs = df.drop_duplicates(["subject", "session"])
print(f"{len(df):,} windows · {songs.shape[0]} songs · {df.subject.nunique()} subjects · {len(features)} features")
display(songs.groupby("quadrant").size().rename("songs").to_frame())

# COMMAND ----------

mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').first()[0]}/braindance-emotion")
results = {}
with mlflow.start_run(run_name="hgb_baseline_norm") as run:
    mlflow.log_params({"model": "HistGradientBoosting", "features": "baseline deltas + asymmetry",
                       "n_features": len(features), "threshold": RATING_THRESHOLD})
    for target in ("valence", "arousal"):
        cross = emotion_cv(df, features, target, group_col="subject")
        within = emotion_cv(df, features, target, group_col="session")
        null = permutation_null(df, features, target, n=20, group_col="subject")
        results[target] = {"cross_subject": cross, "within_subject": within, "null": null}
        for name, res in (("cross", cross), ("within", within)):
            for level in ("window", "song"):
                mlflow.log_metrics({f"{target}_{name}_{level}_{k}": v for k, v in res[level].items()
                                    if isinstance(v, float)})
        mlflow.log_metrics({f"{target}_null_mean": null["mean"], f"{target}_null_p95": null["p95"]})
    run_id = run.info.run_id

rows = []
for target, r in results.items():
    for name in ("cross_subject", "within_subject"):
        s = r[name]["song"]
        rows.append({"target": target, "setting": name, "song_bal_acc": s["balanced_accuracy"],
                     "song_auc": s["auc"], "window_bal_acc": r[name]["window"]["balanced_accuracy"],
                     "null_p95": r["null"]["p95"], "clears_null": s["balanced_accuracy"] > r["null"]["p95"]})
display(pd.DataFrame(rows).round(3))

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, target in zip(axes, ("valence", "arousal")):
    r = results[target]
    ax.hist(r["null"]["draws"], bins=10, color="#5b5b7a", label="shuffled labels")
    ax.axvline(r["cross_subject"]["song"]["balanced_accuracy"], color="#ff2a6d", lw=2, label="cross-subject")
    ax.axvline(r["within_subject"]["song"]["balanced_accuracy"], color="#05d9e8", lw=2, label="within-subject")
    ax.set_title(f"{target}: song-level balanced accuracy"); ax.set_xlim(0.2, 0.9)
axes[0].legend(fontsize=8)
fig.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Out-of-fold predictions → `gold.emotion_predictions`
# MAGIC Cross-subject probabilities per window (what the app's circumplex trail plays back), plus the song-level mean.

# COMMAND ----------

frame = results["valence"]["cross_subject"]["frame"].rename(columns={"p": "p_valence", "y": "valence"})
ar = results["arousal"]["cross_subject"]["frame"].rename(columns={"p": "p_arousal", "y": "arousal"})
pred = frame.merge(ar, on=["subject", "session", "epoch_idx"], how="outer")
pred["run_id"] = run_id
spark.createDataFrame(pred).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(t(SCHEMA_GOLD, "emotion_predictions"))

def strip(m):
    return {k: v for k, v in m.items() if k not in ("window_proba", "frame")}

metrics = {"model": "HistGradientBoosting on baseline-normalised features",
           "n_subjects": int(df.subject.nunique()), "n_songs": int(songs.shape[0]),
           "cross_subject": {k: strip(v["cross_subject"]) for k, v in results.items()},
           "within_subject": {k: strip(v["within_subject"]) for k, v in results.items()},
           "null": {k: v["null"] for k, v in results.items()}, "mlflow_run_id": run_id}
with open(f"{EXPORT_VOLUME}/_metrics_emotion.json", "w") as f:
    json.dump(metrics, f)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Which features carry valence? (sanity, not proof)
# MAGIC Frontal alpha asymmetry is the textbook valence marker. If it doesn't separate high/low valence songs at all here, say so in the write-up.

# COMMAND ----------

display(df.groupby("valence")[[c for c in df.columns if c.startswith("d__asym__")]].mean().round(3))
display(df.groupby("arousal")[[c for c in df.columns if c.startswith("d__") and c.endswith("beta_alpha")]].mean().round(3))
