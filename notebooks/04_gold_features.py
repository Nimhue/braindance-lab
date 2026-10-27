# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Gold — spectral features
# MAGIC Each window → Welch PSD → a flat feature vector:
# MAGIC
# MAGIC * **band power** (absolute log₁₀ and relative): δ 0.5–4 · θ 4–8 · α 8–13 · σ 12–15 · β 13–30 · γ 30–45 Hz (γ only for MET at 250 Hz)
# MAGIC * **spectral entropy**, **95% spectral edge**, **Hjorth** activity / mobility / complexity, θ/β and β/α ratios
# MAGIC * **neural sync** — magnitude-squared coherence per band between channel pairs
# MAGIC * **asymmetry** (MET) — ln α(right) − ln α(left) for F3/F4, F7/F8, Fp1/Fp2
# MAGIC
# MAGIC Then two model-ready tables:
# MAGIC * `gold.sleep_features` — + **context**: the same features for the ±2 neighbouring epochs, and time-of-night
# MAGIC * `gold.emotion_features` — + **baseline-normalised** deltas (music window − that session's own pre-music baseline)

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from braindance.sources.met import SYMMETRIC_PAIRS

FEATURE_CFG = {
    "sleep_edf": dict(lowpass_hz=SLEEP_LOWPASS, seg_s=4.0,
                      coherence_pairs=[tuple(SLEEP_EDF_CHANNELS)], asymmetry_pairs=[]),
    "met": dict(lowpass_hz=MET_LOWPASS, seg_s=2.0, coherence_pairs=SYMMETRIC_PAIRS,
                asymmetry_pairs=[("F3", "F4"), ("F7", "F8"), ("Fp1", "Fp2")]),
}
KEYS = ["source", "subject", "session", "epoch_idx", "t_start", "phase", "is_artifact"]

out_schema = T.StructType(
    [f for f in spark.table(t(SCHEMA_SILVER, "epochs")).schema.fields if f.name in KEYS]
    + [T.StructField("features", T.MapType(T.StringType(), T.DoubleType()))])

def featurise(batches):
    from braindance.features import epoch_features
    from braindance.signal import unflatten
    for pdf in batches:
        feats = []
        for r in pdf.itertuples():
            x = unflatten(r.signal, r.n_samples)
            feats.append(epoch_features(x, list(r.channels), r.fs, **FEATURE_CFG[r.source]))
        out = pdf[KEYS].copy()
        out["features"] = feats
        yield out

spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "512")
(spark.table(t(SCHEMA_SILVER, "epochs"))
 .mapInPandas(featurise, out_schema)
 .write.mode("overwrite").option("overwriteSchema", "true")
 .partitionBy("source").saveAsTable(t(SCHEMA_GOLD, "epoch_features")))

# COMMAND ----------

def feature_names(source: str) -> list[str]:
    row = (spark.table(t(SCHEMA_GOLD, "epoch_features")).where(F.col("source") == source)
           .select(F.map_keys("features").alias("k")).first())
    return sorted(row["k"]) if row else []

def widen(df, names):
    return df.select(*[c for c in df.columns if c != "features"],
                     *[F.col("features")[n].alias(n) for n in names])

# COMMAND ----------

# MAGIC %md
# MAGIC ### `gold.sleep_features` — with context
# MAGIC A scorer never looks at one 30 s page in isolation; they flip back and forward. The ±2 lag/lead columns give the model the same courtesy.

# COMMAND ----------

sleep_names = feature_names("sleep_edf")
if sleep_names:
    base = widen(spark.table(t(SCHEMA_GOLD, "epoch_features")).where("source = 'sleep_edf'"), sleep_names)
    w = Window.partitionBy("session").orderBy("epoch_idx")
    context_cols = [n for n in sleep_names if n.endswith(("_rel", "__entropy", "__sef95"))
                    or n.startswith("coh__")]
    ctx = []
    for k in range(1, SLEEP_CONTEXT + 1):
        for n in context_cols:
            ctx.append(F.lag(n, k).over(w).alias(f"{n}__lag{k}"))
            ctx.append(F.lead(n, k).over(w).alias(f"{n}__lead{k}"))
    start = F.min("t_start").over(Window.partitionBy("session"))
    span = F.max("t_start").over(Window.partitionBy("session")) - start
    sleep_wide = base.select("*", *ctx,
                             ((F.col("t_start") - start) / 3600).alias("hours_into_recording"),
                             ((F.col("t_start") - start) / span).alias("frac_of_recording"))
    stage = (spark.table(t(SCHEMA_SILVER, "epoch_labels")).where("label_type = 'sleep_stage'")
             .select("subject", "session", "epoch_idx", F.col("label_value").alias("stage")))
    (sleep_wide.join(stage, ["subject", "session", "epoch_idx"], "left")
     .write.mode("overwrite").option("overwriteSchema", "true")
     .saveAsTable(t(SCHEMA_GOLD, "sleep_features")))
    print(f"sleep_features: {len(sleep_names)} base + {len(ctx)} context columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ### `gold.emotion_features` — baseline-normalised
# MAGIC Resting EEG differs a lot between people (skull thickness, electrode impedance, just *being* someone).
# MAGIC Subtracting each session's own 4 s pre-music baseline turns "how much alpha" into "how much alpha **changed** when the song began".

# COMMAND ----------

met_names = feature_names("met")
if met_names:
    ef = widen(spark.table(t(SCHEMA_GOLD, "epoch_features")).where("source = 'met'"), met_names)
    baseline = (ef.where("phase = 'baseline'").groupBy("subject", "session")
                .agg(*[F.avg(n).alias(f"b__{n}") for n in met_names]))
    music = ef.where("phase = 'music'").join(baseline, ["subject", "session"], "left")
    deltas = [(F.col(n) - F.col(f"b__{n}")).alias(f"d__{n}") for n in met_names]
    labels = (spark.table(t(SCHEMA_SILVER, "epoch_labels")).where("source = 'met'")
              .groupBy("subject", "session", "epoch_idx").pivot("label_type", ["valence", "arousal", "quadrant"])
              .agg(F.first("label_value")))
    (music.select(*[c for c in music.columns if not c.startswith("b__")], *deltas)
     .join(labels, ["subject", "session", "epoch_idx"], "left")
     .withColumn("valence", F.col("valence").cast("int"))
     .withColumn("arousal", F.col("arousal").cast("int"))
     .write.mode("overwrite").option("overwriteSchema", "true")
     .saveAsTable(t(SCHEMA_GOLD, "emotion_features")))
    print(f"emotion_features: {len(met_names)} raw + {len(deltas)} baseline-delta columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sanity check — does physiology show up?
# MAGIC Before any ML: N3 should be delta-heavy, wake alpha-heavy at Pz-Oz (eyes-closed alpha is occipital), N2 sigma-elevated (spindles).
# MAGIC If this table looks wrong, something upstream is wrong — no model will fix it.

# COMMAND ----------

if sleep_names:
    display(spark.sql(f"""
      SELECT stage, COUNT(*) AS n,
             ROUND(AVG(Fpz_Cz__delta_rel), 3) AS fpz_delta,
             ROUND(AVG(Fpz_Cz__theta_rel), 3) AS fpz_theta,
             ROUND(AVG(Pz_Oz__alpha_rel), 3)  AS pz_alpha,
             ROUND(AVG(Fpz_Cz__sigma_rel), 3) AS fpz_sigma,
             ROUND(AVG(Fpz_Cz__beta_rel), 3)  AS fpz_beta,
             ROUND(AVG(coh__Fpz_Cz__Pz_Oz__delta), 3) AS sync_delta
      FROM {t(SCHEMA_GOLD, 'sleep_features')}
      WHERE NOT is_artifact AND stage IS NOT NULL
      GROUP BY stage ORDER BY array_position(array('W','N1','N2','N3','REM'), stage)
    """))
