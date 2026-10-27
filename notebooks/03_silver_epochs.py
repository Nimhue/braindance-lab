# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Silver — epochs and labels
# MAGIC The windowing step. Every recording is read once, filtered as a **continuous** signal, then cut into windows:
# MAGIC
# MAGIC | | Sleep-EDF | MET |
# MAGIC |---|---|---|
# MAGIC | window | 30 s, aligned to the hypnogram | 4 s, 2 s step, across the 120 s of music |
# MAGIC | rate | 100 Hz (native) | 1000 → 250 Hz (polyphase, anti-aliased) |
# MAGIC | filter | 0.5–40 Hz band-pass | 50 Hz notch + 0.5–45 Hz band-pass |
# MAGIC | extra | wake trimmed to ±30 min around sleep | one 4 s **baseline** window from the pre-music segment |
# MAGIC
# MAGIC **Output:** `silver.epochs` — one row per window, all channels, signal stored flat (channel-major) with `n_samples`.
# MAGIC Artefacts are *flagged, not deleted*: gold filters them, the app shows them as glitches.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import types as T

EPOCH_SCHEMA = T.StructType([
    T.StructField("source", T.StringType()),
    T.StructField("subject", T.StringType()),
    T.StructField("session", T.StringType()),
    T.StructField("epoch_idx", T.IntegerType()),
    T.StructField("t_start", T.DoubleType()),
    T.StructField("phase", T.StringType()),
    T.StructField("fs", T.DoubleType()),
    T.StructField("channels", T.ArrayType(T.StringType())),
    T.StructField("n_samples", T.IntegerType()),
    T.StructField("signal", T.ArrayType(T.FloatType())),
    T.StructField("stage", T.StringType()),
    T.StructField("movement", T.BooleanType()),
    T.StructField("clipped", T.BooleanType()),
    T.StructField("flatline", T.BooleanType()),
    T.StructField("is_artifact", T.BooleanType()),
])
COLS = [f.name for f in EPOCH_SCHEMA.fields]

# Big rows (a 30 s × 2 ch epoch ≈ 24 KB): keep Arrow batches small.
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "256")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sleep-EDF → 30 s epochs

# COMMAND ----------

_sleep_kw = dict(channels=SLEEP_EDF_CHANNELS, epoch_s=SLEEP_EPOCH_S, high=SLEEP_LOWPASS,
                 wake_margin_s=SLEEP_WAKE_MARGIN_S)

def sleep_to_epochs(pdf: pd.DataFrame) -> pd.DataFrame:
    from braindance.sources import sleep_edf
    out = []
    for r in pdf.itertuples():
        out += sleep_edf.read_recording(r.psg_path, r.hypnogram_path, recording_id=r.recording_id,
                                        subject=int(r.subject), night=int(r.night), **_sleep_kw)
    return pd.DataFrame(out, columns=COLS)

files = spark.table(t(SCHEMA_BRONZE, "sleep_edf_files"))
sleep_epochs = (files.repartition(files.count(), "recording_id")
                .groupBy("recording_id").applyInPandas(sleep_to_epochs, EPOCH_SCHEMA))

# COMMAND ----------

# MAGIC %md
# MAGIC ### MET → 4 s windows (+ baseline)

# COMMAND ----------

_met_kw = dict(fs_out=MET_FS_OUT, win_s=MET_WIN_S, step_s=MET_STEP_S, high=MET_LOWPASS,
               mains_hz=MET_MAINS_HZ)
_layout = MET_LAYOUT

def met_to_windows(pdf: pd.DataFrame) -> pd.DataFrame:
    from braindance.sources import met
    out = []
    for r in pdf.to_dict("records"):
        r = {k: (None if v in ("None", "nan") else v) for k, v in r.items()}
        out += met.read_recording(r, _layout, **_met_kw)
    return pd.DataFrame(out, columns=COLS)

met_frames = []
if spark.catalog.tableExists(t(SCHEMA_BRONZE, "met_files")):
    mf = spark.table(t(SCHEMA_BRONZE, "met_files"))
    met_frames.append(mf.repartition(mf.count(), "path").groupBy("path")
                      .applyInPandas(met_to_windows, EPOCH_SCHEMA))
else:
    print("bronze.met_files not found — run 02_bronze_met first. Continuing with Sleep-EDF only.")

# COMMAND ----------

epochs = sleep_epochs
for f in met_frames:
    epochs = epochs.unionByName(f)

(epochs.withColumn("ingested_at", F.current_timestamp())
 .write.mode("overwrite").option("overwriteSchema", "true")
 .partitionBy("source").saveAsTable(t(SCHEMA_SILVER, "epochs")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data-quality contract
# MAGIC Delta `CHECK` constraints make bad rows fail the *next* write instead of silently landing.
# MAGIC The query below is the human-readable version: artefact rate per subject, and the class balance after wake trimming.

# COMMAND ----------

tbl = t(SCHEMA_SILVER, "epochs")
for name, expr in {
    "positive_fs": "fs > 0",
    "signal_shape": "size(signal) = size(channels) * n_samples",
    "met_has_no_stage": "source <> 'met' OR stage IS NULL",
    "valid_stage": "stage IS NULL OR stage IN ('W', 'N1', 'N2', 'N3', 'REM')",
    "known_phase": "phase IN ('sleep', 'baseline', 'music')",
}.items():
    spark.sql(f"ALTER TABLE {tbl} DROP CONSTRAINT IF EXISTS {name}")
    spark.sql(f"ALTER TABLE {tbl} ADD CONSTRAINT {name} CHECK ({expr})")

display(spark.sql(f"""
  SELECT source, subject,
         COUNT(*) AS windows,
         ROUND(100 * AVG(CAST(is_artifact AS INT)), 1) AS pct_artifact,
         ROUND(100 * AVG(CAST(clipped AS INT)), 1)     AS pct_clipped,
         ROUND(100 * AVG(CAST(flatline AS INT)), 1)    AS pct_flat
  FROM {tbl} GROUP BY source, subject ORDER BY source, subject
"""))

display(spark.sql(f"""
  SELECT stage, COUNT(*) AS epochs, ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
  FROM {tbl} WHERE source = 'sleep_edf' GROUP BY stage ORDER BY epochs DESC
"""))
# Typical shape after trimming: N2 largest (~35–45%), then W, REM, N3; N1 smallest (~5–10%).

# COMMAND ----------

# MAGIC %md
# MAGIC ### `silver.epoch_labels` — one long label table for both worlds
# MAGIC `label_type` is `sleep_stage`, `valence`, `arousal` or `quadrant`. Emotion labels are **per song**:
# MAGIC every window of a session inherits the same rating. That's the softness we'll be honest about in 06.

# COMMAND ----------

sleep_labels = (spark.table(tbl).where("source = 'sleep_edf' AND stage IS NOT NULL")
                .select("source", "subject", "session", "epoch_idx",
                        F.lit("sleep_stage").alias("label_type"), F.col("stage").alias("label_value")))

labels = sleep_labels
if spark.catalog.tableExists(t(SCHEMA_BRONZE, "met_ratings")):
    from braindance.sources import met
    rated = met.label_rows(spark.table(t(SCHEMA_BRONZE, "met_ratings")).toPandas(), RATING_THRESHOLD)
    rated = rated.astype({"valence_high": "object", "arousal_high": "object"})
    rated = rated.where(pd.notna(rated), None)
    long = []
    for r in rated.itertuples():
        for lt, v in (("valence", r.valence_high), ("arousal", r.arousal_high), ("quadrant", r.quadrant)):
            if v is not None:
                long.append((r.subject, r.session, lt, str(int(v)) if lt != "quadrant" else v))
    sess_labels = spark.createDataFrame(long, "subject string, session string, label_type string, label_value string")
    met_windows = spark.table(tbl).where("source = 'met' AND phase = 'music'").select(
        "source", "subject", "session", "epoch_idx")
    labels = labels.unionByName(met_windows.join(sess_labels, ["subject", "session"])
                                .select("source", "subject", "session", "epoch_idx", "label_type", "label_value"))

    unmatched = met_windows.select("subject", "session").distinct().join(
        sess_labels.select("subject", "session").distinct(), ["subject", "session"], "left_anti").count()
    print(f"MET sessions with windows but no rating: {unmatched}  (should be 0 — else fix MET_RATINGS keys)")

labels.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(t(SCHEMA_SILVER, "epoch_labels"))
display(spark.sql(f"""SELECT source, label_type, label_value, COUNT(*) n
                     FROM {t(SCHEMA_SILVER, 'epoch_labels')} GROUP BY ALL ORDER BY ALL"""))
