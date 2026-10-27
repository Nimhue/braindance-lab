# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze — Sleep-EDF
# MAGIC Land raw PhysioNet Sleep-EDF (sleep-cassette) files in a Volume, untouched, and record what we got.
# MAGIC
# MAGIC | Table | Grain | What |
# MAGIC |---|---|---|
# MAGIC | `bronze.sleep_edf_files` | one row per recording (night) | PSG + hypnogram paths, sizes, download time |
# MAGIC | `bronze.sleep_edf_hypnogram` | one row per annotation run | raw R&K stage runs exactly as scored |
# MAGIC
# MAGIC The EDF signal itself stays in the Volume — the file *is* the bronze record. Windowing happens in silver.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import os
import time
from datetime import datetime, timezone

import requests

from braindance.sources import sleep_edf

dest = f"{VOLUME}/sleep_edf"
os.makedirs(dest, exist_ok=True)

index = requests.get(sleep_edf.BASE_URL, timeout=60)
index.raise_for_status()
pairs = [p for p in sleep_edf.pair_files(sleep_edf.list_remote_files(index.text))
         if p["subject"] in SLEEP_EDF_SUBJECTS]
print(f"{len(pairs)} recordings selected for subjects {SLEEP_EDF_SUBJECTS[0]}–{SLEEP_EDF_SUBJECTS[-1]}")

# COMMAND ----------

def download(name: str) -> tuple[str, int]:
    """Idempotent download: skip files already in the Volume."""
    path = f"{dest}/{name}"
    if not os.path.exists(path):
        for attempt in range(4):
            try:
                with requests.get(sleep_edf.BASE_URL + name, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    tmp = path + ".part"
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                    os.replace(tmp, path)
                break
            except requests.RequestException as e:
                if attempt == 3:
                    raise
                print(f"retry {name}: {e}")
                time.sleep(2 ** attempt)
    return path, os.path.getsize(path)

manifest = []
for p in pairs:
    psg_path, psg_bytes = download(p["psg"])
    hyp_path, _ = download(p["hypnogram"])
    manifest.append({**p, "psg_path": psg_path, "hypnogram_path": hyp_path,
                     "psg_bytes": psg_bytes, "landed_at": datetime.now(timezone.utc)})

files_df = spark.createDataFrame(manifest).drop("psg", "hypnogram")
files_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(t(SCHEMA_BRONZE, "sleep_edf_files"))
display(files_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Raw hypnogram runs
# MAGIC Kept in R&K vocabulary (`Sleep stage 4`, `Movement time`, `Sleep stage ?`) — mapping to AASM happens in silver, so the original scoring is always recoverable.

# COMMAND ----------

import mne

hyp_rows = []
for m in manifest:
    ann = mne.read_annotations(m["hypnogram_path"])
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        hyp_rows.append({"recording_id": m["recording_id"], "subject": m["subject"], "night": m["night"],
                         "onset_s": float(onset), "duration_s": float(dur), "description": str(desc)})

hyp_df = spark.createDataFrame(hyp_rows)
hyp_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(t(SCHEMA_BRONZE, "sleep_edf_hypnogram"))

display(spark.sql(f"""
  SELECT description, COUNT(*) AS runs, ROUND(SUM(duration_s) / 3600, 1) AS hours
  FROM {t(SCHEMA_BRONZE, 'sleep_edf_hypnogram')}
  GROUP BY description ORDER BY hours DESC
"""))
# Expect 'Sleep stage W' to dominate by hours — that's the ~20 h cassette recordings, trimmed in silver.
