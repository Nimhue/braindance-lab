# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Bronze — MET (Music-Induced EEG)
# MAGIC Download the BIDS dataset from OpenNeuro into a Volume, **inspect its layout**, and record the manifest and raw ratings.
# MAGIC
# MAGIC ⚠️ Run the inspection cells before anything downstream. The adapter in `braindance.sources.met`
# MAGIC is configurable because the session/rating layout must be confirmed from the files themselves —
# MAGIC then set `MET_LAYOUT` and `MET_RATINGS` in `00_config`.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import glob
import os

import openneuro
import pandas as pd

from braindance.sources import met

assert MET_OPENNEURO_ID != "dsXXXXXX", "Set MET_OPENNEURO_ID in 00_config first"
root = f"{VOLUME}/met/{MET_OPENNEURO_ID}"
os.makedirs(root, exist_ok=True)

# Top-level metadata first (cheap), then only the selected subjects.
openneuro.download(dataset=MET_OPENNEURO_ID, target_dir=root,
                   include=["dataset_description.json", "participants.tsv", "README*",
                            "*.json", "phenotype/*", "stimuli/*.tsv"])
openneuro.download(dataset=MET_OPENNEURO_ID, target_dir=root,
                   include=[f"sub-{s}/*" for s in MET_SUBJECTS])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspection — confirm the layout before trusting the adapter
# MAGIC Answer three questions from the output below:
# MAGIC 1. **Session ↔ file**: one 130 s file per song (`per_file`), or a long file with music onsets in `events.tsv` (`events`)?
# MAGIC 2. **Ratings**: in `*_events.tsv`, in `beh/`, or in `phenotype/`? What are the column names?
# MAGIC 3. **Channels**: 10-20 names (`F3`) or HydroCel (`E24`)? Check the mapping in `met.HYDROCEL128_TO_1020`.

# COMMAND ----------

print(open(f"{root}/dataset_description.json").read()[:1500])
for readme in glob.glob(f"{root}/README*"):
    print(open(readme).read()[:3000])

recs = met.discover(root)
print(f"\n{len(recs)} EEG recordings found; first five:")
for r in recs[:5]:
    print(r)

# COMMAND ----------

sub0 = f"{root}/sub-{MET_SUBJECTS[0]}"
for p in sorted(glob.glob(f"{sub0}/**/*", recursive=True))[:60]:
    print(p.replace(root, ""))

for pattern in ["**/*_events.tsv", "**/*_channels.tsv", "**/*_beh.tsv", "phenotype/*.tsv"]:
    hits = sorted(glob.glob(f"{root}/{pattern}", recursive=True))
    if hits:
        print(f"\n=== {pattern} → {hits[0].replace(root, '')}")
        display(pd.read_csv(hits[0], sep="\t").head(12))

# COMMAND ----------

import mne

raw = mne.io.read_raw(recs[0]["path"], preload=False, verbose="error")
print(f"fs={raw.info['sfreq']} Hz · {len(raw.ch_names)} channels · {raw.times[-1]:.1f} s")
print("first channel names:", raw.ch_names[:12])
print("resolved targets:", met.resolve_channels(raw.ch_names))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Manifest + raw ratings (run once the layout is confirmed)

# COMMAND ----------

manifest = pd.DataFrame(recs)
manifest["bytes"] = manifest["path"].map(os.path.getsize)
spark.createDataFrame(manifest.astype({"ses": str, "run": str, "task": str, "events_path": str})) \
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(t(SCHEMA_BRONZE, "met_files"))

ratings = met.load_ratings(MET_RATINGS, root)
spark.createDataFrame(ratings.astype({"quadrant_given": str})) \
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(t(SCHEMA_BRONZE, "met_ratings"))

display(spark.sql(f"""
  SELECT subject, COUNT(*) AS sessions, ROUND(AVG(valence), 2) AS mean_valence,
         ROUND(AVG(arousal), 2) AS mean_arousal
  FROM {t(SCHEMA_BRONZE, 'met_ratings')} GROUP BY subject ORDER BY subject
"""))
