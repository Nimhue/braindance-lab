# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Config
# MAGIC Shared settings for every Braindance Lab notebook. Other notebooks start with `%run ./00_config`.
# MAGIC
# MAGIC **What this does**
# MAGIC 1. Defines catalog / schema / volume names and dataset settings
# MAGIC 2. Makes the `braindance` package (in `../src`) importable on the driver *and* on Spark workers
# MAGIC 3. Creates the Unity Catalog objects if they don't exist

# COMMAND ----------

# MAGIC %pip install -q "mne>=1.6,<2" openneuro-py

# COMMAND ----------

import os
import sys

# ---- Unity Catalog layout --------------------------------------------------
CATALOG = "braindance"
SCHEMA_BRONZE = "bronze"
SCHEMA_SILVER = "silver"
SCHEMA_GOLD = "gold"
VOLUME = f"/Volumes/{CATALOG}/{SCHEMA_BRONZE}/raw"         # raw EDF / BIDS files
EXPORT_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA_GOLD}/tapes"  # JSON for the web app

def t(schema: str, name: str) -> str:
    return f"{CATALOG}.{schema}.{name}"

# ---- Sleep-EDF ---------------------------------------------------------------
SLEEP_EDF_SUBJECTS = list(range(0, 20))      # sleep-cassette subjects 00–19 (≈40 nights); widen later
SLEEP_EDF_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
SLEEP_EPOCH_S = 30.0
SLEEP_LOWPASS = 40.0
SLEEP_WAKE_MARGIN_S = 30 * 60                # keep 30 min of wake either side of sleep
SLEEP_CONTEXT = 2                            # neighbour epochs used as context features (±2)

# ---- MET (OpenNeuro) ---------------------------------------------------------
# Fill in from the dataset's OpenNeuro page (the IEEE DataPort entry links to it).
MET_OPENNEURO_ID = "dsXXXXXX"
MET_SUBJECTS = ["01", "02", "03", "04", "05"]  # start small: 128-ch @ 1000 Hz is large
MET_FS_OUT = 250.0
MET_WIN_S, MET_STEP_S = 4.0, 2.0
MET_LOWPASS = 45.0
MET_MAINS_HZ = 50.0
RATING_THRESHOLD = 5.0
# MET_LAYOUT / MET_RATINGS are defined in the next cell, once the package is importable.
# Set them after running the inspection cell in 02_bronze_met.

# COMMAND ----------

# ---- Make ../src importable everywhere ---------------------------------------
# Driver: put the repo's src/ on sys.path.
# Workers: register the package with cloudpickle *by value*, so functions used
# inside pandas UDFs ship their code with them (works on shared and serverless
# compute where addPyFile is not allowed). For scheduled runs the Asset Bundle
# installs a built wheel instead — see databricks.yml.
_src = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

import braindance  # noqa: E402
from pyspark import cloudpickle  # noqa: E402

try:
    cloudpickle.register_pickle_by_value(braindance)
except Exception as e:  # already registered, or the wheel is installed
    print(f"cloudpickle by-value registration skipped: {e}")

from braindance.sources.met import MetLayout, MetRatings  # noqa: E402,F811

MET_LAYOUT = MetLayout(mode="per_file", pre_s=5.0, music_s=120.0, session_from="run")
MET_RATINGS = MetRatings(
    path_glob="",                 # "" = read ratings from each *_events.tsv
    subject_col="participant_id",
    session_col="run",
    valence_col="valence",
    arousal_col="arousal",
    class_col=None,
)

# COMMAND ----------

try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
except Exception as e:
    print(f"Could not create catalog (it may need an admin, or already exist): {e}")
for schema in (SCHEMA_BRONZE, SCHEMA_SILVER, SCHEMA_GOLD):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA_BRONZE}.raw")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA_GOLD}.tapes")
print(f"braindance {braindance.__version__} ready · raw={VOLUME} · tapes={EXPORT_VOLUME}")
