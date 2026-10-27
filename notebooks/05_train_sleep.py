# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Sleep staging model
# MAGIC Five classes (W, N1, N2, N3, REM) from the gold features, evaluated the honest way:
# MAGIC
# MAGIC 1. **GroupKFold by subject** — every night of a test person is unseen in training
# MAGIC 2. **Raw** per-epoch predictions, then **HMM/Viterbi smoothing** with a transition matrix learned from the training folds only
# MAGIC 3. Metrics: accuracy, balanced accuracy, **Cohen's κ** (the sleep-scoring standard), per-class F1, confusion matrix
# MAGIC
# MAGIC Everything is tracked in MLflow; the final model is refit on all subjects and logged.
# MAGIC
# MAGIC > For scale: human expert scorers agree with each other at roughly κ ≈ 0.75–0.8, and N1 is where they disagree most.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from braindance.labels import SLEEP_STAGES
from braindance.modeling import feature_columns, make_model, sleep_cv
from braindance.smoothing import transition_matrix

df = (spark.table(t(SCHEMA_GOLD, "sleep_features"))
      .where("NOT is_artifact AND stage IS NOT NULL").toPandas())
features = feature_columns(df)
print(f"{len(df):,} epochs · {df.subject.nunique()} subjects · {df.session.nunique()} nights · {len(features)} features")
display(df.stage.value_counts(normalize=True).round(3).rename("share").to_frame())

# COMMAND ----------

mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').first()[0]}/braindance-sleep")

with mlflow.start_run(run_name="hgb_hmm_groupkfold") as run:
    cv = sleep_cv(df, features, n_splits=5)
    mlflow.log_params({"model": "HistGradientBoosting", "smoothing": "viterbi", "cv": "GroupKFold(subject)",
                       "n_features": len(features), "context_epochs": SLEEP_CONTEXT})
    for stage_name, m in (("raw", cv.metrics_raw), ("smoothed", cv.metrics_smooth)):
        mlflow.log_metrics({f"{stage_name}_{k}": v for k, v in m.items() if isinstance(v, float)})
        mlflow.log_metrics({f"{stage_name}_f1_{s}": v for s, v in m["per_class_f1"].items()})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (title, m) in zip(axes, (("raw", cv.metrics_raw), ("HMM-smoothed", cv.metrics_smooth))):
        c = np.array(m["confusion"], dtype=float)
        c = c / c.sum(1, keepdims=True)
        ax.imshow(c, cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(5), SLEEP_STAGES); ax.set_yticks(range(5), SLEEP_STAGES)
        ax.set_xlabel("predicted"); ax.set_ylabel("expert")
        ax.set_title(f"{title} · κ={m['kappa']:.3f}")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center",
                        color="black" if c[i, j] > 0.5 else "white", fontsize=8)
    fig.tight_layout()
    mlflow.log_figure(fig, "confusion_matrices.png")
    display(fig)

    # Final model on all subjects (for scoring new nights) + the transition grammar
    final = make_model(seed=7).fit(df[features].to_numpy(np.float32),
                                   df.stage.map({s: i for i, s in enumerate(SLEEP_STAGES)}))
    y_all = df.stage.map({s: i for i, s in enumerate(SLEEP_STAGES)}).to_numpy()
    trans = transition_matrix([g.sort_values("epoch_idx").stage.map({s: i for i, s in enumerate(SLEEP_STAGES)}).to_numpy()
                               for _, g in df.groupby("session")], 5)
    mlflow.sklearn.log_model(final, "model", input_example=df[features].head(3).astype(np.float32))
    mlflow.log_dict({"stages": SLEEP_STAGES, "transition": trans.tolist(),
                     "prior": (np.bincount(y_all, minlength=5) / len(y_all)).tolist(),
                     "features": features}, "hmm.json")
    run_id = run.info.run_id

summary = pd.DataFrame({k: {m: v for m, v in d.items() if isinstance(v, float)}
                        for k, d in (("raw", cv.metrics_raw), ("smoothed", cv.metrics_smooth))}).round(3)
display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Out-of-fold predictions → `gold.sleep_predictions`
# MAGIC These are *cross-validated* predictions (each night scored by a model that never saw that person) — exactly what the app should show.

# COMMAND ----------

pred = df[["subject", "session", "epoch_idx", "t_start", "stage"]].copy()
pred["pred_raw"] = [SLEEP_STAGES[i] for i in cv.pred_raw]
pred["pred"] = [SLEEP_STAGES[i] for i in cv.pred_smooth]
pred["confidence"] = cv.proba.max(1)
pred["proba"] = cv.proba.round(4).tolist()
pred["fold"] = cv.fold
pred["run_id"] = run_id
spark.createDataFrame(pred).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(t(SCHEMA_GOLD, "sleep_predictions"))

metrics = {"model": "HistGradientBoosting + HMM/Viterbi", "classes": SLEEP_STAGES,
           "n_subjects": int(df.subject.nunique()), "n_epochs": int(len(df)),
           "raw": cv.metrics_raw, "smoothed": cv.metrics_smooth,
           "transition": np.round(cv.transition, 3).tolist(), "mlflow_run_id": run_id}
dbutils.fs.mkdirs(EXPORT_VOLUME)
with open(f"{EXPORT_VOLUME}/_metrics_sleep.json", "w") as f:
    json.dump(metrics, f)

# COMMAND ----------

# MAGIC %md
# MAGIC ### The leak, demonstrated
# MAGIC Same model, but folds split by *epoch* instead of subject. The number goes up — and means nothing,
# MAGIC because neighbouring 30 s windows of the same night end up on both sides of the split.

# COMMAND ----------

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold, cross_val_predict

X = df[features].to_numpy(np.float32)
leaky = cross_val_predict(make_model(seed=7), X, y_all, cv=KFold(5, shuffle=True, random_state=0))
print(f"κ, split by subject (honest): {cv.metrics_raw['kappa']:.3f}")
print(f"κ, split by epoch   (leaky) : {cohen_kappa_score(y_all, leaky):.3f}")
