"""Cross-validated training for sleep staging and emotion classification.

Pure pandas / scikit-learn so it runs (and is tested) outside Spark; the
notebooks collect the gold tables to pandas and call these functions.

The one rule everything here follows: **split by subject**. Windows from the
same person are near-duplicates of each other; letting them sit on both sides
of a split measures how well the model memorises people, not brain states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

from .labels import SLEEP_STAGES
from .smoothing import transition_matrix, viterbi

KEY_COLS = {"source", "subject", "session", "epoch_idx", "t_start", "phase", "is_artifact",
            "stage", "valence", "arousal", "quadrant", "ingested_at"}


def feature_columns(df: pd.DataFrame, exclude: Sequence[str] = ()) -> list[str]:
    drop = KEY_COLS | set(exclude)
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def make_model(seed: int = 7, balanced: bool = True, max_iter: int = 300) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        class_weight="balanced" if balanced else None, random_state=seed)


def sleep_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(SLEEP_STAGES)))
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "per_class_f1": {s: float(v) for s, v in zip(SLEEP_STAGES, f1)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


@dataclass
class SleepCV:
    proba: np.ndarray
    pred_raw: np.ndarray
    pred_smooth: np.ndarray
    y: np.ndarray
    metrics_raw: dict
    metrics_smooth: dict
    fold: np.ndarray
    transition: np.ndarray = field(default=None)


def _ordered_sessions(df: pd.DataFrame):
    for _, g in df.sort_values(["session", "epoch_idx"]).groupby("session", sort=False):
        yield g.index.to_numpy()


def sleep_cv(df: pd.DataFrame, features: Sequence[str], n_splits: int = 5,
             seed: int = 7, emission_weight: float = 1.0, max_iter: int = 300) -> SleepCV:
    """Subject-grouped CV, then per-session Viterbi smoothing with a transition
    matrix learned only from that fold's *training* nights."""
    df = df.reset_index(drop=True)
    y = df["stage"].map({s: i for i, s in enumerate(SLEEP_STAGES)}).to_numpy()
    X = df[list(features)].to_numpy(dtype=np.float32)
    K = len(SLEEP_STAGES)
    proba = np.zeros((len(df), K))
    smooth = np.zeros(len(df), dtype=int)
    fold_of = np.zeros(len(df), dtype=int)
    n_splits = min(n_splits, df["subject"].nunique())
    trans_all = []
    for f, (tr, te) in enumerate(GroupKFold(n_splits).split(X, y, df["subject"])):
        m = make_model(seed + f, max_iter=max_iter).fit(X[tr], y[tr])
        p = np.zeros((len(te), K))
        p[:, m.classes_] = m.predict_proba(X[te])
        proba[te] = p
        fold_of[te] = f
        train_df = df.iloc[tr]
        seqs = [y[idx] for idx in _ordered_sessions(train_df)]
        trans = transition_matrix(seqs, K, alpha=1.0)
        trans_all.append(trans)
        prior = np.bincount(y[tr], minlength=K) / len(tr)
        test_df = df.iloc[te]
        for idx in _ordered_sessions(test_df):
            smooth[idx] = viterbi(proba[idx], trans, prior, emission_weight)
    raw = proba.argmax(1)
    return SleepCV(proba, raw, smooth, y, sleep_metrics(y, raw), sleep_metrics(y, smooth),
                   fold_of, np.mean(trans_all, axis=0))


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)), "auc": auc,
            "accuracy": float(accuracy_score(y, pred)), "n": int(len(y)),
            "majority_rate": float(max(y.mean(), 1 - y.mean()))}


def emotion_cv(df: pd.DataFrame, features: Sequence[str], target: str,
               group_col: str = "subject", n_splits: int = 5, seed: int = 7,
               shuffle_labels: bool = False, max_iter: int = 300,
               balanced: bool = True) -> dict:
    """Window-level CV for one binary target, scored at window and song level.

    ``group_col="subject"``  → cross-subject (can it read a stranger?)
    ``group_col="session"``  → within-subject-ish (can it read songs it hasn't heard?)
    ``shuffle_labels=True``  → permutes *session* labels first: one draw of the null
                               distribution, i.e. what "chance" looks like here.
    """
    d = df.dropna(subset=[target]).reset_index(drop=True).copy()
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        sess = d[["subject", "session", target]].drop_duplicates(["subject", "session"])
        sess[target] = rng.permutation(sess[target].to_numpy())
        d = d.drop(columns=[target]).merge(sess, on=["subject", "session"], how="left")
    y = d[target].astype(int).to_numpy()
    X = d[list(features)].to_numpy(dtype=np.float32)
    groups = d[group_col] if group_col != "session" else d["subject"] + "_" + d["session"]
    n_splits = min(n_splits, groups.nunique())
    p = np.zeros(len(d))
    for f, (tr, te) in enumerate(GroupKFold(n_splits).split(X, y, groups)):
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean()
            continue
        m = make_model(seed + f, balanced=balanced, max_iter=max_iter).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, list(m.classes_).index(1)]
    d["p"] = p
    song = d.groupby(["subject", "session"]).agg(p=("p", "mean"), y=(target, "first")).reset_index()
    return {"window": binary_metrics(y, p), "song": binary_metrics(song["y"].to_numpy(), song["p"].to_numpy()),
            "window_proba": p, "frame": d[["subject", "session", "epoch_idx"]].assign(p=p, y=y)}


def permutation_null(df: pd.DataFrame, features: Sequence[str], target: str,
                     n: int = 10, **kw) -> dict:
    """Song-level balanced accuracy under shuffled labels: mean, and the 95th percentile."""
    vals = [emotion_cv(df, features, target, seed=100 + i, shuffle_labels=True, **kw)["song"]["balanced_accuracy"]
            for i in range(n)]
    return {"mean": float(np.mean(vals)), "p95": float(np.percentile(vals, 95)), "draws": [float(v) for v in vals]}
