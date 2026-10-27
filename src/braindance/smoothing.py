"""Hidden-Markov smoothing of per-epoch sleep-stage probabilities.

A classifier sees one 30 s window at a time, but sleep has grammar: you don't
jump from N3 to wake and back in a minute. We learn a transition matrix from
the training nights and decode the most likely stage *sequence* with Viterbi.

Emission scores use the "scaled likelihood" trick: a discriminative model
gives P(stage | x); dividing by the class prior P(stage) gives something
proportional to P(x | stage), which is what an HMM wants.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def transition_matrix(sequences: Iterable[Sequence[int]], n_states: int,
                      alpha: float = 1.0) -> np.ndarray:
    """Row-stochastic transition counts with additive (Laplace) smoothing."""
    counts = np.full((n_states, n_states), alpha, dtype=float)
    for seq in sequences:
        seq = np.asarray(seq)
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a, b] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def viterbi(probs: np.ndarray, trans: np.ndarray, prior: np.ndarray,
            emission_weight: float = 1.0) -> np.ndarray:
    """Most likely state path.

    probs: (T, K) classifier probabilities, rows in temporal order
    trans: (K, K) transition matrix
    prior: (K,) class priors from training labels
    emission_weight: <1 trusts the transition grammar more, >1 trusts the classifier more
    """
    eps = 1e-12
    log_e = emission_weight * (np.log(probs + eps) - np.log(prior + eps))
    log_t = np.log(trans + eps)
    T, K = probs.shape
    score = np.log(prior + eps) + log_e[0]
    back = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        cand = score[:, None] + log_t          # (from, to)
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(K)] + log_e[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
