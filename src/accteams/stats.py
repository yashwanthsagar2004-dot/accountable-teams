"""Reporting statistics for the confirmatory runs: robust aggregates, intervals and multiple-comparison control.

Follows Agarwal et al. (NeurIPS 2021, "Deep RL at the Edge of the Statistical Precipice"): interquartile mean,
stratified bootstrap confidence intervals and probability of improvement; plus a permutation test and Holm's
correction for the pre-registered comparisons. Scores are arrays of shape (runs, tasks); a run is one seed.
"""
from __future__ import annotations
import numpy as np


def iqm(scores) -> float:
    """Interquartile mean: the mean of the middle 50% of all scores (runs and tasks pooled)."""
    x = np.sort(np.asarray(scores, dtype=float).ravel())
    n = x.size
    lo, hi = n / 4.0, 3.0 * n / 4.0                      # fractional trimming, exact for any n
    w = np.clip(np.minimum(np.arange(1, n + 1), hi) - np.maximum(np.arange(n), lo), 0.0, 1.0)
    return float((w * x).sum() / w.sum())


def stratified_bootstrap_ci(scores, stat=iqm, reps: int = 2000, level: float = 0.95, seed: int = 0):
    """Percentile interval for `stat`, resampling runs with replacement independently within each task."""
    s = np.asarray(scores, dtype=float)
    s = s[:, None] if s.ndim == 1 else s
    rng = np.random.default_rng(seed)
    runs, tasks = s.shape
    draws = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, runs, size=(runs, tasks))
        draws[r] = stat(np.take_along_axis(s, idx, axis=0))
    a = (1.0 - level) / 2.0
    return float(stat(s)), float(np.quantile(draws, a)), float(np.quantile(draws, 1.0 - a))


def prob_improvement(x, y, lower_is_better: bool = False) -> float:
    """P(a run of X beats a run of Y), averaged over tasks; ties count one half."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x = x[:, None] if x.ndim == 1 else x
    y = y[:, None] if y.ndim == 1 else y
    if lower_is_better:
        x, y = -x, -y
    wins = (x[:, None, :] > y[None, :, :]).mean((0, 1)) + 0.5 * (x[:, None, :] == y[None, :, :]).mean((0, 1))
    return float(wins.mean())


def permutation_test(x, y, reps: int = 20000, seed: int = 0) -> float:
    """Two-sided p-value for a difference in means between two independent sets of runs (labels permuted)."""
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    pooled, obs = np.concatenate([x, y]), abs(x.mean() - y.mean())
    hits = 0
    for _ in range(reps):
        perm = rng.permutation(pooled)
        hits += abs(perm[: x.size].mean() - perm[x.size:].mean()) >= obs - 1e-12
    return (hits + 1) / (reps + 1)


def holm(pvalues, alpha: float = 0.05):
    """Holm's step-down procedure. Returns (adjusted p-values, reject flags) in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (p.size - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj, adj <= alpha
