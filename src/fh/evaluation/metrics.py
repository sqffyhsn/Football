"""Probability-quality metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-15


def log_loss(y, p) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def reliability_bins(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins: mean predicted vs observed frequency."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="mergesort")
    chunks = np.array_split(order, n_bins)
    return pd.DataFrame([{"p_mean": p[c].mean(), "y_mean": y[c].mean(), "n": len(c)}
                         for c in chunks if len(c)])


def ece(y, p, n_bins: int = 10) -> float:
    b = reliability_bins(y, p, n_bins)
    return float((b["n"] * (b["p_mean"] - b["y_mean"]).abs()).sum() / b["n"].sum())


def summarise(y, p, n_bins: int = 10) -> dict:
    return {"n": int(len(y)), "log_loss": log_loss(y, p), "brier": brier(y, p),
            "auc": auc(y, p), "ece": ece(y, p, n_bins), "mean_p": float(np.mean(p)),
            "base_rate": float(np.mean(y))}


def bootstrap_delta(y, p_a, p_b, metric=log_loss, n: int = 1000, seed: int = 0,
                    groups=None) -> dict:
    """Paired bootstrap of metric(a) - metric(b). Negative = a is better (for losses).

    If `groups` (e.g. match dates) is given, resample whole groups (block bootstrap)
    to respect within-day correlation.
    """
    y, p_a, p_b = map(lambda v: np.asarray(v, float), (y, p_a, p_b))
    rng = np.random.default_rng(seed)
    point = metric(y, p_a) - metric(y, p_b)
    if groups is None:
        idx_sets = (rng.integers(0, len(y), len(y)) for _ in range(n))
    else:
        codes, uniq = pd.factorize(pd.Series(groups))
        members = pd.Series(np.arange(len(y))).groupby(codes).apply(np.asarray).to_list()
        def gen():
            for _ in range(n):
                pick = rng.integers(0, len(uniq), len(uniq))
                yield np.concatenate([members[k] for k in pick])
        idx_sets = gen()
    deltas = np.array([metric(y[i], p_a[i]) - metric(y[i], p_b[i]) for i in idx_sets])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "p_a_better": float(np.mean(deltas < 0))}
