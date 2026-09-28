"""Metric tables, bootstrap comparisons and plain-language verdicts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fh.evaluation.metrics import bootstrap_delta, log_loss, summarise


def metrics_table(y, preds: dict[str, np.ndarray], n_bins: int = 10) -> pd.DataFrame:
    return pd.DataFrame({name: summarise(y, p, n_bins) for name, p in preds.items()}).T


def comparisons(y, preds: dict[str, np.ndarray], models: list[str], refs: list[str],
                groups=None, n_boot: int = 1000) -> pd.DataFrame:
    out = []
    for m in models:
        for r in refs:
            if m == r:
                continue
            d = bootstrap_delta(y, preds[m], preds[r], log_loss, n=n_boot, groups=groups)
            out.append({"model": m, "reference": r, **d, "verdict": verdict(d)})
    return pd.DataFrame(out)


def verdict(d: dict) -> str:
    if d["ci_high"] < 0:
        return "BEATS reference (95% CI entirely below 0)"
    if d["ci_low"] > 0:
        return "WORSE than reference (95% CI entirely above 0)"
    return "no significant difference (95% CI spans 0)"


def per_league(df: pd.DataFrame, preds: dict[str, np.ndarray], models: list[str]) -> pd.DataFrame:
    out = []
    for div, idx in df.groupby("div").indices.items():
        y = df["target"].to_numpy()[idx]
        row = {"div": div, "n": len(idx), "base_rate": float(y.mean())}
        for m in models:
            p = preds[m][idx]
            ok = ~np.isnan(p)
            row[m] = log_loss(y[ok], p[ok]) if ok.any() else np.nan
        out.append(row)
    return pd.DataFrame(out).set_index("div")


INT_COLS = {"n", "matches", "n_bets", "n_settled"}


def fmt(df: pd.DataFrame, digits: int = 4) -> str:
    """Markdown table without optional dependencies."""
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = []
        for col, v in zip(cols, row):
            if col in INT_COLS and isinstance(v, (int, float, np.number)) and not np.isnan(v):
                cells.append(f"{int(v):,}")
            elif isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else f"{v:.{digits}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)
