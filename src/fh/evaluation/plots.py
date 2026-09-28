"""Static report plots (matplotlib PNG). Colours: fixed categorical order, never cycled."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fh.evaluation.metrics import reliability_bins  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def reliability_plot(y, preds: dict[str, np.ndarray], path: Path, title: str, n_bins: int = 10):
    """preds: label -> probabilities. Colour is bound to the label order given."""
    fig, ax = plt.subplots(figsize=(6.4, 6), facecolor=SURFACE)
    _style(ax)
    lo = min(min(np.min(p) for p in preds.values()), float(np.mean(y))) - 0.05
    lo = max(0.0, lo)
    ax.plot([lo, 1], [lo, 1], color=INK2, linewidth=1, linestyle="--", label="perfect calibration")
    for i, (label, p) in enumerate(preds.items()):
        b = reliability_bins(y, p, n_bins)
        ax.plot(b["p_mean"], b["y_mean"], color=SERIES[i % len(SERIES)], linewidth=2,
                marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax.set_xlim(lo, 1)
    ax.set_ylim(lo, 1)
    ax.set_xlabel("Mean predicted P(FH goal)", color=INK)
    ax.set_ylabel("Observed frequency", color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def league_delta_plot(table: pd.DataFrame, path: Path, title: str):
    """table: index=div, column 'delta' (model minus reference log loss). Negative = model better."""
    t = table.sort_values("delta")
    fig, ax = plt.subplots(figsize=(6.4, 0.35 * len(t) + 1.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    ax.barh(t.index, t["delta"], color=SERIES[0], height=0.6)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlabel("Δ log loss vs reference (negative = model better)", color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pnl_plot(curves: dict[str, pd.Series], path: Path, title: str):
    """curves: label -> cumulative profit indexed by bet order."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _style(ax)
    for i, (label, c) in enumerate(curves.items()):
        ax.plot(np.arange(len(c)), c.to_numpy(), color=SERIES[i % len(SERIES)], linewidth=2, label=label)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xlabel("Bet number (chronological)", color=INK)
    ax.set_ylabel("Cumulative profit (units)", color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    if len(curves) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
