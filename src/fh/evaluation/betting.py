"""Flat-stake betting simulation — for information only.

Real first-half O/U 0.5 odds are NOT in football-data.co.uk. Rows are priced with
real odds (fh_o05_odds / fh_u05_odds) when present, otherwise with SYNTHETIC odds:
    fair = p_market_derived;  odds = 1 / (fair * (1 + margin))
REAL and HYPOTHETICAL groups are always reported separately, never pooled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

REAL, HYPO = "REAL", "HYPOTHETICAL"


def price(df: pd.DataFrame, p_market: np.ndarray, margin: float) -> pd.DataFrame:
    """Return odds_over, odds_under and odds_kind for every row."""
    out = pd.DataFrame(index=df.index)
    real_o = df["fh_o05_odds"] if "fh_o05_odds" in df else pd.Series(np.nan, index=df.index)
    real_u = df["fh_u05_odds"] if "fh_u05_odds" in df else pd.Series(np.nan, index=df.index)
    has_real = real_o.notna() & real_u.notna()
    pm = np.clip(np.asarray(p_market, float), 1e-6, 1 - 1e-6)
    out["odds_over"] = np.where(has_real, real_o, 1 / (pm * (1 + margin)))
    out["odds_under"] = np.where(has_real, real_u, 1 / ((1 - pm) * (1 + margin)))
    out["odds_kind"] = np.where(has_real, REAL, np.where(np.isnan(np.asarray(p_market, float)), None, HYPO))
    return out


def max_drawdown(profits: np.ndarray) -> float:
    if len(profits) == 0:
        return 0.0
    cum = np.cumsum(profits)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float(np.max(peak - cum))


def bets(df: pd.DataFrame, p_model: np.ndarray, priced: pd.DataFrame, threshold: float,
         stake: float = 1.0) -> pd.DataFrame:
    """Bet OVER when p_model - 1/odds_over > threshold; UNDER likewise. Chronological order."""
    p = np.asarray(p_model, float)
    y = df["target"].to_numpy(float)
    rows = []
    for side, odds_col, p_side, win in (("over", "odds_over", p, y == 1),
                                        ("under", "odds_under", 1 - p, y == 0)):
        odds = priced[odds_col].to_numpy(float)
        edge = p_side - 1 / odds
        mask = (edge > threshold) & priced["odds_kind"].notna().to_numpy() & ~np.isnan(y)
        rows.append(pd.DataFrame({
            "date": df["date"].to_numpy()[mask], "side": side, "kind": priced["odds_kind"].to_numpy()[mask],
            "odds": odds[mask], "edge": edge[mask],
            "profit": np.where(win[mask], stake * (odds[mask] - 1), -stake), "stake": stake,
        }))
    return pd.concat(rows, ignore_index=True).sort_values("date", kind="mergesort").reset_index(drop=True)


def summarise_bets(b: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    if b.empty:
        return {"n_bets": 0, "staked": 0.0, "profit": 0.0, "roi": float("nan"),
                "roi_ci_low": float("nan"), "roi_ci_high": float("nan"), "max_drawdown": 0.0}
    profits, stakes = b["profit"].to_numpy(), b["stake"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), size=(n_boot, len(b)))
    boot = profits[idx].sum(1) / stakes[idx].sum(1)
    return {"n_bets": int(len(b)), "staked": float(stakes.sum()), "profit": float(profits.sum()),
            "roi": float(profits.sum() / stakes.sum()),
            "roi_ci_low": float(np.percentile(boot, 2.5)), "roi_ci_high": float(np.percentile(boot, 97.5)),
            "max_drawdown": max_drawdown(profits), "mean_odds": float(b["odds"].mean())}


def simulate(df: pd.DataFrame, p_model, p_market, margin: float, thresholds, stake: float = 1.0):
    """Returns (summary table, {(kind, threshold): bets frame}). Kinds never mixed."""
    priced = price(df, p_market, margin)
    out, all_bets = [], {}
    for t in thresholds:
        b = bets(df, p_model, priced, t, stake)
        for kind in (REAL, HYPO):
            for side in ("over", "under", "both"):
                sub = b[(b["kind"] == kind) & ((b["side"] == side) | (side == "both"))]
                if kind == REAL and sub.empty:
                    continue
                out.append({"odds_kind": kind, "threshold": t, "side": side, **summarise_bets(sub)})
                if side == "both":
                    all_bets[(kind, t)] = sub
    return pd.DataFrame(out), all_bets


def bets_needed(edge: float = 0.025, odds: float = 1.35, alpha: float = 0.05, power: float = 0.80) -> int:
    """Bets needed to detect a true ROI of `edge` vs 0 with a one-sided z-test.

    Win prob if ROI = edge: p = (1 + edge) / odds. Per-unit profit variance
    sigma^2 = p (1 - p) odds^2.  n = ((z_{1-alpha} + z_{power}) * sigma / edge)^2.
    """
    p = (1 + edge) / odds
    sigma = np.sqrt(p * (1 - p)) * odds
    z = norm.ppf(1 - alpha) + norm.ppf(power)
    return int(np.ceil((z * sigma / edge) ** 2))
