"""Bookmaker-implied probabilities and the market-DERIVED first-half benchmark.

football-data.co.uk has no first-half O/U 0.5 odds. We derive a benchmark:
  1. de-vig O/U 2.5 (proportional normalisation) -> p(over 2.5)
  2. invert a Poisson total-goals model: find lambda with P(N >= 3 | lambda) = p_over
  3. p_fh = 1 - exp(-s * lambda), where s (first-half share of goal intensity) is
     fitted on TRAINING data only.
This is an indirect, model-based estimate, not a quoted market price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

_LAMBDA_GRID = np.linspace(0.05, 10.0, 4000)
_P_OVER25_GRID = poisson.sf(2, _LAMBDA_GRID)  # P(N >= 3), increasing in lambda


def devig(*odds: pd.Series) -> tuple[list[pd.Series], pd.Series]:
    """Proportional overround removal. Returns (fair probabilities, overround)."""
    inv = [1.0 / o for o in odds]
    book = sum(inv)
    return [i / book for i in inv], book - 1.0


def lambda_from_over25(p_over: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(p_over, dtype=float)
    lam = np.interp(p, _P_OVER25_GRID, _LAMBDA_GRID)
    return np.where(np.isnan(p), np.nan, lam)


def p_fh_from_lambda(lam, s: float) -> np.ndarray:
    return 1.0 - np.exp(-s * np.asarray(lam, dtype=float))


def fit_fh_share(lam: np.ndarray, y: np.ndarray) -> float:
    """MLE of s in p = 1 - exp(-s*lambda) on training rows (a strong, fair benchmark)."""
    mask = ~np.isnan(lam) & ~np.isnan(y)
    lam, y = lam[mask], y[mask]

    def nll(s):
        p = np.clip(p_fh_from_lambda(lam, s), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    return float(minimize_scalar(nll, bounds=(0.05, 1.0), method="bounded").x)


def odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-match odds features. Uses only pre-closing odds columns (odds_*)."""
    (ph, pd_, pa), over_1x2 = devig(df["odds_h"], df["odds_d"], df["odds_a"])
    (po, _), over_ou = devig(df["odds_o25"], df["odds_u25"])
    out = pd.DataFrame(index=df.index)
    out["mkt_p_home"] = ph
    out["mkt_p_draw"] = pd_
    out["mkt_p_away"] = pa
    out["mkt_overround_1x2"] = over_1x2
    out["mkt_p_over25"] = po
    out["mkt_overround_ou"] = over_ou
    out["mkt_lambda"] = lambda_from_over25(po)
    return out


def closing_lambda(df: pd.DataFrame) -> np.ndarray:
    """Lambda implied by CLOSING O/U 2.5 odds. For CLV analysis only, never a feature."""
    (po, _), _ = devig(df["close_o25"], df["close_u25"])
    return lambda_from_over25(po)
