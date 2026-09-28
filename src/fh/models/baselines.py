"""Baselines. All inputs are pre-match features; nothing is fitted on evaluation data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fh.features.odds import fit_fh_share, p_fh_from_lambda


def league_base_rate(df: pd.DataFrame) -> np.ndarray:
    """Baseline A: the league's expanding historical FH>0.5 rate (strictly earlier dates)."""
    return df["league_rate"].to_numpy(dtype=float)


def _ratio(mean: pd.Series, n: pd.Series, mu: pd.Series, k: float) -> pd.Series:
    """Team rate relative to league average, shrunk toward 1 with k pseudo-matches."""
    total = (mean.fillna(0) * n) + k * mu
    return total / ((n + k) * mu)


def fh_poisson(df: pd.DataFrame, window: int = 10, shrink: float = 10.0) -> np.ndarray:
    """Baseline B: independent Poisson FH goals from team attack/defence ratios.

    lambda_home = league_home_fh * att(home) * def(away)
    lambda_away = league_away_fh * att(away) * def(home)
    p = 1 - exp(-(lambda_home + lambda_away))
    """
    mu = df["league_fh_goals"] / 2.0  # FH goals per team per match
    att_h = _ratio(df[f"home_fh_for_{window}"], df[f"home_n_{window}"], mu, shrink)
    def_h = _ratio(df[f"home_fh_against_{window}"], df[f"home_n_{window}"], mu, shrink)
    att_a = _ratio(df[f"away_fh_for_{window}"], df[f"away_n_{window}"], mu, shrink)
    def_a = _ratio(df[f"away_fh_against_{window}"], df[f"away_n_{window}"], mu, shrink)
    lam = df["league_home_fh_goals"] * att_h * def_a + df["league_away_fh_goals"] * att_a * def_h
    return (1.0 - np.exp(-lam)).to_numpy(dtype=float)


class MarketDerived:
    """Bookmaker-DERIVED benchmark: p = 1 - exp(-s * lambda_OU2.5), s fitted on training rows."""

    def fit(self, train: pd.DataFrame) -> "MarketDerived":
        self.s_ = fit_fh_share(train["mkt_lambda"].to_numpy(float), train["target"].to_numpy(float))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return p_fh_from_lambda(df["mkt_lambda"], self.s_)
