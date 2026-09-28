"""Rolling pre-match team form.

Leak-freedom by construction: rolling stats are computed on PLAYED matches only
and describe a team's state *after* each match. Every target row (historical or
upcoming) then looks up the most recent state strictly before its own date via
`merge_asof(allow_exact_matches=False)`. A match can therefore never see its own
result or any later one, and the same code path serves history and fixtures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATS = ["fh_for", "fh_against", "ft_for", "ft_against", "fh_any"]


def long_table(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match."""
    common = ["match_id", "div", "date"]
    home = matches[common].assign(
        team=matches["home"], venue="H",
        fh_for=matches["HTHG"], fh_against=matches["HTAG"],
        ft_for=matches["FTHG"], ft_against=matches["FTAG"])
    away = matches[common].assign(
        team=matches["away"], venue="A",
        fh_for=matches["HTAG"], fh_against=matches["HTHG"],
        ft_for=matches["FTAG"], ft_against=matches["FTHG"])
    long = pd.concat([home, away], ignore_index=True)
    long["fh_any"] = np.where(long["fh_for"].isna() | long["fh_against"].isna(), np.nan,
                              ((long["fh_for"] + long["fh_against"]) >= 1).astype(float))
    return long.sort_values(["date", "match_id", "venue"], kind="mergesort").reset_index(drop=True)


def _post_match_state(played: pd.DataFrame, keys: list[str], windows: list[int],
                      min_periods: int, prefix: str) -> pd.DataFrame:
    played = played.sort_values(keys + ["date"], kind="mergesort")
    g = played.groupby(keys, sort=False)
    out = played[keys + ["date"]].copy()
    for n in windows:
        roll = g[STATS].rolling(n, min_periods=min_periods).mean().reset_index(level=keys, drop=True)
        for s in STATS:
            out[f"{prefix}{s}_{n}"] = roll[s]
        out[f"{prefix}n_{n}"] = g["fh_for"].rolling(n, min_periods=1).count().reset_index(level=keys, drop=True)
    out[f"{prefix}last_date"] = played["date"]
    return out.sort_values("date", kind="mergesort")


def _asof(targets: pd.DataFrame, state: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Attach, for each target row, the latest state with date strictly before its date."""
    t = targets.sort_values("date", kind="mergesort")
    merged = pd.merge_asof(t, state, on="date", by=by, allow_exact_matches=False)
    return merged


def team_form_features(matches: pd.DataFrame, windows: list[int], min_periods: int) -> pd.DataFrame:
    """Return one row per match_id with home_* and away_* pre-match form features."""
    long = long_table(matches)
    played = long.dropna(subset=["fh_for", "fh_against", "ft_for", "ft_against"])

    overall = _post_match_state(played, ["team"], windows, min_periods, prefix="")
    venue = _post_match_state(played, ["team", "venue"], windows, min_periods, prefix="v_")

    targets = long[["match_id", "date", "team", "venue"]]
    feat = _asof(targets, overall, by=["team"])
    feat = _asof(feat, venue.drop(columns=["v_last_date"]), by=["team", "venue"])
    feat["days_since_last"] = (feat["date"] - feat["last_date"]).dt.days
    feat = feat.drop(columns=["last_date"])

    value_cols = [c for c in feat.columns if c not in ("match_id", "date", "team", "venue")]
    for c in value_cols:
        if c.startswith("n_") or c.startswith("v_n_"):
            feat[c] = feat[c].fillna(0)
    home = feat[feat["venue"] == "H"].set_index("match_id")[value_cols].add_prefix("home_")
    away = feat[feat["venue"] == "A"].set_index("match_id")[value_cols].add_prefix("away_")
    out = home.join(away, how="outer")
    out["rest_diff"] = out["home_days_since_last"] - out["away_days_since_last"]
    return out
