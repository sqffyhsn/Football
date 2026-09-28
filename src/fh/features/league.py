"""League- and season-level base rates from matches on strictly earlier dates.

Same-day matches are excluded (not just the match itself): an earlier kick-off
on the same day would otherwise leak into a later one's features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AGG = {"n": ("target", "size"), "t": ("target", "sum"), "fh": ("fh_goals", "sum"),
       "fhh": ("HTHG", "sum"), "fha": ("HTAG", "sum")}


def _cum_by_day(played: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    daily = played.groupby(keys + ["date"], sort=True).agg(**AGG).reset_index()
    daily = daily.sort_values(keys + ["date"])
    cols = list(AGG)
    daily[cols] = daily.groupby(keys)[cols].cumsum() if keys else daily[cols].cumsum()
    return daily.sort_values("date", kind="mergesort")


def _asof(targets: pd.DataFrame, state: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    state = state.rename(columns={c: f"{prefix}{c}" for c in AGG})
    return pd.merge_asof(targets.sort_values("date", kind="mergesort"), state, on="date",
                         by=keys or None, allow_exact_matches=False)


def league_features(matches: pd.DataFrame, season_shrinkage: float = 20.0) -> pd.DataFrame:
    """One row per match_id: expanding league/season/global rates known before the match date."""
    m = matches.assign(fh_goals=matches["HTHG"] + matches["HTAG"])
    played = m.dropna(subset=["target"])
    targets = m[["match_id", "date", "div", "season"]]

    f = _asof(targets, _cum_by_day(played, []), [], "g_")
    f = _asof(f, _cum_by_day(played, ["div"]), ["div"], "l_")
    f = _asof(f, _cum_by_day(played, ["div", "season"]), ["div", "season"], "s_")
    for p in ("g_", "l_", "s_"):
        f[[f"{p}{c}" for c in AGG]] = f[[f"{p}{c}" for c in AGG]].fillna(0)

    with np.errstate(invalid="ignore", divide="ignore"):
        g_rate = f["g_t"] / f["g_n"]
        l_rate = (f["l_t"] / f["l_n"]).where(f["l_n"] > 0, g_rate)
        out = pd.DataFrame({
            "match_id": f["match_id"],
            "league_rate": l_rate,
            "league_n": f["l_n"],
            # season rate shrunk toward the league's long-run rate early in a season
            "season_rate": (f["s_t"] + season_shrinkage * l_rate) / (f["s_n"] + season_shrinkage),
            "season_n": f["s_n"],
            "league_fh_goals": (f["l_fh"] / f["l_n"]).where(f["l_n"] > 0, f["g_fh"] / f["g_n"]),
            "league_home_fh_goals": (f["l_fhh"] / f["l_n"]).where(f["l_n"] > 0, f["g_fhh"] / f["g_n"]),
            "league_away_fh_goals": (f["l_fha"] / f["l_n"]).where(f["l_n"] > 0, f["g_fha"] / f["g_n"]),
        })
    return out.set_index("match_id")
