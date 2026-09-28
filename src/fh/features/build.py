"""Assemble the feature matrix for historical matches and/or upcoming fixtures.

Upcoming fixtures are appended to history with NaN results and pass through the
exact same code path, so there is no training/serving skew.
"""
from __future__ import annotations

import pandas as pd

from fh.data.clean import TIDY_COLUMNS, sort_matches
from fh.features.league import league_features
from fh.features.odds import odds_features
from fh.features.target import add_target
from fh.features.team_form import team_form_features

ID_COLS = ["match_id", "div", "season", "date", "time", "home", "away"]


def build_features(matches: pd.DataFrame, windows: list[int], min_periods: int) -> pd.DataFrame:
    m = add_target(sort_matches(matches[TIDY_COLUMNS]))
    form = team_form_features(m, windows, min_periods)
    league = league_features(m)
    odds = odds_features(m).set_axis(m["match_id"])
    out = (m.set_index("match_id")
             .join(form).join(league).join(odds)
             .reset_index())
    return out.loc[:, ~out.columns.duplicated()]


def feature_columns(df: pd.DataFrame, windows: list[int], with_odds: bool) -> list[str]:
    cols = []
    for side in ("home", "away"):
        for n in windows:
            cols += [f"{side}_{s}_{n}" for s in ("fh_for", "fh_against", "ft_for", "ft_against", "fh_any")]
            cols += [f"{side}_v_{s}_{n}" for s in ("fh_for", "fh_against", "ft_for", "ft_against", "fh_any")]
            cols += [f"{side}_n_{n}", f"{side}_v_n_{n}"]
        cols.append(f"{side}_days_since_last")
    cols += ["rest_diff", "league_rate", "season_rate", "season_n",
             "league_fh_goals", "league_home_fh_goals", "league_away_fh_goals"]
    if with_odds:
        cols += ["mkt_p_home", "mkt_p_draw", "mkt_p_away", "mkt_overround_1x2",
                 "mkt_p_over25", "mkt_overround_ou", "mkt_lambda"]
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"missing feature columns: {missing}"
    return cols
