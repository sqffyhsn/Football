"""Leakage tests: no feature of match i may depend on match i's result or on any later match."""
import numpy as np
import pandas as pd
import pytest

from fh.config import ROOT
from fh.data.clean import sort_matches
from fh.data.synthetic import make_synthetic
from fh.features.build import build_features, feature_columns

WINDOWS, MINP = [5, 10], 3
RESULTS = ["HTHG", "HTAG", "FTHG", "FTAG"]
ODDS = ["odds_h", "odds_d", "odds_a", "odds_o25", "odds_u25",
        "close_h", "close_d", "close_a", "close_o25", "close_u25"]


@pytest.fixture(scope="module")
def matches():
    return sort_matches(make_synthetic(n_leagues=2, n_teams=8, seasons=("1617", "1718"), seed=1))


def _features(m: pd.DataFrame) -> pd.DataFrame:
    f = build_features(m, WINDOWS, MINP).set_index("match_id")
    return f[feature_columns(f, WINDOWS, with_odds=True)]


def _assert_rows_equal(a: pd.Series, b: pd.Series):
    a, b = a.astype(float).to_numpy(), b.astype(float).to_numpy()
    assert np.allclose(a, b, equal_nan=True), np.flatnonzero(~np.isclose(a, b, equal_nan=True))


def _perturb(m: pd.DataFrame, i: int, rng) -> pd.DataFrame:
    """Scramble match i's RESULTS (not its pre-match odds) and results+odds of all later rows."""
    p = m.copy()
    p.loc[i, RESULTS] = [5, 4, 9, 7]
    later = p.index > i
    k = int(later.sum())
    if k:
        ht = rng.integers(0, 4, size=(k, 2))
        p.loc[later, ["HTHG", "HTAG"]] = ht
        p.loc[later, ["FTHG", "FTAG"]] = ht + rng.integers(0, 3, size=(k, 2))
        p.loc[later, ODDS] = rng.uniform(1.05, 15.0, size=(k, len(ODDS)))
    return p


def test_perturbing_own_result_and_future_leaves_features_unchanged(matches):
    rng = np.random.default_rng(0)
    base = _features(matches)
    for i in rng.choice(len(matches), size=12, replace=False):
        mid = matches.loc[i, "match_id"]
        pert = _features(_perturb(matches, int(i), rng))
        _assert_rows_equal(base.loc[mid], pert.loc[mid])


def test_own_odds_are_a_legitimate_feature(matches):
    """Sanity check the test above is meaningful: changing match i's own odds DOES change its features."""
    i = len(matches) // 2
    mid = matches.loc[i, "match_id"]
    p = matches.copy()
    p.loc[i, "odds_o25"] = p.loc[i, "odds_o25"] * 2
    assert not np.isclose(_features(matches).loc[mid, "mkt_lambda"], _features(p).loc[mid, "mkt_lambda"])


def test_perturbing_past_changes_features(matches):
    """Guard against a vacuous test: past results must influence later features."""
    i = len(matches) - 5
    mid = matches.loc[i, "match_id"]
    p = matches.copy()
    p.loc[: i - 1, ["HTHG", "HTAG"]] = 0
    p.loc[: i - 1, ["FTHG", "FTAG"]] = 0
    assert not np.allclose(_features(matches).loc[mid].astype(float),
                           _features(p).loc[mid].astype(float), equal_nan=True)


def test_truncation_equivalence(matches):
    """Features on full history == features on history ending just before match i (i unplayed)."""
    full = _features(matches)
    n = len(matches)
    for i in [n // 5, n // 2, (4 * n) // 5, n - 1]:
        mid = matches.loc[i, "match_id"]
        trunc = matches.loc[:i].copy()
        trunc.loc[i, RESULTS] = np.nan
        _assert_rows_equal(full.loc[mid], _features(trunc).loc[mid])


def test_same_day_matches_do_not_see_each_other(matches):
    counts = matches.groupby(["div", "date"]).size()
    div, date = counts[counts >= 2].index[5]
    same_day = matches[(matches["div"] == div) & (matches["date"] == date)]
    first, second = same_day.index[0], same_day.index[1]
    p = matches.copy()
    p.loc[first, RESULTS] = [6, 6, 6, 6]
    mid = matches.loc[second, "match_id"]
    _assert_rows_equal(_features(matches).loc[mid], _features(p).loc[mid])


def test_first_match_of_each_team_has_no_form(matches):
    f = build_features(matches, WINDOWS, MINP)
    first_home = f.sort_values("date").groupby("home").head(1)
    for _, r in first_home.iterrows():
        seen_before = ((f["date"] < r["date"]) & ((f["home"] == r["home"]) | (f["away"] == r["home"]))).any()
        if not seen_before:
            assert np.isnan(r["home_fh_for_5"]) and r["home_n_5"] == 0 and np.isnan(r["home_days_since_last"])


def test_upcoming_fixture_uses_same_path_as_history(matches):
    """An unplayed fixture appended to history gets the same features as if its result existed."""
    i = len(matches) - 1
    mid = matches.loc[i, "match_id"]
    fixture = matches.copy()
    fixture.loc[i, RESULTS] = np.nan
    _assert_rows_equal(_features(matches).loc[mid], _features(fixture).loc[mid])


REAL = ROOT / "data" / "processed" / "matches.parquet"


@pytest.mark.skipif(not REAL.exists(), reason="real dataset not built")
def test_perturbation_on_real_data_sample():
    m = pd.read_parquet(REAL)
    m = sort_matches(m[m["div"] == m["div"].iloc[0]].head(1500).reset_index(drop=True))
    rng = np.random.default_rng(1)
    base = _features(m)
    for i in rng.choice(np.arange(200, len(m)), size=5, replace=False):
        mid = m.loc[i, "match_id"]
        _assert_rows_equal(base.loc[mid], _features(_perturb(m, int(i), rng)).loc[mid])
