import numpy as np
import pandas as pd
import pytest

from fh.config import load_config
from fh.data.clean import sort_matches
from fh.data.synthetic import make_synthetic
from fh.features.build import build_features
from fh.live.log import append_predictions, empty_log
from fh.live.predict import prediction_rows
from fh.models.pipeline import fit_bundle, predict_bundle
from fh.models.splits import Fold
from fh.models.train import Spec

PRIMARY = "LogReg (form+odds)"


@pytest.fixture(scope="module")
def setup():
    cfg = load_config()
    m = sort_matches(make_synthetic(n_leagues=2, n_teams=8, seasons=("1617", "1718", "1819"), seed=3))
    last = m[m["season"] == "1819"]
    fixtures = last.tail(6).copy()
    history = m.drop(fixtures.index)
    fixtures[["HTHG", "HTAG", "FTHG", "FTAG"]] = np.nan
    feats = build_features(history, cfg["features"]["windows"], cfg["features"]["min_periods"])
    fold = Fold("t", ("1617",), "1718", "1819")
    bundles = {}
    for label, odds in ((PRIMARY, True), ("LogReg (form only)", False)):
        bundles[label], _ = fit_bundle(feats, cfg, Spec("logreg", (10,), odds, {"C": 0.1}), fold, "test")
    return cfg, history, fixtures, bundles


def test_prediction_rows_match_backtest_feature_path(setup):
    cfg, history, fixtures, bundles = setup
    rows = prediction_rows(cfg, history, fixtures, bundles, PRIMARY)
    assert len(rows) == 6
    assert rows["p_model"].between(0, 1).all() and rows["p_logreg_form"].between(0, 1).all()
    assert rows["p_market_derived"].notna().all() and rows["kickoff_utc"].notna().all()
    # identical to the backtest path run on data where these matches HAVE results
    full = pd.concat([history, fixtures.assign(HTHG=3, HTAG=2, FTHG=4, FTAG=3)])
    feats = build_features(full, cfg["features"]["windows"], cfg["features"]["min_periods"]).set_index("match_id")
    # Only the first fixture date is comparable: later fixtures legitimately see earlier ones' results there.
    first = rows[rows["date"] == rows["date"].min()]
    expected = predict_bundle(bundles[PRIMARY], feats.loc[first["match_id"]])
    assert len(first) >= 1 and np.allclose(first["p_model"], expected)


def test_already_played_fixtures_are_not_predicted(setup):
    cfg, history, fixtures, bundles = setup
    played = history.tail(2).copy()
    rows = prediction_rows(cfg, history, pd.concat([fixtures, played]), bundles, PRIMARY)
    assert len(rows) == 6


def test_rows_append_to_log(setup):
    cfg, history, fixtures, bundles = setup
    rows = prediction_rows(cfg, history, fixtures, bundles, PRIMARY)
    now = pd.to_datetime(rows["kickoff_utc"], utc=True).min() - pd.Timedelta(hours=1)
    log, stats = append_predictions(empty_log(), rows, now)
    assert stats["added"] == 6 and log["p_model"].notna().all() and log["fh_o05_odds"].isna().all()
