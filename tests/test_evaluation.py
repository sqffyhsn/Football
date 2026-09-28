import numpy as np
import pandas as pd

from fh.evaluation.betting import HYPO, REAL, bets_needed, max_drawdown, price, simulate
from fh.evaluation.metrics import bootstrap_delta, brier, ece, log_loss
from fh.models.calibrate import Calibrator, choose_calibrator


def test_metrics_basic():
    y = np.array([1, 0, 1, 1])
    p = np.array([0.9, 0.1, 0.8, 0.7])
    assert np.isclose(brier(y, p), np.mean((p - y) ** 2))
    assert log_loss(y, np.full(4, 0.75)) > log_loss(y, p)
    assert ece(np.array([1, 0] * 50), np.full(100, 0.5), 5) < 1e-9


def test_bootstrap_detects_clear_improvement():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.5, 0.95, 3000)
    y = (rng.uniform(size=3000) < truth).astype(float)
    d = bootstrap_delta(y, truth, np.full(3000, y.mean()), n=300)
    assert d["ci_high"] < 0


def test_calibrators_fix_a_miscalibrated_model():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.4, 0.95, 6000)
    y = (rng.uniform(size=6000) < truth).astype(int)
    skewed = truth ** 2
    method, scores = choose_calibrator(skewed, y, np.arange(6000))
    assert method in ("platt", "isotonic") and scores[method] < scores["none"]
    assert log_loss(y, Calibrator(method).fit(skewed, y).transform(skewed)) < log_loss(y, skewed)


def test_max_drawdown():
    assert max_drawdown(np.array([1, -1, -1, 2, -3])) == 3
    assert max_drawdown(np.array([-2.0])) == 2


def test_real_and_hypothetical_odds_never_mixed():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=4), "target": [1, 0, 1, 1],
                       "fh_o05_odds": [1.5, np.nan, 1.4, np.nan], "fh_u05_odds": [2.8, np.nan, 3.0, np.nan]})
    p_mkt = np.array([0.7, 0.7, 0.7, np.nan])
    pr = price(df, p_mkt, 0.06)
    assert pr["odds_kind"].tolist()[:3] == [REAL, HYPO, REAL] and pd.isna(pr["odds_kind"].iloc[3])
    table, _ = simulate(df, np.full(4, 0.99), p_mkt, 0.06, [0.0])
    assert set(table["odds_kind"]) == {REAL, HYPO}
    both = table[table["side"] == "both"].set_index("odds_kind")
    assert both.loc[REAL, "n_bets"] == 2 and both.loc[HYPO, "n_bets"] == 1


def test_bets_needed_formula():
    n = bets_needed(edge=0.025, odds=1.35)
    assert 3000 < n < 3600
