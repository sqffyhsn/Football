"""End-to-end settle dry run on a synthetic, backdated prediction log with a fake results provider."""
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from fh.config import ROOT, load_config
from fh.live.log import LOG_COLUMNS, empty_log, load_log, save_log
from fh.live.report import (HOLDING, INCONCLUSIVE, INCONSISTENT, MIN_CI_MATCHES, convergence,
                            convergence_status)

spec = importlib.util.spec_from_file_location("settle_results", ROOT / "scripts" / "settle_results.py")
settle_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settle_results)

NOW = pd.Timestamp("2026-10-20 12:00", tz="UTC")
N = 260
S = 0.45
MANIFEST = {
    "primary_model": "LogReg (form+odds)", "primary_version": "live-test",
    "models": {"LogReg (form+odds)": {"market_fh_share_s": S}},
    "reference": {"backtest_primary_vs_market": {"delta": -0.00063, "ci_low": -0.00116, "ci_high": -0.00012},
                  "test_primary_vs_market": {"delta": 0.00145, "ci_low": -0.00059, "ci_high": 0.00396},
                  "sigma_loss_diff_per_match": 0.0386, "matches_to_resolve_backtest_edge_80pct": 29517},
}


def _odds_for_p_over(p, margin=1.05):
    return 1 / (p * margin), 1 / ((1 - p) * margin)


def synthetic_log():
    rows = []
    for i in range(N + 2):
        ko = pd.Timestamp("2026-08-01 14:00", tz="UTC") + pd.Timedelta(hours=7 * i)
        rows.append({"prediction_ts_utc": (ko - pd.Timedelta(days=1)).isoformat(), "match_id": f"E0_m{i}",
                     "div": "E0", "season": "2627", "date": ko.strftime("%Y-%m-%d"), "kickoff_utc": ko.isoformat(),
                     "home": f"H{i}", "away": f"A{i}", "model_version": "live-test",
                     "p_model": 0.72 if i % 2 == 0 else 0.68, "p_market_derived": 0.70,
                     "p_baseline_a": 0.70, "p_baseline_b": 0.69})
    df = pd.concat([empty_log(), pd.DataFrame(rows)], ignore_index=True)
    # one fixture never gets a result and kicked off > 3 days ago; one kicks off after NOW
    df.loc[N, "kickoff_utc"] = (NOW - pd.Timedelta(days=10)).isoformat()
    df.loc[N + 1, "kickoff_utc"] = (NOW + pd.Timedelta(days=1)).isoformat()
    return df[LOG_COLUMNS]


class FakeProvider:
    """Known results: rows i%4 in {0,1} -> market moved UP at the close, {2,3} -> DOWN."""
    name = "fake"

    def get_results(self, season):
        out = []
        for i in range(N):
            up = i % 4 in (0, 1)
            o, u = _odds_for_p_over(0.65 if up else 0.35)
            out.append({"match_id": f"E0_m{i}", "HTHG": i % 3 == 0, "HTAG": i % 5 == 0,
                        "close_h": 2.0, "close_d": 3.4, "close_a": 3.8, "close_o25": o, "close_u25": u})
        return pd.DataFrame(out).astype({"HTHG": float, "HTAG": float})


@pytest.fixture
def settled_run(tmp_path):
    log_path = tmp_path / "log.csv"
    save_log(synthetic_log(), log_path)
    info = settle_results.run(load_config(), FakeProvider(), log_path, MANIFEST, tmp_path / "live", NOW, n_boot=200)
    return info, load_log(log_path), tmp_path / "live"


def test_settle_fills_results_and_statuses(settled_run):
    info, log, _ = settled_run
    assert info["newly_settled"] == N and info["pending"] == 1 and info["unmatched"] == 1
    s = log[log["status"] == "settled"]
    assert (s["target"] == ((s["HTHG"] + s["HTAG"]) >= 1)).all()
    assert s["settled_ts_utc"].notna().all()


def test_clv_signs(settled_run):
    _, log, _ = settled_run
    s = log[log["status"] == "settled"].set_index("match_id")
    for i in range(N):
        model_up = i % 2 == 0          # p_model 0.72 > market 0.70
        market_up = i % 4 in (0, 1)    # closing lambda moved p_market_close above/below 0.70
        expected = 1 if model_up == market_up else -1
        assert np.sign(s.loc[f"E0_m{i}", "clv_derived"]) == expected, i
        assert (s.loc[f"E0_m{i}", "p_market_close"] > 0.70) == market_up


def test_convergence_csv_rows_and_history(settled_run, tmp_path):
    _, _, out = settled_run
    conv = pd.read_csv(out / "convergence.csv")
    assert conv["n"].tolist() == [50, 100, 150, 200, 250, 260]
    assert conv.loc[conv["n"] < MIN_CI_MATCHES, ["ci_low", "ci_high"]].isna().all().all()
    assert conv.loc[conv["n"] >= MIN_CI_MATCHES, ["ci_low", "ci_high"]].notna().all().all()
    assert conv.loc[0, "status"].startswith("Insufficient data — 50 matches settled")
    assert (out / "live_report.md").exists() and (out / "convergence.png").exists()
    assert len(pd.read_csv(out / "convergence_history.csv")) == 1


def test_resettle_is_idempotent(settled_run, tmp_path):
    _, log, out = settled_run
    log_path = tmp_path / "log.csv"
    info = settle_results.run(load_config(), FakeProvider(), log_path, MANIFEST, out, NOW, n_boot=200)
    assert info["newly_settled"] == 0
    assert len(pd.read_csv(out / "convergence_history.csv")) == 2


def test_status_rules():
    bt = -0.00063
    assert convergence_status(150, np.nan, np.nan, bt).startswith("Insufficient data — 150")
    assert convergence_status(500, -0.003, -0.0001, bt) == HOLDING
    assert convergence_status(500, -0.0002, 0.004, bt) == INCONSISTENT
    assert convergence_status(500, -0.002, 0.001, bt) == INCONCLUSIVE


def test_report_never_pools_real_and_hypothetical(settled_run):
    _, _, out = settled_run
    text = (out / "live_report.md").read_text()
    assert "### REAL" in text and "### HYPOTHETICAL" in text
