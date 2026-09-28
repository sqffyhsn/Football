import numpy as np
import pandas as pd
import pytest

from fh.data.odds_providers import ManualCsvOddsProvider, merge_fh_odds, validate_manual
from fh.live.log import FH_ODDS, LOG_COLUMNS, append_predictions, empty_log, kickoff_utc, load_log, save_log

NOW = pd.Timestamp("2026-10-03 10:00", tz="UTC")


def fixtures(kickoffs):
    n = len(kickoffs)
    return pd.DataFrame({
        "match_id": [f"E0_m{i}" for i in range(n)], "div": "E0", "season": "2627",
        "date": [pd.Timestamp(k).strftime("%Y-%m-%d") for k in kickoffs],
        "kickoff_utc": [pd.Timestamp(k, tz="UTC").isoformat() for k in kickoffs],
        "home": [f"H{i}" for i in range(n)], "away": [f"A{i}" for i in range(n)],
        "model_version": "v1", "p_model": np.linspace(0.6, 0.8, n)})


def test_schema_has_nullable_fh_odds_columns():
    for c in FH_ODDS:
        assert c in LOG_COLUMNS
    assert empty_log()["fh_o05_odds"].isna().all()


def test_append_rejects_started_and_keeps_first_prediction(tmp_path):
    fx = fixtures(["2026-10-03 14:00", "2026-10-03 10:00", "2026-10-02 19:00"])
    log, stats = append_predictions(empty_log(), fx, NOW)
    assert stats == {"added": 1, "skipped_started": 2, "skipped_already_logged": 0}
    assert (pd.to_datetime(log["prediction_ts_utc"], utc=True) < pd.to_datetime(log["kickoff_utc"], utc=True)).all()
    save_log(log, tmp_path / "log.csv")
    log = load_log(tmp_path / "log.csv")
    again = fx.assign(p_model=0.1)  # a re-run with different numbers must not overwrite
    log2, stats2 = append_predictions(log, again, NOW)
    assert stats2["added"] == 0 and stats2["skipped_already_logged"] == 1
    assert log2.loc[0, "p_model"] == pytest.approx(0.6)
    assert not log2["match_id"].duplicated().any()


def test_kickoff_uk_time_to_utc_across_dst():
    d = pd.Series(pd.to_datetime(["2025-10-25", "2025-10-26"]))
    t = pd.Series(["15:00", "15:00"])
    ko = kickoff_utc(d, t, "Europe/London")
    assert ko.iloc[0] == pd.Timestamp("2025-10-25 14:00", tz="UTC")  # BST
    assert ko.iloc[1] == pd.Timestamp("2025-10-26 15:00", tz="UTC")  # GMT


def manual(rows):
    cols = ["div", "date", "home", "away", "fh_o05_odds", "fh_u05_odds", "source", "ts_utc",
            "fh_o05_close_odds", "fh_u05_close_odds"]
    return pd.DataFrame(rows, columns=cols)


def test_manual_validation_rejects_with_reasons():
    df = manual([
        ["E0", "2026-10-03", "H0", "A0", 1.33, 3.40, "bet365", "2026-10-02T12:00Z", np.nan, np.nan],  # ok (1.046)
        ["E0", "2026-10-03", "H1", "A1", 1.0, 3.4, "bet365", "2026-10-02T12:00Z", np.nan, np.nan],     # odds <= 1
        ["E0", "2026-10-03", "H2", "A2", 1.20, 2.0, "bet365", "2026-10-02T12:00Z", np.nan, np.nan],    # sum 1.33
        ["E0", "2026-10-03", "H3", "A3", 1.33, 3.4, "", "2026-10-02T12:00Z", np.nan, np.nan],          # no source
        ["E0", "2026-10-03", "H4", "A4", 1.33, 3.4, "bet365", np.nan, np.nan, np.nan],                 # no ts
        ["E0", "2026-10-03", "H5", "A5", 2.00, 2.20, "bet365", "2026-10-02T12:00Z", np.nan, np.nan],   # sum 0.96 (arb)
    ])
    valid, warnings = validate_manual(df)
    assert valid["home"].tolist() == ["H0"]
    assert len(warnings) == 5
    assert "> 1.0" in warnings[0] and "outside" in warnings[1] and "source is empty" in warnings[2]
    assert "ts_utc is empty" in warnings[3] and "outside" in warnings[4]
    assert all(w.startswith("REJECTED") for w in warnings)


def test_manual_csv_provider_loads_and_warns(tmp_path):
    p = tmp_path / "m.csv"
    manual([["E0", "2026-10-03", "H0", "A0", 1.33, 3.4, "bet365", "2026-10-02T12:00Z", np.nan, np.nan],
            ["E0", "2026-10-03", "H1", "A1", 0.5, 3.4, "bet365", "2026-10-02T12:00Z", np.nan, np.nan]]).to_csv(p, index=False)
    prov = ManualCsvOddsProvider(p)
    got = prov.get_fh_odds(pd.DataFrame())
    assert len(got) == 1 and len(prov.warnings) == 1


def test_merge_fills_only_empty_and_rejects_after_kickoff():
    log, _ = append_predictions(empty_log(), fixtures(["2026-10-03 14:00", "2026-10-03 16:00"]), NOW)
    log.loc[1, ["fh_o05_odds", "fh_u05_odds", "fh_odds_source"]] = [1.30, 3.60, "earlier"]
    odds = manual([
        ["E0", "2026-10-03", "H0", "A0", 1.33, 3.40, "bet365", "2026-10-03T14:05Z", 1.30, 3.60],  # after KO
        ["E0", "2026-10-03", "H1", "A1", 1.35, 3.30, "bet365", "2026-10-03T09:00Z", 1.31, 3.55],  # already filled
    ])
    out, notes = merge_fh_odds(log, odds)
    assert np.isnan(out.loc[0, "fh_o05_odds"]) and len(notes) == 1 and "not before kick-off" in notes[0]
    assert out.loc[0, "fh_o05_close_odds"] == 1.30          # closing odds accepted
    assert out.loc[1, "fh_o05_odds"] == 1.30 and out.loc[1, "fh_odds_source"] == "earlier"  # never overwritten
    assert out.loc[1, "fh_o05_close_odds"] == 1.31
