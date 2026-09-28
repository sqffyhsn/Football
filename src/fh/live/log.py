"""The forward-testing prediction log (predictions/log.csv).

Invariants: the first prediction for a fixture wins (model columns are never
overwritten); fixtures at/after kick-off are never added; writes are atomic.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ID = ["prediction_ts_utc", "match_id", "div", "season", "date", "kickoff_utc", "home", "away", "model_version"]
PRED = ["p_model", "p_logreg_form", "p_lgbm_form", "p_lgbm_odds",
        "p_baseline_a", "p_baseline_b", "p_market_derived", "market_lambda"]
ODDS = ["odds_h", "odds_d", "odds_a", "odds_o25", "odds_u25"]
FH_ODDS = ["fh_o05_odds", "fh_u05_odds", "fh_odds_source", "fh_odds_ts_utc",
           "fh_o05_close_odds", "fh_u05_close_odds"]
SETTLE = ["HTHG", "HTAG", "target", "settled_ts_utc",
          "close_h", "close_d", "close_a", "close_o25", "close_u25",
          "p_market_close", "clv_derived", "status"]
LOG_COLUMNS = ID + PRED + ODDS + FH_ODDS + SETTLE

# Log column for each non-primary model; the primary model is logged as p_model.
MODEL_COLUMNS = {"LogReg (form only)": "p_logreg_form", "LightGBM (form only)": "p_lgbm_form",
                 "LightGBM (form+odds)": "p_lgbm_odds"}


def model_column(label: str, primary: str) -> str:
    return "p_model" if label == primary else MODEL_COLUMNS[label]


_STR = {"prediction_ts_utc", "match_id", "div", "season", "date", "kickoff_utc", "home", "away",
        "model_version", "fh_odds_source", "fh_odds_ts_utc", "settled_ts_utc", "status"}


def empty_log() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object if c in _STR else float) for c in LOG_COLUMNS})


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Text columns as object (so they can hold strings even when all-NaN), the rest float."""
    df = df.copy()
    for c in LOG_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype(object) if c in _STR else pd.to_numeric(df[c], errors="coerce")
    return df


def load_log(p: Path) -> pd.DataFrame:
    if not Path(p).exists():
        return empty_log()
    df = pd.read_csv(p, dtype={c: str for c in _STR})
    for c in LOG_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    extra = [c for c in df.columns if c not in LOG_COLUMNS]
    return _coerce(df[LOG_COLUMNS + extra])


def save_log(df: pd.DataFrame, p: Path) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, p)


def kickoff_utc(date: pd.Series, time: pd.Series, tz: str) -> pd.Series:
    """Local date + HH:MM in `tz` -> UTC. Missing time -> 00:00 local (conservative:
    the prediction must then be made before the day starts)."""
    t = time.fillna("00:00").astype(str)
    local = pd.to_datetime(pd.to_datetime(date).dt.strftime("%Y-%m-%d") + " " + t, format="%Y-%m-%d %H:%M")
    return local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")


def append_predictions(log_df: pd.DataFrame, new: pd.DataFrame, now_utc: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Append rows for fixtures that kick off strictly after `now_utc` and are not yet logged."""
    new = new.copy()
    ko = pd.to_datetime(new["kickoff_utc"], utc=True)
    started = ko.isna() | (ko <= now_utc)
    logged = new["match_id"].isin(set(log_df["match_id"]))
    add = new[~started & ~logged].copy()
    add["prediction_ts_utc"] = now_utc.isoformat()
    for c in LOG_COLUMNS:
        if c not in add.columns:
            add[c] = np.nan
    assert (pd.to_datetime(add["kickoff_utc"], utc=True) > pd.to_datetime(add["prediction_ts_utc"], utc=True)).all()
    frames = [f for f in (log_df, add[log_df.columns.intersection(add.columns).tolist()]) if len(f)]
    out = pd.concat(frames, ignore_index=True) if frames else empty_log()
    out = _coerce(out)
    assert not out["match_id"].duplicated().any()
    return out, {"added": int(len(add)), "skipped_started": int(started.sum()),
                 "skipped_already_logged": int((logged & ~started).sum())}
