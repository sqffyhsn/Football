"""Build prediction-log rows for upcoming fixtures (same feature path as the backtest)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fh.data.clean import TIDY_COLUMNS, make_match_id
from fh.features.build import build_features
from fh.live.log import kickoff_utc, model_column
from fh.models.baselines import fh_poisson, league_base_rate
from fh.models.pipeline import predict_bundle
from fh.models.registry import Bundle

log = logging.getLogger(__name__)


def prediction_rows(cfg: dict, history: pd.DataFrame, fixtures: pd.DataFrame,
                    bundles: dict[str, Bundle], primary: str) -> pd.DataFrame:
    """history: tidy played matches; fixtures: tidy rows with NaN results."""
    fx = fixtures.copy()
    fx["match_id"] = make_match_id(fx["div"], fx["date"], fx["home"], fx["away"])
    played = set(history.dropna(subset=["HTHG", "HTAG"])["match_id"])
    fx = fx[~fx["match_id"].isin(played)].drop_duplicates("match_id")
    hist = history[~history["match_id"].isin(set(fx["match_id"]))]
    combined = pd.concat([hist[TIDY_COLUMNS], fx[TIDY_COLUMNS]], ignore_index=True)
    feats = build_features(combined, cfg["features"]["windows"], cfg["features"]["min_periods"])
    f = feats[feats["match_id"].isin(set(fx["match_id"]))].reset_index(drop=True)
    if f.empty:
        return pd.DataFrame()

    tz = cfg["data"]["kickoff_timezone"]
    if not cfg["data"].get("kickoff_timezone_verified"):
        log.warning("kick-off timezone %s is NOT verified; run scripts/verify_timezone.py", tz)
    out = f[["match_id", "div", "season", "home", "away"] + [c for c in f.columns if c.startswith("odds_")]].copy()
    out["date"] = f["date"].dt.strftime("%Y-%m-%d")
    out["kickoff_utc"] = kickoff_utc(f["date"], f["time"], tz).map(lambda t: t.isoformat() if pd.notna(t) else None)
    out["model_version"] = bundles[primary].version
    for label, b in bundles.items():
        out[model_column(label, primary)] = predict_bundle(b, f)
    a = league_base_rate(f)
    out["p_baseline_a"] = np.where(np.isnan(a), np.nan, a)
    out["p_baseline_b"] = fh_poisson(f, window=max(cfg["features"]["windows"]),
                                     shrink=cfg["features"]["poisson_shrinkage_matches"])
    out["p_market_derived"] = bundles[primary].market.predict(f)
    out["market_lambda"] = f["mkt_lambda"].to_numpy()
    return out
