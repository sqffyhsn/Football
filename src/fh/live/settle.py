"""Fill actual results and closing odds into the prediction log, and compute CLV.

DERIVED CLV (indirect, no real FH prices): did the market-derived FH probability
move toward the model between prediction time and the close?
    p_market_close = 1 - exp(-s * lambda_close)    (lambda from closing O/U 2.5)
    clv_derived    = sign(p_model - p_market_open) * (p_market_close - p_market_open)
REAL CLV is computed separately, only where real FH opening and closing odds exist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fh.features.odds import closing_lambda, p_fh_from_lambda

CLOSE = ["close_h", "close_d", "close_a", "close_o25", "close_u25"]
UNMATCHED_AFTER = pd.Timedelta(days=3)


def settle(log_df: pd.DataFrame, results: pd.DataFrame, s: float, now_utc: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    out = log_df.copy()
    res = results.dropna(subset=["HTHG", "HTAG"]).drop_duplicates("match_id").set_index("match_id")
    open_ = out["target"].isna()
    hit = open_ & out["match_id"].isin(res.index)
    ids = out.loc[hit, "match_id"]
    for c in ["HTHG", "HTAG"] + CLOSE:
        out.loc[hit, c] = res.loc[ids, c].to_numpy(dtype=float)
    out.loc[hit, "target"] = ((out.loc[hit, "HTHG"] + out.loc[hit, "HTAG"]) >= 1).astype(float)
    out.loc[hit, "settled_ts_utc"] = now_utc.isoformat()
    out.loc[hit, "status"] = "settled"

    lam_close = closing_lambda(out.loc[hit, CLOSE])
    p_close = p_fh_from_lambda(lam_close, s)
    out.loc[hit, "p_market_close"] = p_close
    p_open = out.loc[hit, "p_market_derived"].to_numpy(float)
    lean = np.sign(out.loc[hit, "p_model"].to_numpy(float) - p_open)
    out.loc[hit, "clv_derived"] = lean * (p_close - p_open)

    ko = pd.to_datetime(out["kickoff_utc"], utc=True)
    still_open = out["target"].isna()
    out.loc[still_open & (ko < now_utc - UNMATCHED_AFTER), "status"] = "unmatched"
    out.loc[still_open & (ko >= now_utc - UNMATCHED_AFTER), "status"] = "pending"
    return out, {"newly_settled": int(hit.sum()), "pending": int((out["status"] == "pending").sum()),
                 "unmatched": int((out["status"] == "unmatched").sum())}


def real_clv(log_df: pd.DataFrame) -> pd.DataFrame:
    """REAL CLV where real FH opening AND closing odds exist: open/close - 1 on the side the model leans."""
    d = log_df.dropna(subset=["fh_o05_odds", "fh_u05_odds", "fh_o05_close_odds", "fh_u05_close_odds", "p_model"])
    if d.empty:
        return pd.DataFrame(columns=["match_id", "side", "clv_real"])
    fair_over = (1 / d["fh_o05_odds"]) / (1 / d["fh_o05_odds"] + 1 / d["fh_u05_odds"])
    over = d["p_model"] > fair_over
    clv = np.where(over, d["fh_o05_odds"] / d["fh_o05_close_odds"], d["fh_u05_odds"] / d["fh_u05_close_odds"]) - 1
    return pd.DataFrame({"match_id": d["match_id"], "side": np.where(over, "over", "under"), "clv_real": clv})
