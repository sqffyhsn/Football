"""Walk-forward CV, model selection and bundle fitting."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fh.evaluation.metrics import log_loss
from fh.features.build import feature_columns
from fh.models import train as T
from fh.models.baselines import MarketDerived, fh_poisson, league_base_rate
from fh.models.calibrate import Calibrator, choose_calibrator
from fh.models.registry import Bundle, new_version
from fh.models.splits import Fold, rows

log = logging.getLogger(__name__)

BASE_A, BASE_B, MARKET = "Baseline A: league rate", "Baseline B: FH Poisson", "Market-derived (O/U 2.5)"


def baseline_preds(fit_df: pd.DataFrame, eval_df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    a = league_base_rate(eval_df)
    a = np.where(np.isnan(a), fit_df["target"].mean(), a)
    b = fh_poisson(eval_df, window=max(cfg["features"]["windows"]),
                   shrink=cfg["features"]["poisson_shrinkage_matches"])
    market = MarketDerived().fit(fit_df.dropna(subset=["mkt_lambda"]))
    return {BASE_A: a, BASE_B: b, MARKET: market.predict(eval_df)}


def run_cv(feats: pd.DataFrame, cfg: dict, folds: list[Fold], specs: list[T.Spec]):
    """Returns (oof long frame [match_id, fold, model, p], fold metrics frame)."""
    oof, metrics = [], []
    rs, lgb_cfg = cfg["models"]["random_state"], cfg["models"]["lightgbm"]
    for fold in folds:
        fit_df, stop_df, ev = rows(feats, fold.fit_seasons), rows(feats, fold.stop_season), rows(feats, fold.eval_season)
        log.info("%s: fit=%d stop=%d eval=%d", fold.name, len(fit_df), len(stop_df), len(ev))
        preds = baseline_preds(pd.concat([fit_df, stop_df]), ev, cfg)
        for spec in specs:
            cols = feature_columns(feats, list(spec.windows), spec.with_odds)
            model = T.fit(spec, cols, fit_df, stop_df, rs, lgb_cfg)
            preds[spec.name] = T.predict(model, cols, ev)
        for name, p in preds.items():
            oof.append(pd.DataFrame({"match_id": ev["match_id"].to_numpy(), "fold": fold.name,
                                     "model": name, "p": p}))
            m = ~np.isnan(p)
            metrics.append({"fold": fold.name, "model": name, "n": int(m.sum()),
                            "log_loss": log_loss(ev["target"].to_numpy()[m], p[m])})
    return pd.concat(oof, ignore_index=True), pd.DataFrame(metrics)


def select_specs(fold_metrics: pd.DataFrame, specs: list[T.Spec]) -> dict[str, T.Spec]:
    """Best spec per (family, with_odds) by mean CV log loss. Returns label -> spec."""
    mean = fold_metrics.groupby("model")["log_loss"].mean()
    chosen = {}
    for spec in specs:
        if spec.label not in chosen or mean[spec.name] < mean[chosen[spec.label].name]:
            chosen[spec.label] = spec
    return chosen


def fit_bundle(feats: pd.DataFrame, cfg: dict, spec: T.Spec, fold: Fold, prefix: str) -> tuple[Bundle, dict]:
    """Fit on fold.fit_seasons (+stop), calibrate on fold.eval_season. Never touches later seasons."""
    fit_df, stop_df, cal = rows(feats, fold.fit_seasons), rows(feats, fold.stop_season), rows(feats, fold.eval_season)
    cols = feature_columns(feats, list(spec.windows), spec.with_odds)
    model = T.fit(spec, cols, fit_df, stop_df, cfg["models"]["random_state"], cfg["models"]["lightgbm"])
    p_raw = T.predict(model, cols, cal)
    y = cal["target"].to_numpy(int)
    order = cal["date"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    method, scores = choose_calibrator(p_raw, y, order)
    calibrator = Calibrator(method).fit(p_raw, y)
    market = MarketDerived().fit(pd.concat([fit_df, stop_df]).dropna(subset=["mkt_lambda"]))
    meta = {"version": new_version(prefix), "fit_seasons": list(fold.fit_seasons),
            "stop_season": fold.stop_season, "calibration_season": fold.eval_season,
            "calibration_method": method, "calibration_holdout_logloss": scores,
            "market_fh_share_s": market.s_, "label": spec.label, "spec_name": spec.name}
    return Bundle(spec, cols, model, calibrator, market, meta), {"p_raw": p_raw, "cal": cal}


def predict_bundle(bundle: Bundle, df: pd.DataFrame) -> np.ndarray:
    return bundle.calibrator.transform(T.predict(bundle.model, bundle.features, df))
