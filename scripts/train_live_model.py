"""Fit the live model bundles: the backtest recipe shifted forward one season.

Fit on seasons up to live_fold.fit_seasons (+ stop season), calibrate on the last
full season. The specs are the ones selected in the backtest: no re-selection.
Writes models/live/<model>/ (gitignored) and predictions/models_live_manifest.json.
"""
import json
import logging
import math
from datetime import datetime, timezone

import pandas as pd

from fh.config import load_config, path
from fh.features.build import build_features
from fh.models import registry
from fh.models.pipeline import fit_bundle
from fh.models.splits import live_fold
from fh.models.train import Spec

log = logging.getLogger("train_live_model")
Z_975, Z_80 = 1.959964, 0.841621


def spec_from_name(name: str, specs_json: dict) -> Spec:
    return Spec(**specs_json[name])


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    cfg = load_config()
    primary = cfg["live"]["primary_model"]
    backtest = json.loads(path("reports/backtest/backtest_metrics.json").read_text())
    test = json.loads(path("reports/test/test_metrics.json").read_text())
    bt_bundles = {label: registry.load(path("models/backtest") / registry.slug(label))
                  for label in backtest["selected_specs"]}

    feats = build_features(pd.read_parquet(path(cfg["data"]["processed_path"])),
                           cfg["features"]["windows"], cfg["features"]["min_periods"])
    fold = live_fold(cfg)
    manifest_models = {}
    for label, bt in bt_bundles.items():
        bundle, _ = fit_bundle(feats, cfg, bt.spec, fold, prefix="live")
        registry.save(bundle, path(cfg["live"]["models_dir"]) / "live" / registry.slug(label))
        manifest_models[label] = {k: bundle.meta[k] for k in
                                  ("version", "fit_seasons", "stop_season", "calibration_season",
                                   "calibration_method", "market_fh_share_s", "spec_name")}
        log.info("%s -> %s (%s)", label, bundle.version, bundle.meta["calibration_method"])

    bt = next(r for r in backtest["comparisons_vs_market"] if r["model"] == primary)
    n_bt = int(backtest["metrics_odds_rows"][primary]["n"])
    # Per-match SD of the loss difference, backed out of the block-bootstrap CI width.
    sigma_d = (bt["ci_high"] - bt["ci_low"]) / (2 * Z_975) * math.sqrt(n_bt)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "primary_model": primary,
        "primary_version": manifest_models[primary]["version"],
        "models": manifest_models,
        "reference": {
            "backtest_primary_vs_market": {k: bt[k] for k in ("delta", "ci_low", "ci_high")} | {"n": n_bt},
            "test_primary_vs_market": {k: test["primary_vs_market"][k] for k in ("delta", "ci_low", "ci_high")},
            "test_interpretation": test["interpretation"],
            "sigma_loss_diff_per_match": sigma_d,
            "matches_to_resolve_backtest_edge_80pct": math.ceil(((Z_975 + Z_80) * sigma_d / abs(bt["delta"])) ** 2),
        },
    }
    out = path("predictions/models_live_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["reference"], indent=2))


if __name__ == "__main__":
    main()
