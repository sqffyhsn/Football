"""ONE-TIME evaluation on the held-out test season. Refuses to run twice.

Uses the backtest bundles exactly as fitted (fit <= stop season, calibrated on the
calibration season). Nothing is fitted or tuned on test data.
"""
import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fh.config import load_config, path
from fh.evaluation import plots
from fh.evaluation.betting import simulate
from fh.evaluation.report import (ci_position, comparisons, fmt, metrics_table, per_league,
                                  test_interpretation)
from fh.features.build import build_features
from fh.models import registry
from fh.models import train as T
from fh.models.baselines import fh_poisson, league_base_rate
from fh.models.pipeline import BASE_A, BASE_B, MARKET, predict_bundle
from fh.models.splits import rows

log = logging.getLogger("evaluate_test")


def marker_path(out_dir: Path) -> Path:
    return out_dir / "TEST_EVALUATED"


def check_not_evaluated(out_dir: Path) -> None:
    m = marker_path(out_dir)
    if m.exists():
        raise SystemExit(f"Test season already evaluated ({m}). This script runs once only.\n{m.read_text()}")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    cfg = load_config()
    out_dir = path("reports/test")
    check_not_evaluated(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_season = cfg["splits"]["test"]
    primary = cfg["live"]["primary_model"]
    n_boot, n_bins = cfg["evaluation"]["bootstrap_samples"], cfg["evaluation"]["ece_bins"]

    selected = json.loads(path("reports/backtest/backtest_metrics.json").read_text())["selected_specs"]
    bundles = {label: registry.load(path("models/backtest") / registry.slug(label)) for label in selected}
    for label, b in bundles.items():
        assert b.spec.name == selected[label], f"bundle/spec mismatch for {label}"
        assert test_season not in b.meta["fit_seasons"] + [b.meta["stop_season"], b.meta["calibration_season"]]

    matches = pd.read_parquet(path(cfg["data"]["processed_path"]))
    feats = build_features(matches, cfg["features"]["windows"], cfg["features"]["min_periods"])
    ev = rows(feats, test_season).sort_values("date", kind="mergesort").reset_index(drop=True)
    y = ev["target"].to_numpy()
    log.info("test rows: %d", len(ev))

    market = bundles[primary].market  # s fitted on pre-calibration seasons only
    a = league_base_rate(ev)
    preds = {BASE_A: np.where(np.isnan(a), np.nanmean(a), a),
             BASE_B: fh_poisson(ev, window=max(cfg["features"]["windows"]),
                                shrink=cfg["features"]["poisson_shrinkage_matches"]),
             MARKET: market.predict(ev)}
    raw = {}
    for label, b in bundles.items():
        preds[label] = predict_bundle(b, ev)
        raw[label] = T.predict(b.model, b.features, ev)
    labels = sorted(bundles)

    tab_all = metrics_table(y, {k: preds[k] for k in [BASE_A, BASE_B] + labels}, n_bins)
    cmp_all = comparisons(y, preds, labels, [BASE_A, BASE_B], groups=ev["date"], n_boot=n_boot)
    mk = ~np.isnan(preds[MARKET])
    ev_mk, y_mk = ev[mk].reset_index(drop=True), y[mk]
    preds_mk = {k: v[mk] for k, v in preds.items()}
    tab_mk = metrics_table(y_mk, {k: preds_mk[k] for k in [MARKET, BASE_A, BASE_B] + labels}, n_bins)
    cmp_mk = comparisons(y_mk, preds_mk, labels, [MARKET], groups=ev_mk["date"], n_boot=n_boot)
    prim = cmp_mk[cmp_mk["model"] == primary].iloc[0].to_dict()
    interpretation = test_interpretation(prim)

    raw_vs_cal = pd.DataFrame({label: {"raw_log_loss": metrics_table(y, {label: raw[label]}).loc[label, "log_loss"],
                                       "calibrated_log_loss": tab_all.loc[label, "log_loss"],
                                       "raw_ece": metrics_table(y, {label: raw[label]}).loc[label, "ece"],
                                       "calibrated_ece": tab_all.loc[label, "ece"],
                                       "method": bundles[label].meta["calibration_method"]}
                               for label in labels}).T
    league_tab = per_league(ev_mk, preds_mk, [MARKET, BASE_A] + labels)
    league_tab.to_csv(out_dir / "per_league_test.csv")

    plots.reliability_plot(y_mk, {MARKET: preds_mk[MARKET], **{k: preds_mk[k] for k in labels}},
                           out_dir / "reliability_test_models.png", f"Reliability — test season {test_season}", n_bins)
    plots.league_delta_plot((league_tab[primary] - league_tab[MARKET]).to_frame("delta"),
                            out_dir / "league_delta_vs_market.png", f"{primary} vs market-derived, by league (test)")

    bcfg = cfg["betting"]
    bet_tables = []
    for k in labels:
        t, _ = simulate(ev_mk, preds_mk[k], preds_mk[MARKET], bcfg["assumed_margin"],
                        bcfg["edge_thresholds"], bcfg["stake"])
        bet_tables.append(t.assign(model=k))
    bet_tab = pd.concat(bet_tables, ignore_index=True)
    bet_tab.to_csv(out_dir / "betting_sim_test.csv", index=False)
    hyp = bet_tab[(bet_tab["odds_kind"] == "HYPOTHETICAL") & (bet_tab["side"] == "both")]

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    backtest = json.loads(path("reports/backtest/backtest_metrics.json").read_text())
    bt_prim = next(r for r in backtest["comparisons_vs_market"] if r["model"] == primary)
    summary = {
        "generated_utc": now, "test_season": test_season, "git_commit": git_commit(),
        "bundle_versions": {k: b.version for k, b in bundles.items()},
        "primary_model": primary, "primary_vs_market": prim, "interpretation": interpretation,
        "backtest_primary_vs_market": bt_prim,
        "metrics_all_rows": tab_all.to_dict(orient="index"),
        "metrics_odds_rows": tab_mk.to_dict(orient="index"),
        "comparisons_vs_baselines": cmp_all.to_dict(orient="records"),
        "comparisons_vs_market": cmp_mk.to_dict(orient="records"),
        "raw_vs_calibrated": raw_vs_cal.to_dict(orient="index"),
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2, default=str))

    md = [
        f"# Test-season evaluation — {test_season} (run once)",
        "",
        f"Generated {now} at commit `{summary['git_commit'][:10]}`. Bundles fitted on seasons up to "
        f"`{bundles[primary].meta['stop_season']}`, calibrated on `{bundles[primary].meta['calibration_season']}`; "
        "nothing was fitted or tuned on the test season.",
        "",
        "## Pre-registered primary comparison",
        f"**{primary} vs market-derived benchmark** (chosen from the backtest before this season was seen).",
        "",
        f"- Backtest (CV): Δ log loss = {bt_prim['delta']:+.5f} [{bt_prim['ci_low']:+.5f}, {bt_prim['ci_high']:+.5f}]",
        f"- **Test: Δ log loss = {prim['delta']:+.5f} [{prim['ci_low']:+.5f}, {prim['ci_high']:+.5f}]** "
        f"(95% CI {ci_position(prim)} 0; n = {int(mk.sum()):,})",
        "",
        f"> **Interpretation (auto-generated from the CI):** {interpretation}",
        "",
        "## All test rows — models vs baselines",
        fmt(tab_all[["n", "log_loss", "brier", "auc", "ece", "mean_p", "base_rate"]]),
        "",
        fmt(cmp_all.set_index("model")[["reference", "delta", "ci_low", "ci_high", "verdict"]], 5),
        "",
        "## Rows with odds — models vs market-derived benchmark (DERIVED, not a quoted FH price)",
        fmt(tab_mk[["n", "log_loss", "brier", "auc", "ece", "mean_p", "base_rate"]]),
        "",
        fmt(cmp_mk.set_index("model")[["reference", "delta", "ci_low", "ci_high", "verdict"]], 5),
        "",
        "## Per-league log loss (test)",
        fmt(league_tab),
        "",
        "![by league](league_delta_vs_market.png)",
        "",
        "## Calibration: raw vs calibrated on the test season",
        fmt(raw_vs_cal),
        "",
        "![reliability](reliability_test_models.png)",
        "",
        "## Betting simulation — HYPOTHETICAL",
        "**No real FH O/U 0.5 odds.** Synthetic odds = 1 / (p_market_derived × (1 + "
        f"{bcfg['assumed_margin']})). Informational only; not evidence of a real-world edge.",
        "",
        fmt(hyp.set_index("model")[["threshold", "n_bets", "roi", "roi_ci_low", "roi_ci_high", "max_drawdown", "mean_odds"]], 3),
        "",
        "## All verdicts (computed)",
        *[f"- {r.model} vs {r.reference}: Δ={r.delta:+.5f} [{r.ci_low:+.5f}, {r.ci_high:+.5f}] → {r.verdict}"
          for r in pd.concat([cmp_all, cmp_mk]).itertuples()],
    ]
    (out_dir / "metrics_summary.md").write_text("\n".join(md) + "\n")
    marker_path(out_dir).write_text(json.dumps({
        "evaluated_utc": now, "git_commit": summary["git_commit"], "bundle_versions": summary["bundle_versions"],
        "primary_vs_market": {k: prim[k] for k in ("delta", "ci_low", "ci_high")},
        "interpretation": interpretation}, indent=2) + "\n")
    print("\n".join(md[:14]))


if __name__ == "__main__":
    main()
