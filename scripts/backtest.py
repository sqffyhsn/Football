"""Walk-forward backtest on development seasons + calibration fit. NEVER touches the test season.

Writes reports/backtest/ (or reports/_scratch/backtest/ with --synthetic) and
saves the calibrated model bundles to models/backtest/ for scripts/evaluate_test.py.
"""
import argparse
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fh.config import load_config, path
from fh.data.synthetic import make_synthetic
from fh.evaluation import plots
from fh.evaluation.betting import simulate
from fh.evaluation.report import comparisons, fmt, metrics_table, per_league
from fh.features.build import build_features
from fh.models import registry
from fh.models.pipeline import BASE_A, BASE_B, MARKET, fit_bundle, run_cv, select_specs
from fh.models.splits import calibration_fold, cv_folds
from fh.models.train import candidate_specs

log = logging.getLogger("backtest")


def load_matches(cfg, synthetic: bool) -> pd.DataFrame:
    if synthetic:
        return make_synthetic(n_leagues=3, n_teams=12, seasons=tuple(cfg["seasons"]["history"]), seed=7)
    p = path(cfg["data"]["processed_path"])
    if not p.exists():
        raise SystemExit("No processed dataset. Run download_data.py and build_dataset.py first.")
    return pd.read_parquet(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true", help="smoke-run on synthetic data (not a real result)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    cfg = load_config()
    out_dir = path("reports/_scratch/backtest" if args.synthetic else "reports/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    n_boot, n_bins = cfg["evaluation"]["bootstrap_samples"], cfg["evaluation"]["ece_bins"]
    test_season = cfg["splits"]["test"]

    matches = load_matches(cfg, args.synthetic)
    feats = build_features(matches, cfg["features"]["windows"], cfg["features"]["min_periods"])
    if not args.synthetic:
        feats.to_parquet(path("data/processed/features.parquet"), index=False)
    # Hard guard: the test season (and anything after) is removed before any modelling.
    order = cfg["seasons"]["history"]
    allowed = order[: order.index(test_season)]
    feats = feats[feats["season"].isin(allowed)].reset_index(drop=True)
    assert test_season not in set(feats["season"])

    # ---- walk-forward CV -------------------------------------------------------
    folds, specs = cv_folds(cfg), candidate_specs(cfg)
    oof, fold_metrics = run_cv(feats, cfg, folds, specs)
    chosen = select_specs(fold_metrics, specs)
    mean_ll = fold_metrics.groupby("model")["log_loss"].agg(["mean", "std"]).sort_values("mean")
    mean_ll.to_csv(out_dir / "cv_all_candidates.csv")

    names = {BASE_A: BASE_A, BASE_B: BASE_B, MARKET: MARKET, **{lbl: s.name for lbl, s in chosen.items()}}
    wide = oof[oof["model"].isin(names.values())].pivot_table(index="match_id", columns="model", values="p")
    wide = wide.rename(columns={v: k for k, v in names.items()})
    ev = feats.set_index("match_id").loc[wide.index].reset_index()
    ev = ev.assign(**{c: wide[c].to_numpy() for c in wide.columns})
    ev = ev.sort_values("date", kind="mergesort").reset_index(drop=True)
    model_labels = sorted(chosen)
    all_labels = [BASE_A, BASE_B] + model_labels

    y = ev["target"].to_numpy()
    preds = {k: ev[k].to_numpy() for k in [BASE_A, BASE_B, MARKET] + model_labels}
    tab_all = metrics_table(y, {k: preds[k] for k in all_labels}, n_bins)
    cmp_all = comparisons(y, preds, model_labels, [BASE_A, BASE_B], groups=ev["date"], n_boot=n_boot)

    mk = ~np.isnan(preds[MARKET])
    ev_mk = ev[mk].reset_index(drop=True)
    preds_mk = {k: v[mk] for k, v in preds.items()}
    tab_mk = metrics_table(y[mk], {k: preds_mk[k] for k in [MARKET] + all_labels}, n_bins)
    cmp_mk = comparisons(y[mk], preds_mk, model_labels, [MARKET], groups=ev_mk["date"], n_boot=n_boot)

    league_tab = per_league(ev_mk, preds_mk, [MARKET, BASE_A] + model_labels)
    league_tab.to_csv(out_dir / "per_league_cv.csv")

    # ---- plots -----------------------------------------------------------------
    plots.reliability_plot(y[mk], {MARKET: preds_mk[MARKET], **{k: preds_mk[k] for k in model_labels}},
                           out_dir / "reliability_cv_models.png", "Reliability — walk-forward CV (rows with odds)", n_bins)
    plots.reliability_plot(y, {k: preds[k] for k in [BASE_A, BASE_B]},
                           out_dir / "reliability_cv_baselines.png", "Reliability — baselines, walk-forward CV", n_bins)
    best = min(model_labels, key=lambda k: tab_mk.loc[k, "log_loss"])
    plots.league_delta_plot((league_tab[best] - league_tab[MARKET]).to_frame("delta"),
                            out_dir / "league_delta_vs_market.png", f"{best} vs market-derived, by league (CV)")

    # ---- hypothetical betting sim on CV out-of-fold predictions ------------------
    bcfg = cfg["betting"]
    bet_tables, curves = [], {}
    for k in model_labels:
        t, all_b = simulate(ev_mk, preds_mk[k], preds_mk[MARKET], bcfg["assumed_margin"],
                            bcfg["edge_thresholds"], bcfg["stake"])
        bet_tables.append(t.assign(model=k))
        b = all_b.get(("HYPOTHETICAL", bcfg["edge_thresholds"][0]))
        if b is not None and len(b):
            curves[k] = b["profit"].cumsum()
    bet_tab = pd.concat(bet_tables, ignore_index=True)
    bet_tab.to_csv(out_dir / "betting_sim_cv.csv", index=False)
    if curves:
        plots.pnl_plot(curves, out_dir / "pnl_cv_hypothetical.png",
                       f"HYPOTHETICAL P&L (synthetic odds, {bcfg['assumed_margin']:.0%} margin, edge>{bcfg['edge_thresholds'][0]})")

    # ---- calibration fold: fit final backtest bundles --------------------------
    cal_fold = calibration_fold(cfg)
    bundles_dir = path("models/_scratch_backtest" if args.synthetic else "models/backtest")
    cal_rows = []
    for k in model_labels:
        bundle, info = fit_bundle(feats, cfg, chosen[k], cal_fold, prefix="backtest")
        registry.save(bundle, bundles_dir / k.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_"))
        yc = info["cal"]["target"].to_numpy()
        cal_rows.append({"model": k, "raw_logloss_on_cal_season": metrics_table(yc, {k: info["p_raw"]}).loc[k, "log_loss"],
                         "chosen_calibration": bundle.meta["calibration_method"],
                         **{f"holdout_ll_{m}": v for m, v in bundle.meta["calibration_holdout_logloss"].items()},
                         "market_s": bundle.market.s_})
    cal_tab = pd.DataFrame(cal_rows).set_index("model")

    # ---- write summary -----------------------------------------------------------
    coverage = feats.groupby("season").agg(matches=("match_id", "size"), fh_rate=("target", "mean"),
                                           odds_coverage=("mkt_lambda", lambda s: s.notna().mean()))
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic": args.synthetic, "cv_folds": [f.__dict__ for f in folds],
        "selected_specs": {k: s.name for k, s in chosen.items()},
        "metrics_all_rows": tab_all.to_dict(orient="index"),
        "metrics_odds_rows": tab_mk.to_dict(orient="index"),
        "comparisons_vs_baselines": cmp_all.to_dict(orient="records"),
        "comparisons_vs_market": cmp_mk.to_dict(orient="records"),
        "calibration": cal_tab.to_dict(orient="index"),
    }
    (out_dir / "backtest_metrics.json").write_text(json.dumps(summary, indent=2, default=str))

    hyp = bet_tab[(bet_tab["odds_kind"] == "HYPOTHETICAL") & (bet_tab["side"] == "both")]
    md = [
        "# Backtest report — walk-forward CV (test season NOT evaluated)",
        "",
        "> **SYNTHETIC DATA SMOKE RUN — these numbers mean nothing.**" if args.synthetic else "",
        f"Generated {summary['generated_utc']}. Test season `{test_season}` was excluded before any modelling.",
        "",
        "## Setup",
        f"- CV folds (expanding window): " + ", ".join(
            f"eval `{f.eval_season}` (fit ≤`{f.fit_seasons[-1]}`, early-stop `{f.stop_season}`)" for f in folds),
        f"- Out-of-fold rows: {len(ev):,} (with market odds: {int(mk.sum()):,})",
        f"- Candidates evaluated: {len(specs)}; best per family by mean CV log loss:",
        *[f"  - **{k}**: `{s.name}`" for k, s in chosen.items()],
        "",
        "## Data coverage (development seasons)",
        fmt(coverage, 3),
        "",
        "## All out-of-fold rows — models vs baselines",
        fmt(tab_all[["n", "log_loss", "brier", "auc", "ece", "mean_p", "base_rate"]]),
        "",
        "Paired block-bootstrap (by date) of Δ log loss = model − reference (negative = model better):",
        "",
        fmt(cmp_all.set_index("model")[["reference", "delta", "ci_low", "ci_high", "verdict"]], 5),
        "",
        "## Rows with bookmaker odds — models vs market-derived benchmark",
        "The market benchmark is **DERIVED**, not a quoted FH price: λ from de-vigged O/U 2.5, "
        "p = 1 − exp(−s·λ) with s fitted on each fold's training seasons.",
        "",
        fmt(tab_mk[["n", "log_loss", "brier", "auc", "ece", "mean_p", "base_rate"]]),
        "",
        fmt(cmp_mk.set_index("model")[["reference", "delta", "ci_low", "ci_high", "verdict"]], 5),
        "",
        "## Per-league log loss (CV, rows with odds)",
        fmt(league_tab),
        "",
        "![by league](league_delta_vs_market.png)",
        "",
        "## Calibration",
        "Model fitted on seasons before the calibration season; calibrator chosen by fitting on the first half "
        "of the calibration season and scoring the second half, then refitted on the full season.",
        "",
        fmt(cal_tab),
        "",
        "![reliability models](reliability_cv_models.png)",
        "![reliability baselines](reliability_cv_baselines.png)",
        "",
        "## Betting simulation — HYPOTHETICAL",
        f"**No real first-half O/U 0.5 odds exist in this dataset.** Bets are priced at SYNTHETIC odds = "
        f"1 / (p_market_derived × (1 + {bcfg['assumed_margin']})). Because the price is itself derived from the "
        "market benchmark, this only shows whether the model disagrees with the market profitably *under that "
        "assumption*. It is not evidence of a real-world edge. Informational only.",
        "",
        fmt(hyp.set_index("model")[["threshold", "n_bets", "roi", "roi_ci_low", "roi_ci_high", "max_drawdown", "mean_odds"]], 3),
        "",
        "![pnl](pnl_cv_hypothetical.png)" if curves else "_No bets placed at the lowest threshold._",
        "",
        "## Verdict (computed from the numbers above)",
        *[f"- {r.model} vs {r.reference}: Δ={r.delta:+.5f} [{r.ci_low:+.5f}, {r.ci_high:+.5f}] → {r.verdict}"
          for r in pd.concat([cmp_all, cmp_mk]).itertuples()],
    ]
    (out_dir / "metrics_summary.md").write_text("\n".join(md) + "\n")
    print(f"backtest report written to {out_dir}")


if __name__ == "__main__":
    main()
