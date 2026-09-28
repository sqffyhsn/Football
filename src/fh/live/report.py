"""Live (forward-test) report: sample size, metrics, CLV, betting, convergence tracker."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fh.evaluation import plots  # noqa: E402
from fh.evaluation.betting import bets_needed, simulate  # noqa: E402
from fh.evaluation.metrics import log_loss  # noqa: E402
from fh.evaluation.plots import GRID, INK, INK2, SERIES, SURFACE, _style  # noqa: E402
from fh.evaluation.report import comparisons, fmt, metrics_table  # noqa: E402
from fh.live.settle import real_clv  # noqa: E402

MIN_CI_MATCHES = 200
CHECKPOINT = 50
LABELS = {"p_model": "Primary model", "p_logreg_form": "LogReg (form only)", "p_lgbm_form": "LightGBM (form only)",
          "p_lgbm_odds": "LightGBM (form+odds)", "p_baseline_a": "Baseline A: league rate",
          "p_baseline_b": "Baseline B: FH Poisson", "p_market_derived": "Market-derived (O/U 2.5)"}
INSUFFICIENT = "Insufficient data — {n} matches settled, ~200 needed for reliable CIs."
HOLDING = "HOLDING — edge vs market is significant (95% CI entirely below 0)."
INCONSISTENT = "INCONSISTENT WITH BACKTEST — edge shrinking/noise (95% CI excludes the backtest delta from above)."
INCONCLUSIVE = "INCONCLUSIVE — consistent with both zero and the backtest edge."


def _loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def convergence_status(n: int, ci_low: float, ci_high: float, backtest_delta: float) -> str:
    if n < MIN_CI_MATCHES:
        return INSUFFICIENT.format(n=n)
    if ci_high < 0:
        return HOLDING
    if ci_low > backtest_delta:
        return INCONSISTENT
    return INCONCLUSIVE


def _block_ci(d: np.ndarray, dates: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    """Date-block bootstrap CI of the mean of d."""
    codes, _ = pd.factorize(pd.Series(dates))
    sums = np.bincount(codes, weights=d)
    counts = np.bincount(codes).astype(float)
    pick = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def convergence(settled: pd.DataFrame, backtest_delta: float, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """One row per checkpoint (every 50 settled matches, plus the latest): running Δ log loss
    of the primary model vs the market-derived benchmark, with a CI once n >= 200."""
    d = settled.dropna(subset=["p_model", "p_market_derived", "target"]).copy()
    d["_ko"] = pd.to_datetime(d["kickoff_utc"], utc=True)
    d = d.sort_values(["_ko", "match_id"], kind="mergesort")
    y = d["target"].to_numpy(float)
    diff = _loss(y, d["p_model"].to_numpy(float)) - _loss(y, d["p_market_derived"].to_numpy(float))
    dates = d["date"].to_numpy()
    n_total = len(d)
    checkpoints = list(range(CHECKPOINT, n_total + 1, CHECKPOINT))
    if n_total and (not checkpoints or checkpoints[-1] != n_total):
        checkpoints.append(n_total)
    rng = np.random.default_rng(seed)
    rows = []
    for k in checkpoints:
        lo = hi = np.nan
        if k >= MIN_CI_MATCHES:
            lo, hi = _block_ci(diff[:k], dates[:k], n_boot, rng)
        rows.append({"n": k, "last_kickoff_utc": d["_ko"].iloc[k - 1].isoformat(),
                     "running_delta": float(diff[:k].mean()), "ci_low": lo, "ci_high": hi,
                     "status": convergence_status(k, lo, hi, backtest_delta)})
    return pd.DataFrame(rows, columns=["n", "last_kickoff_utc", "running_delta", "ci_low", "ci_high", "status"])


def convergence_plot(conv: pd.DataFrame, ref: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _style(ax)
    band = conv[conv["n"] >= MIN_CI_MATCHES]
    if len(band):
        ax.fill_between(band["n"], band["ci_low"], band["ci_high"], color=SERIES[0], alpha=0.18, linewidth=0,
                        label="95% CI (n ≥ 200)")
    ax.plot(conv["n"], conv["running_delta"], color=SERIES[0], linewidth=2, marker="o", markersize=4,
            label="Running Δ log loss (primary − market)")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.axhline(ref["backtest"], color=SERIES[1], linewidth=1.5, linestyle="--", label=f"Backtest Δ {ref['backtest']:+.5f}")
    if ref.get("test") is not None:
        ax.axhline(ref["test"], color=SERIES[2], linewidth=1.5, linestyle=":", label=f"Test-season Δ {ref['test']:+.5f}")
    ax.set_xlabel("Settled matches (kick-off order)", color=INK)
    ax.set_ylabel("Δ log loss (negative = model better)", color=INK)
    n = int(conv["n"].iloc[-1]) if len(conv) else 0
    title = "Convergence vs market-derived benchmark"
    if n < MIN_CI_MATCHES:
        title += f" — insufficient data ({n} settled)"
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def write_live_report(log_df: pd.DataFrame, manifest: dict, cfg: dict, out_dir: Path, now_utc: pd.Timestamp,
                      n_boot: int = 1000) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = manifest["reference"]
    bt_delta = ref["backtest_primary_vs_market"]["delta"]
    settled = log_df[log_df["status"] == "settled"].copy()
    n = len(settled)

    conv = convergence(settled, bt_delta, n_boot=n_boot)
    conv.to_csv(out_dir / "convergence.csv", index=False)
    latest = conv.iloc[-1] if len(conv) else pd.Series({"n": 0, "running_delta": np.nan, "ci_low": np.nan,
                                                         "ci_high": np.nan, "status": INSUFFICIENT.format(n=0)})
    hist_path = out_dir / "convergence_history.csv"
    hist_row = pd.DataFrame([{"run_utc": now_utc.isoformat(), "n_settled": int(latest["n"]),
                              "running_delta": latest["running_delta"], "ci_low": latest["ci_low"],
                              "ci_high": latest["ci_high"], "status": latest["status"]}])
    hist = pd.concat([pd.read_csv(hist_path), hist_row], ignore_index=True) if hist_path.exists() else hist_row
    hist.to_csv(hist_path, index=False)
    ref_lines = {"backtest": bt_delta, "test": ref.get("test_primary_vs_market", {}).get("delta")}
    if len(conv):
        convergence_plot(conv, ref_lines, out_dir / "convergence.png")

    still_needed = max(0, int(ref["matches_to_resolve_backtest_edge_80pct"]) - n)
    lines = [
        "# Live forward-test report",
        "",
        f"Updated {now_utc.isoformat()} · primary model `{manifest['primary_model']}` "
        f"(`{manifest['primary_version']}`)",
        "",
        "## Convergence: is the backtest edge holding?",
        f"**Status: {latest['status']}**",
        "",
    ]
    if int(latest["n"]) >= MIN_CI_MATCHES:
        lines.append(f"- Running Δ log loss (primary − market-derived): **{latest['running_delta']:+.5f}** "
                     f"[{latest['ci_low']:+.5f}, {latest['ci_high']:+.5f}] over {int(latest['n']):,} matches")
    else:
        lines.append(f"- Running Δ log loss (primary − market-derived): {latest['running_delta']:+.5f} "
                     f"over {int(latest['n'])} matches (no CI shown below {MIN_CI_MATCHES})"
                     if int(latest["n"]) else "- No settled matches yet.")
    lines += [
        f"- Reference: backtest Δ {bt_delta:+.5f} "
        f"[{ref['backtest_primary_vs_market']['ci_low']:+.5f}, {ref['backtest_primary_vs_market']['ci_high']:+.5f}]; "
        f"test-season Δ {ref['test_primary_vs_market']['delta']:+.5f} "
        f"[{ref['test_primary_vs_market']['ci_low']:+.5f}, {ref['test_primary_vs_market']['ci_high']:+.5f}]",
        f"- To resolve an edge the size of the backtest's at 80% power needs ≈ "
        f"{ref['matches_to_resolve_backtest_edge_80pct']:,} settled matches "
        f"(σ_d = {ref['sigma_loss_diff_per_match']:.4f} per match; n = ((z₀.₉₇₅ + z₀.₈₀)·σ_d / |δ|)²). "
        f"**Still needed: ≈ {still_needed:,}.**",
        "- Full trajectory: `convergence.csv` (every 50 matches); one row per settle run in `convergence_history.csv`.",
        "",
        "![convergence](convergence.png)" if len(conv) else "",
        "",
        "## Sample size",
        f"- Logged predictions: {len(log_df):,} · settled: {n:,} · pending: {int((log_df['status'] == 'pending').sum()):,}"
        f" · unmatched (3+ days past kick-off, e.g. postponed): {int((log_df['status'] == 'unmatched').sum()):,}",
    ]
    unmatched = log_df[log_df["status"] == "unmatched"]
    if len(unmatched):
        lines.append("  - Unmatched: " + ", ".join(f"{r.div} {r.date} {r.home} v {r.away}" for r in unmatched.itertuples()))
    if n:
        lines += ["", fmt(settled.groupby("div").agg(n=("match_id", "size"), fh_rate=("target", "mean")), 3)]

    cols = [c for c in LABELS if c in settled.columns and settled[c].notna().any()]
    ok = settled.dropna(subset=cols + ["target"]) if n else settled
    lines += ["", "## Probability metrics"]
    if len(ok) >= 30:
        y = ok["target"].to_numpy(float)
        preds = {LABELS[c]: ok[c].to_numpy(float) for c in cols}
        lines += [fmt(metrics_table(y, preds, cfg["evaluation"]["ece_bins"])[["n", "log_loss", "brier", "auc", "ece", "mean_p", "base_rate"]]),
                  "", "Δ log loss with date-block bootstrap 95% CI (negative = model better):", "",
                  fmt(comparisons(y, preds, [LABELS["p_model"]],
                                  [LABELS["p_market_derived"], LABELS["p_baseline_a"], LABELS["p_baseline_b"]],
                                  groups=ok["date"], n_boot=n_boot).set_index("model")
                      [["reference", "delta", "ci_low", "ci_high", "verdict"]], 5)]
        per_div = ok.groupby("div").apply(lambda g: pd.Series({
            "n": len(g), "primary": log_loss(g["target"], g["p_model"]),
            "market": log_loss(g["target"], g["p_market_derived"])}), include_groups=False)
        per_div["delta"] = per_div["primary"] - per_div["market"]
        lines += ["", "Per league:", "", fmt(per_div)]
        if len(ok) >= 100:
            plots.reliability_plot(y, {LABELS["p_market_derived"]: preds[LABELS["p_market_derived"]],
                                       LABELS["p_model"]: preds[LABELS["p_model"]]},
                                   out_dir / "reliability_live.png", "Reliability — live", cfg["evaluation"]["ece_bins"])
            lines += ["", "![reliability](reliability_live.png)"]
    else:
        lines.append(f"_Fewer than 30 settled matches ({len(ok)}); metrics not shown yet._")

    lines += ["", "## Closing line value", "",
              "### DERIVED / INDIRECT (no real FH prices)",
              "Change in the market-derived FH probability (from O/U 2.5) between prediction time and the close, "
              "signed so that positive = the market moved toward the model. Indirect evidence only."]
    clv = settled["clv_derived"].dropna().to_numpy(float) if n else np.array([])
    if len(clv) >= 30:
        rng = np.random.default_rng(1)
        boot = clv[rng.integers(0, len(clv), size=(n_boot, len(clv)))].mean(1)
        lines.append(f"- n = {len(clv):,}; mean = {clv.mean():+.5f} [{np.percentile(boot, 2.5):+.5f}, "
                     f"{np.percentile(boot, 97.5):+.5f}]; positive in {np.mean(clv > 0):.1%}")
    else:
        lines.append(f"- Only {len(clv)} matches with closing odds; shown from 30.")
    rc = real_clv(settled) if n else pd.DataFrame()
    lines += ["", "### REAL (real FH O/U 0.5 opening and closing odds)"]
    if len(rc):
        lines.append(f"- n = {len(rc):,}; mean CLV = {rc['clv_real'].mean():+.4f}; positive in {np.mean(rc['clv_real'] > 0):.1%}")
    else:
        lines.append("- Not available yet: no settled match has both real opening and closing FH odds.")

    bcfg = cfg["betting"]
    lines += ["", "## Betting (flat stakes) — REAL and HYPOTHETICAL are never pooled"]
    bet_ok = settled.dropna(subset=["p_model", "target"]) if n else settled
    if len(bet_ok):
        table, _ = simulate(bet_ok, bet_ok["p_model"].to_numpy(float), bet_ok["p_market_derived"].to_numpy(float),
                            bcfg["assumed_margin"], bcfg["edge_thresholds"], bcfg["stake"])
        both = table[table["side"] == "both"]
        for kind in ("REAL", "HYPOTHETICAL"):
            t = both[both["odds_kind"] == kind]
            lines += ["", f"### {kind}" + (" (synthetic odds from market-derived p + "
                                          f"{bcfg['assumed_margin']:.0%} margin)" if kind == "HYPOTHETICAL" else "")]
            lines.append(fmt(t.set_index("threshold")[["n_bets", "roi", "roi_ci_low", "roi_ci_high", "max_drawdown"]], 3)
                         if len(t) else "_No bets._")
    else:
        lines.append("_No settled matches yet._")

    pm = settled["p_market_derived"].dropna() if n else pd.Series(dtype=float)
    over_odds = float(1 / (pm.mean() * (1 + bcfg["assumed_margin"]))) if len(pm) else 1.35
    lines += ["", "## Power: how many bets to detect a 2.5% edge?",
              "One-sided test (α = 0.05, power 80%) of ROI = 2.5% vs 0 at decimal odds o: win probability "
              "p = 1.025 / o, per-bet profit SD σ = o·√(p(1−p)), n = ((z₀.₉₅ + z₀.₈₀)·σ / 0.025)².", "",
              f"- FH over 0.5 at typical odds {over_odds:.2f}: **n ≈ {bets_needed(0.025, over_odds):,} bets**",
              f"- For comparison — at odds 1.25: n ≈ {bets_needed(0.025, 1.25):,}; at odds 3.20 (under 0.5 side): "
              f"n ≈ {bets_needed(0.025, 3.20):,}"]
    (out_dir / "live_report.md").write_text("\n".join(lines) + "\n")
    return {"n_settled": n, "status": latest["status"]}
