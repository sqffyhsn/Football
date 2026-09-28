"""Settle logged predictions with actual results + closing odds, then refresh the live report."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from fh.config import load_config, path
from fh.data.odds_providers import get_odds_provider, merge_fh_odds
from fh.data.providers import FixtureProvider, get_fixture_provider
from fh.live.log import load_log, save_log
from fh.live.report import write_live_report
from fh.live.settle import settle

log = logging.getLogger("settle_results")


def run(cfg: dict, provider: FixtureProvider, log_path: Path, manifest: dict, out_dir: Path,
        now_utc: pd.Timestamp, odds_provider=None, n_boot: int = 1000) -> dict:
    log_df = load_log(log_path)
    if odds_provider is not None:  # late closing FH odds typed in after kick-off
        open_ = log_df["target"].isna()
        merged, _ = merge_fh_odds(log_df[open_], odds_provider.get_fh_odds(log_df[open_]))
        log_df.loc[open_] = merged
    seasons = sorted(set(log_df.loc[log_df["target"].isna(), "season"].dropna()))
    results = pd.concat([provider.get_results(s) for s in seasons], ignore_index=True) if seasons else pd.DataFrame()
    s = manifest["models"][manifest["primary_model"]]["market_fh_share_s"]
    if len(results):
        log_df, info = settle(log_df, results, s, now_utc)
    else:
        info = {"newly_settled": 0}
    save_log(log_df, log_path)
    info.update(write_live_report(log_df, manifest, cfg, out_dir, now_utc, n_boot=n_boot))
    return info


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    cfg = load_config()
    manifest = json.loads(path("predictions/models_live_manifest.json").read_text())
    info = run(cfg, get_fixture_provider(cfg), path(cfg["live"]["log_path"]), manifest,
               path(cfg["live"]["reports_dir"]), pd.Timestamp.now(tz="UTC").floor("s"), get_odds_provider(cfg))
    print(info)


if __name__ == "__main__":
    main()
