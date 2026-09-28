"""Predict upcoming fixtures and append them to predictions/log.csv BEFORE kick-off.

Refreshes current-season results (so rolling form is up to date), fetches fixtures
from the configured FixtureProvider, builds features through the backtest code
path, and merges any first-half odds from the configured OddsProvider.
"""
import argparse
import json
import logging

import pandas as pd

from fh.config import load_config, path
from fh.data.dataset import rebuild
from fh.data.download import download_seasons
from fh.data.odds_providers import get_odds_provider, merge_fh_odds
from fh.data.providers import get_fixture_provider
from fh.live.log import append_predictions, load_log, save_log
from fh.live.predict import prediction_rows
from fh.models import registry

log = logging.getLogger("predict_upcoming")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-refresh", action="store_true", help="skip re-downloading current-season results")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    cfg = load_config()
    now = pd.Timestamp.now(tz="UTC").floor("s")
    primary = cfg["live"]["primary_model"]

    if not args.no_refresh:
        current = cfg["seasons"]["current"]
        download_seasons(cfg, [current], refresh={current})
    history, _ = rebuild(cfg)

    live_dir = path(cfg["live"]["models_dir"]) / "live"
    manifest = json.loads(path("predictions/models_live_manifest.json").read_text())
    bundles = {label: registry.load(live_dir / registry.slug(label), m["version"])
               for label, m in manifest["models"].items()}

    fixtures = get_fixture_provider(cfg).get_fixtures()
    rows = prediction_rows(cfg, history, fixtures, bundles, primary)
    log_path = path(cfg["live"]["log_path"])
    log_df = load_log(log_path)
    if len(rows):
        log_df, stats = append_predictions(log_df, rows, now)
    else:
        stats = {"added": 0, "skipped_started": 0, "skipped_already_logged": 0}

    odds_provider = get_odds_provider(cfg)
    unsettled = log_df["target"].isna()
    merged, notes = merge_fh_odds(log_df[unsettled], odds_provider.get_fh_odds(log_df[unsettled]))
    log_df.loc[unsettled] = merged
    for w in getattr(odds_provider, "warnings", []):
        print(w)
    save_log(log_df, log_path)
    print(f"{now.isoformat()}: fixtures={len(fixtures)} added={stats['added']} "
          f"skipped_started={stats['skipped_started']} already_logged={stats['skipped_already_logged']} "
          f"fh_odds_notes={len(notes)} -> {log_path}")


if __name__ == "__main__":
    main()
