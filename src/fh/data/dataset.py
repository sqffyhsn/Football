"""Rebuild the tidy match table from all cached raw files."""
from __future__ import annotations

import pandas as pd

from fh.config import enabled_leagues, path
from fh.data.clean import build_dataset
from fh.data.download import raw_path


def rebuild(cfg: dict, write: bool = True) -> tuple[pd.DataFrame, dict]:
    seasons = list(cfg["seasons"]["history"]) + [cfg["seasons"]["current"]]
    files, missing = [], []
    for season in seasons:
        for div in enabled_leagues(cfg):
            p = raw_path(cfg, season, div)
            (files if p.exists() else missing).append((season, div, p))
    if not files:
        raise SystemExit("No raw files found. Run scripts/download_data.py first.")
    df, report = build_dataset(files)
    report["missing_files"] = [f"{s}/{d}" for s, d, _ in missing]
    report["rows_by_season"] = df.groupby("season").size().to_dict()
    report["rows_by_div"] = df.groupby("div").size().to_dict()
    if write:
        out = path(cfg["data"]["processed_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    return df, report
