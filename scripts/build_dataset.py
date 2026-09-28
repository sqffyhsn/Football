"""Clean all cached raw CSVs into data/processed/matches.parquet (+ cleaning report)."""
import json
import logging

from fh.config import enabled_leagues, load_config, path
from fh.data.clean import build_dataset
from fh.data.download import raw_path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
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
    out = path(cfg["data"]["processed_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    rp = path(cfg["data"]["cleaning_report_path"])
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {len(df)} matches to {out}; report at {rp}")
