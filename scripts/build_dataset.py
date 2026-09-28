"""Clean all cached raw CSVs into data/processed/matches.parquet (+ cleaning report)."""
import json
import logging

from fh.config import load_config, path
from fh.data.dataset import rebuild

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    df, report = rebuild(cfg)
    rp = path(cfg["data"]["cleaning_report_path"])
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {len(df)} matches to {path(cfg['data']['processed_path'])}; report at {rp}")
