"""Download historical + current-season CSVs for enabled leagues into data/raw/."""
import argparse
import logging

from fh.config import load_config
from fh.data.download import download_seasons

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download cached files")
    ap.add_argument("--no-current", action="store_true", help="skip the in-progress season")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    seasons = list(cfg["seasons"]["history"])
    if not args.no_current:
        seasons.append(cfg["seasons"]["current"])
    files = download_seasons(cfg, seasons, force=args.force, refresh={cfg["seasons"]["current"]})
    print(f"{len(files)} files available in {cfg['data']['raw_dir']}")
