"""Download football-data.co.uk CSVs into a local cache (data/raw/, gitignored)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from fh.config import enabled_leagues, path

log = logging.getLogger(__name__)

USER_AGENT = "fh-research/0.1 (personal research; cached downloads)"


def raw_path(cfg: dict, season: str, div: str) -> Path:
    return path(cfg["data"]["raw_dir"]) / season / f"{div}.csv"


def fetch(url: str, dest: Path, retries: int = 3, timeout: int = 30) -> bool:
    """Fetch url to dest. Returns False on 404 (e.g. a league not published that season)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if resp.status_code == 404:
                log.warning("404 for %s", url)
                return False
            resp.raise_for_status()
            if not resp.content.strip():
                log.warning("empty response for %s", url)
                return False
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(resp.content)
            tmp.replace(dest)
            return True
        except requests.RequestException as exc:
            wait = 2 ** (attempt + 1)
            log.warning("download failed (%s), retrying in %ss: %s", url, wait, exc)
            time.sleep(wait)
    raise RuntimeError(f"could not download {url} after {retries} attempts")


def download_seasons(cfg: dict, seasons: list[str], force: bool = False,
                     refresh: set[str] | None = None) -> list[Path]:
    """Download every enabled league for the given seasons.

    Cached files are reused unless `force`, or the season is in `refresh`
    (the in-progress season must be re-fetched to pick up new results).
    """
    refresh = refresh or set()
    got = []
    for season in seasons:
        for div in enabled_leagues(cfg):
            dest = raw_path(cfg, season, div)
            if dest.exists() and not force and season not in refresh:
                got.append(dest)
                continue
            url = cfg["data"]["base_url"].format(season=season, div=div)
            log.info("downloading %s", url)
            if fetch(url, dest):
                got.append(dest)
            time.sleep(0.5)  # be polite to the host
    return got


def download_fixtures(cfg: dict) -> Path:
    dest = path(cfg["data"]["raw_dir"]) / "fixtures.csv"
    if not fetch(cfg["data"]["fixtures_url"], dest):
        raise RuntimeError("fixtures file not available")
    return dest
