"""Fixture/result providers. Swap the source via FIXTURE_PROVIDER in .env.

Every provider returns frames in the tidy schema of `fh.data.clean.TIDY_COLUMNS`
(results NaN for fixtures), with team names in football-data.co.uk spelling so
features line up with history. Providers using other spellings should map
through data/team_aliases.csv.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from fh.config import ROOT, enabled_leagues, env
from fh.data.clean import normalise, read_raw_csv
from fh.data.download import download_fixtures, download_seasons, raw_path


class FixtureProvider(Protocol):
    name: str

    def get_fixtures(self) -> pd.DataFrame: ...

    def get_results(self, season: str) -> pd.DataFrame: ...


class FootballDataProvider:
    name = "football_data"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def get_fixtures(self) -> pd.DataFrame:
        p = download_fixtures(self.cfg)
        df = normalise(read_raw_csv(p), season=self.cfg["seasons"]["current"], require_results=False)
        return df[df["div"].isin(enabled_leagues(self.cfg))].reset_index(drop=True)

    def get_results(self, season: str) -> pd.DataFrame:
        download_seasons(self.cfg, [season], refresh={season})
        frames = []
        for div in enabled_leagues(self.cfg):
            p = raw_path(self.cfg, season, div)
            if p.exists():
                frames.append(normalise(read_raw_csv(p), season=season, div=div))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class ApiFootballProvider:
    """Placeholder for a paid API (e.g. API-Football). Not implemented yet.

    To implement: fetch fixtures/results with API_FOOTBALL_KEY, map league ids to
    football-data `div` codes, map team names via data/team_aliases.csv, and
    return the tidy schema.
    """
    name = "api_football"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.key = env("API_FOOTBALL_KEY")
        if not self.key:
            raise RuntimeError("API_FOOTBALL_KEY is not set in .env")

    def get_fixtures(self) -> pd.DataFrame:
        raise NotImplementedError("ApiFootballProvider is a stub; see class docstring")

    def get_results(self, season: str) -> pd.DataFrame:
        raise NotImplementedError("ApiFootballProvider is a stub; see class docstring")


def load_team_aliases(source: str, path: Path | None = None) -> dict[str, str]:
    """Map provider team names -> football-data names for one source."""
    p = path or ROOT / "data" / "team_aliases.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    df = df[df["source"] == source]
    return dict(zip(df["provider_name"], df["football_data_name"]))


def get_fixture_provider(cfg: dict) -> FixtureProvider:
    name = env("FIXTURE_PROVIDER", "football_data")
    providers = {"football_data": FootballDataProvider, "api_football": ApiFootballProvider}
    if name not in providers:
        raise ValueError(f"unknown FIXTURE_PROVIDER {name!r}; choose from {sorted(providers)}")
    return providers[name](cfg)
