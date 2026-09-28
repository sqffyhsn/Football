"""First-half O/U 0.5 odds providers (separate from fixture providers).

Select with ODDS_PROVIDER in .env: none (default) | manual_csv | api.
Every provider returns MANUAL_COLUMNS; `merge_fh_odds` puts them into the log.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from fh.config import env, path

log = logging.getLogger(__name__)

OPEN = ["fh_o05_odds", "fh_u05_odds"]
CLOSE = ["fh_o05_close_odds", "fh_u05_close_odds"]
MANUAL_COLUMNS = ["div", "date", "home", "away", *OPEN, "source", "ts_utc", *CLOSE]
KEY = ["div", "date", "home", "away"]
OVERROUND_RANGE = (1.00, 1.20)


class OddsProvider(Protocol):
    name: str

    def get_fh_odds(self, fixtures: pd.DataFrame) -> pd.DataFrame: ...


class NullOddsProvider:
    name = "none"

    def get_fh_odds(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(columns=MANUAL_COLUMNS)


def _pair_problem(o, u) -> str | None:
    """None if the (over, under) pair is valid; otherwise the reason."""
    if pd.isna(o) and pd.isna(u):
        return None
    if pd.isna(o) or pd.isna(u):
        return "only one side of the pair given"
    if o <= 1.0 or u <= 1.0:
        return f"odds must be > 1.0 (got {o}, {u})"
    book = 1 / o + 1 / u
    if not OVERROUND_RANGE[0] <= book <= OVERROUND_RANGE[1]:
        return f"implied probability sum {book:.3f} outside {OVERROUND_RANGE}"
    return None


def validate_manual(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (valid rows, warnings). Every rejected row produces a warning naming it."""
    warnings, keep = [], []
    for i, r in df.iterrows():
        label = f"row {i + 2} ({r.get('div')} {r.get('date')} {r.get('home')} v {r.get('away')})"
        problems = []
        if any(pd.isna(r.get(k)) or str(r.get(k)).strip() == "" for k in KEY):
            problems.append("missing div/date/home/away")
        for k in ("source", "ts_utc"):
            if pd.isna(r.get(k)) or str(r.get(k)).strip() == "":
                problems.append(f"{k} is empty")
        if pd.isna(pd.to_datetime(r.get("ts_utc"), utc=True, errors="coerce")) and not pd.isna(r.get("ts_utc")):
            problems.append("ts_utc is not a valid timestamp")
        if all(pd.isna(r.get(c)) for c in OPEN + CLOSE):
            problems.append("no odds given")
        for cols, what in ((OPEN, "pre-match"), (CLOSE, "closing")):
            p = _pair_problem(r.get(cols[0]), r.get(cols[1]))
            if p:
                problems.append(f"{what} odds: {p}")
        if problems:
            warnings.append(f"REJECTED manual FH odds {label}: " + "; ".join(problems))
        else:
            keep.append(i)
    for w in warnings:
        log.warning(w)
    return df.loc[keep].reset_index(drop=True), warnings


class ManualCsvOddsProvider:
    """Odds typed in by hand from a bookmaker before kick-off (predictions/manual_fh_odds.csv)."""
    name = "manual_csv"

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.warnings: list[str] = []

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame(columns=MANUAL_COLUMNS)
        df = pd.read_csv(self.csv_path, dtype={"div": str, "home": str, "away": str, "source": str, "ts_utc": str})
        for c in MANUAL_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        for c in OPEN + CLOSE:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        valid, self.warnings = validate_manual(df[MANUAL_COLUMNS])
        return valid

    def get_fh_odds(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        return self.load()


class ApiOddsProvider:
    """Placeholder for a paid odds API (e.g. API-Football, Betfair). Not implemented.

    To implement: fetch FH O/U 0.5 prices for `fixtures` with ODDS_API_KEY, map team
    names via data/team_aliases.csv and return MANUAL_COLUMNS with ts_utc = fetch time.
    """
    name = "api"

    def __init__(self):
        if not env("ODDS_API_KEY"):
            raise RuntimeError("ODDS_API_KEY is not set in .env")

    def get_fh_odds(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("ApiOddsProvider is a stub; see class docstring")


def get_odds_provider(cfg: dict) -> OddsProvider:
    name = env("ODDS_PROVIDER", "none") or "none"
    if name == "none":
        return NullOddsProvider()
    if name == "manual_csv":
        return ManualCsvOddsProvider(path(cfg["live"]["manual_odds_path"]))
    if name == "api":
        return ApiOddsProvider()
    raise ValueError(f"unknown ODDS_PROVIDER {name!r}; choose none | manual_csv | api")


def merge_fh_odds(log_df: pd.DataFrame, odds: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fill EMPTY FH-odds fields in the log. Never overwrites.

    Pre-match odds are rejected if their ts_utc is at/after kick-off. Closing odds
    are accepted (they are by definition taken at kick-off).
    """
    out, notes = log_df.copy(), []
    if odds.empty or out.empty:
        return out, notes
    idx = {tuple(k): i for i, k in zip(out.index, out[KEY].astype(str).to_numpy())}
    for _, r in odds.iterrows():
        i = idx.get(tuple(str(r[k]) for k in KEY))
        if i is None:
            continue
        label = f"{r['div']} {r['date']} {r['home']} v {r['away']}"
        if not pd.isna(r["fh_o05_odds"]) and pd.isna(out.at[i, "fh_o05_odds"]):
            ts = pd.to_datetime(r["ts_utc"], utc=True)
            ko = pd.to_datetime(out.at[i, "kickoff_utc"], utc=True)
            if ts >= ko:
                notes.append(f"REJECTED pre-match FH odds for {label}: ts_utc {ts} is not before kick-off {ko}")
            else:
                out.loc[i, OPEN] = [r["fh_o05_odds"], r["fh_u05_odds"]]
                out.at[i, "fh_odds_source"] = r["source"]
                out.at[i, "fh_odds_ts_utc"] = ts.isoformat()
        if not pd.isna(r["fh_o05_close_odds"]) and pd.isna(out.at[i, "fh_o05_close_odds"]):
            out.loc[i, CLOSE] = [r["fh_o05_close_odds"], r["fh_u05_close_odds"]]
    for n in notes:
        log.warning(n)
    return out, notes
