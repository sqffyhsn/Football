"""Normalise raw football-data.co.uk CSVs into one tidy match table.

Handles the format drift across seasons/leagues:
- encodings (utf-8 with/without BOM, latin-1) and rows with more/fewer fields than the header
- trailing empty rows/columns
- dates as dd/mm/yy or dd/mm/yyyy; `Time` column only present from 2019/20
- odds column renames (BbAv* up to 2018/19 -> Avg* from 2019/20; B365 fallback)
- older/alternative result column names (HG/AG, Home/Away)
"""
from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COLUMN_ALIASES = {
    "HG": "FTHG", "AG": "FTAG", "Res": "FTR",
    "Home": "HomeTeam", "Away": "AwayTeam", "HT": "HomeTeam", "AT": "AwayTeam",
}

# Canonical odds column -> candidate source columns, in order of preference.
# Pre-closing (available before kick-off, also in the fixtures file):
PRE_ODDS = {
    "odds_h":   ["AvgH", "BbAvH", "B365H"],
    "odds_d":   ["AvgD", "BbAvD", "B365D"],
    "odds_a":   ["AvgA", "BbAvA", "B365A"],
    "odds_o25": ["Avg>2.5", "BbAv>2.5", "B365>2.5"],
    "odds_u25": ["Avg<2.5", "BbAv<2.5", "B365<2.5"],
}
# Closing odds: NEVER used as model features (not known at prediction time).
# Only used after the fact for closing-line-value analysis.
CLOSE_ODDS = {
    "close_h":   ["AvgCH", "B365CH"],
    "close_d":   ["AvgCD", "B365CD"],
    "close_a":   ["AvgCA", "B365CA"],
    "close_o25": ["AvgC>2.5", "B365C>2.5"],
    "close_u25": ["AvgC<2.5", "B365C<2.5"],
}

RESULT_COLS = ["HTHG", "HTAG", "FTHG", "FTAG"]
TIDY_COLUMNS = (["match_id", "div", "season", "date", "time", "home", "away"]
                + RESULT_COLS + list(PRE_ODDS) + list(CLOSE_ODDS))


def read_raw_csv(source: str | Path | bytes) -> pd.DataFrame:
    """Robustly read a football-data CSV into all-string columns."""
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return pd.DataFrame()
    header = [h.strip() for h in rows[0]]
    # Drop unnamed trailing header cells; pad/trim data rows to header width.
    while header and header[-1] == "":
        header.pop()
    width = len(header)
    body = [(r + [""] * width)[:width] for r in rows[1:]]
    df = pd.DataFrame(body, columns=header, dtype=str)
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    df = df.loc[:, ~df.columns.duplicated()]
    return df.rename(columns={k: v for k, v in COLUMN_ALIASES.items()
                              if k in df.columns and v not in df.columns})


def parse_dates(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    long = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return long.fillna(short)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col].astype(str).str.strip().replace("", np.nan), errors="coerce")


def _first_available(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for col in candidates:
        out = out.fillna(_num(df, col))
    return out.where(out > 1.0)  # odds <= 1 are invalid


def make_match_id(div: pd.Series, date: pd.Series, home: pd.Series, away: pd.Series) -> pd.Series:
    slug = lambda s: s.str.replace(r"[^A-Za-z0-9]+", "", regex=True)
    return div + "_" + date.dt.strftime("%Y%m%d") + "_" + slug(home) + "_" + slug(away)


def normalise(df: pd.DataFrame, season: str | None, div: str | None = None,
              require_results: bool = True, report: dict | None = None) -> pd.DataFrame:
    """Map one raw file to the tidy schema. `season` is None for the fixtures file."""
    report = report if report is not None else {}
    if df.empty:
        return pd.DataFrame(columns=TIDY_COLUMNS)
    out = pd.DataFrame(index=df.index)
    out["div"] = df["Div"].str.strip() if "Div" in df.columns else div
    if div is not None:
        out["div"] = out["div"].fillna(div).replace("", div)
    out["season"] = season
    out["date"] = parse_dates(df["Date"]) if "Date" in df.columns else pd.NaT
    time_col = df["Time"].astype(str).str.strip() if "Time" in df.columns else ""
    out["time"] = pd.Series(time_col, index=df.index).where(
        lambda t: t.str.match(r"^\d{1,2}:\d{2}$", na=False), None)
    out["home"] = df.get("HomeTeam", pd.Series("", index=df.index)).astype(str).str.strip()
    out["away"] = df.get("AwayTeam", pd.Series("", index=df.index)).astype(str).str.strip()
    for col in RESULT_COLS:
        out[col] = _num(df, col)
    for canon, cands in {**PRE_ODDS, **CLOSE_ODDS}.items():
        out[canon] = _first_available(df, cands)

    n0 = len(out)
    valid_id = out["date"].notna() & (out["home"] != "") & (out["away"] != "") & (out["home"] != "nan")
    out = out[valid_id]
    report["dropped_no_date_or_team"] = report.get("dropped_no_date_or_team", 0) + n0 - len(out)

    if require_results:
        n1 = len(out)
        out = out.dropna(subset=RESULT_COLS)
        report["dropped_missing_scores"] = report.get("dropped_missing_scores", 0) + n1 - len(out)
        n2 = len(out)
        ok = (out["HTHG"] <= out["FTHG"]) & (out["HTAG"] <= out["FTAG"]) & (out[RESULT_COLS] >= 0).all(axis=1)
        out = out[ok]
        report["dropped_ht_gt_ft"] = report.get("dropped_ht_gt_ft", 0) + n2 - len(out)
        for col in RESULT_COLS:
            out[col] = out[col].astype(int)

    out["match_id"] = make_match_id(out["div"], out["date"], out["home"], out["away"])
    n3 = len(out)
    out = out.drop_duplicates(subset=["div", "date", "home", "away"], keep="first")
    report["dropped_duplicates"] = report.get("dropped_duplicates", 0) + n3 - len(out)
    return out[TIDY_COLUMNS].reset_index(drop=True)


def sort_matches(df: pd.DataFrame) -> pd.DataFrame:
    return (df.assign(_t=df["time"].fillna("99:99"))
              .sort_values(["date", "_t", "div", "home"], kind="mergesort")
              .drop(columns="_t").reset_index(drop=True))


def build_dataset(files: list[tuple[str, str, Path]]) -> tuple[pd.DataFrame, dict]:
    """files: list of (season, div, path). Returns tidy dataset and a cleaning report."""
    frames, report = [], {"files": 0, "rows_raw": 0}
    for season, div, p in files:
        raw = read_raw_csv(p)
        report["files"] += 1
        report["rows_raw"] += len(raw)
        frames.append(normalise(raw, season=season, div=div, report=report))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TIDY_COLUMNS)
    n = len(df)
    df = df.drop_duplicates(subset=["div", "date", "home", "away"], keep="first")
    report["dropped_duplicates"] = report.get("dropped_duplicates", 0) + n - len(df)
    df = sort_matches(df)
    report["rows_clean"] = len(df)
    report["odds_coverage"] = {c: float(df[c].notna().mean()) if len(df) else 0.0
                               for c in list(PRE_ODDS) + list(CLOSE_ODDS)}
    validate(df, require_results=True)
    return df, report


def validate(df: pd.DataFrame, require_results: bool = True) -> None:
    key = ["div", "date", "home", "away"]
    assert not df.duplicated(subset=key).any(), "duplicate matches"
    assert df["date"].notna().all(), "missing dates"
    if require_results and len(df):
        assert (df["HTHG"] <= df["FTHG"]).all() and (df["HTAG"] <= df["FTAG"]).all(), "HT > FT"
    odds = df[list(PRE_ODDS) + list(CLOSE_ODDS)]
    assert ((odds > 1) | odds.isna()).all().all(), "invalid odds"
    # a team cannot play twice on one day in one league
    long = pd.concat([df[["div", "date", "home"]].rename(columns={"home": "team"}),
                      df[["div", "date", "away"]].rename(columns={"away": "team"})])
    assert not long.duplicated().any(), "team plays twice on the same date"
