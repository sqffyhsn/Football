import numpy as np
import pandas as pd
import pytest

from fh.data.clean import TIDY_COLUMNS, build_dataset, normalise, read_raw_csv

# 2016/17 style: 2-digit years, no Time, BbAv odds, latin-1, trailing empty cols/rows.
OLD = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,B365H,BbAvH,BbAvD,BbAvA,BbAv>2.5,BbAv<2.5,,,\n"
    "E0,13/08/16,Burnley,Swansea,0,1,A,0,0,D,2.4,2.35,3.2,3.1,2.1,1.75,,,\n"
    "E0,13/08/16,Crystal Palace,West Brom,0,1,A,0,0,D,2.0,1.95,3.3,4.2,2.3,1.6,,,\n"
    "E0,14/08/16,M\xe1laga,Arsenal,3,4,A,1,1,D,,,,,,,,,\n"
    ",,,,,,,,,,,,,,,,,,\n"
).encode("latin-1")

# 2019/20+ style: 4-digit years, Time, Avg + closing odds, UTF-8 BOM.
NEW = (
    "﻿Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5,"
    "AvgCH,AvgCD,AvgCA,AvgC>2.5,AvgC<2.5\n"
    "E0,09/08/2019,20:00,Liverpool,Norwich,4,1,H,4,0,H,1.14,9.5,20,1.4,3.0,1.14,9.0,21,1.38,3.1\n"
    "E0,10/08/2019,15:00,Burnley,Southampton,3,0,H,0,0,D,2.4,3.3,3.1,2.0,1.8,2.3,3.3,3.2,2.1,1.75\n"
    "E0,10/08/2019,15:00,Burnley,Southampton,3,0,H,0,0,D,2.4,3.3,3.1,2.0,1.8,2.3,3.3,3.2,2.1,1.75\n"
    "E0,11/08/2019,14:00,Leicester,Wolves,0,0,D,,,,1.9,3.5,4.2,2.0,1.8,,,,,\n"
    "E0,12/08/2019,20:00,Bad,Data,1,0,H,2,0,H,0.9,3.5,4.2,2.0,1.8,,,,,\n"
).encode("utf-8")


def test_old_and_new_formats_normalise_to_same_schema():
    rep_old, rep_new = {}, {}
    old = normalise(read_raw_csv(OLD), season="1617", div="E0", report=rep_old)
    new = normalise(read_raw_csv(NEW), season="1920", div="E0", report=rep_new)
    assert list(old.columns) == list(new.columns) == TIDY_COLUMNS
    assert old["date"].tolist() == [pd.Timestamp("2016-08-13")] * 2 + [pd.Timestamp("2016-08-14")]
    assert old["time"].isna().all()
    assert new["time"].tolist()[:2] == ["20:00", "15:00"]
    # latin-1 team names decoded correctly
    assert "Málaga" in old["home"].tolist()
    # BbAv -> odds_h; Avg -> odds_h
    assert old.loc[0, "odds_h"] == 2.35 and new.loc[0, "odds_h"] == 1.14
    assert new.loc[0, "close_o25"] == 1.38
    assert old["close_o25"].isna().all()
    assert rep_old["dropped_no_date_or_team"] == 0  # blank trailing rows are skipped on read


def test_invalid_rows_are_dropped_and_counted():
    rep = {}
    new = normalise(read_raw_csv(NEW), season="1920", div="E0", report=rep)
    assert rep["dropped_duplicates"] == 1
    assert rep["dropped_missing_scores"] == 1
    assert rep["dropped_ht_gt_ft"] == 1  # "Bad v Data": HT 2 > FT 1
    assert len(new) == 2


def test_odds_le_one_become_nan():
    raw = read_raw_csv(
        b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG,AvgH,AvgD,AvgA\n"
        b"E0,01/09/2020,A,B,1,0,1,0,1.0,3.0,0.5\n")
    df = normalise(raw, season="2021", div="E0")
    assert np.isnan(df.loc[0, "odds_h"]) and np.isnan(df.loc[0, "odds_a"])
    assert df.loc[0, "odds_d"] == 3.0


def test_b365_fallback_when_average_missing():
    raw = read_raw_csv(
        b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG,B365H,B365D,B365A,AvgH\n"
        b"E0,01/09/2020,A,B,1,0,1,0,2.0,3.0,4.0,\n")
    df = normalise(raw, season="2021", div="E0")
    assert df.loc[0, "odds_h"] == 2.0


def test_ragged_rows_are_padded_or_trimmed():
    raw = read_raw_csv(
        b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG\n"
        b"E0,01/09/2020,A,B,1,0,1,0,extra,fields\n"
        b"E0,02/09/2020,C,D,2,2,1\n")
    df = normalise(raw, season="2021", div="E0")
    assert len(df) == 1 and df.loc[0, "home"] == "A"


def test_alias_columns():
    raw = read_raw_csv(b"Div,Date,Home,Away,HG,AG,HTHG,HTAG\nSC0,01/09/2020,A,B,1,0,0,0\n")
    df = normalise(raw, season="2021", div="SC0")
    assert df.loc[0, "FTHG"] == 1 and df.loc[0, "home"] == "A"


def test_build_dataset_sorted_and_reported(tmp_path):
    (tmp_path / "a.csv").write_bytes(OLD)
    (tmp_path / "b.csv").write_bytes(NEW)
    df, rep = build_dataset([("1617", "E0", tmp_path / "a.csv"), ("1920", "E0", tmp_path / "b.csv")])
    assert df["date"].is_monotonic_increasing
    assert rep["rows_clean"] == len(df) == 5
    assert set(rep["odds_coverage"]) >= {"odds_h", "close_o25"}
