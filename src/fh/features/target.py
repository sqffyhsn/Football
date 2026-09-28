import numpy as np
import pandas as pd


def fh_over05(hthg: pd.Series, htag: pd.Series) -> pd.Series:
    """1 if at least one first-half goal, 0 if none, NaN if the half-time score is unknown."""
    total = pd.to_numeric(hthg, errors="coerce") + pd.to_numeric(htag, errors="coerce")
    return pd.Series(np.where(total.isna(), np.nan, (total >= 1).astype(float)), index=total.index)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(target=fh_over05(df["HTHG"], df["HTAG"]))
