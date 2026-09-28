"""Synthetic match data in the tidy schema — for tests and pipeline smoke runs ONLY.

Never used for reported results. Goals are Poisson from latent team strengths;
odds are priced from the true intensities plus noise and margin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from fh.data.clean import TIDY_COLUMNS, make_match_id


def make_synthetic(n_leagues: int = 2, n_teams: int = 10, seasons=("1617", "1718", "1819"),
                   seed: int = 0, games_per_day: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for li in range(n_leagues):
        div = f"L{li}"
        teams = [f"{div}_T{i}" for i in range(n_teams)]
        att = dict(zip(teams, rng.normal(0, 0.25, n_teams)))
        dfn = dict(zip(teams, rng.normal(0, 0.25, n_teams)))
        for si, season in enumerate(seasons):
            start = pd.Timestamp(f"20{season[:2]}-08-01") + pd.Timedelta(days=li)
            fixtures = [(h, a) for h in teams for a in teams if h != a]
            rng.shuffle(fixtures)
            day, used = start, set()
            pending = list(fixtures)
            while pending:
                today, rest = [], []
                for h, a in pending:
                    if h not in used and a not in used and len(today) < games_per_day:
                        today.append((h, a)); used |= {h, a}
                    else:
                        rest.append((h, a))
                for k, (h, a) in enumerate(today):
                    lh = np.exp(0.35 + att[h] - dfn[a])
                    la = np.exp(0.10 + att[a] - dfn[h])
                    hthg, htag = rng.poisson(0.45 * lh), rng.poisson(0.45 * la)
                    fthg, ftag = hthg + rng.poisson(0.55 * lh), htag + rng.poisson(0.55 * la)
                    lam = (lh + la) * np.exp(rng.normal(0, 0.08))
                    p_over = poisson.sf(2, lam)
                    ph = 0.45 + 0.3 * np.tanh(att[h] - att[a]); pa = 0.8 - ph; pd_ = 0.2
                    m = 1.06
                    rows.append(dict(div=div, season=season, date=day, time=f"{13 + k}:00",
                                     home=h, away=a, HTHG=hthg, HTAG=htag, FTHG=fthg, FTAG=ftag,
                                     odds_h=1 / (ph * m), odds_d=1 / (pd_ * m), odds_a=1 / (pa * m),
                                     odds_o25=1 / (p_over * m), odds_u25=1 / ((1 - p_over) * m),
                                     close_h=1 / (ph * m), close_d=1 / (pd_ * m), close_a=1 / (pa * m),
                                     close_o25=1 / (p_over * m), close_u25=1 / ((1 - p_over) * m)))
                pending, used = rest, set()
                day += pd.Timedelta(days=int(rng.integers(2, 5)))
    df = pd.DataFrame(rows)
    df["match_id"] = make_match_id(df["div"], df["date"], df["home"], df["away"])
    return df[TIDY_COLUMNS]
