import numpy as np
import pandas as pd

from fh.features.target import fh_over05


def test_target_cases():
    hthg = pd.Series([0, 1, 0, 2, 0, np.nan])
    htag = pd.Series([0, 0, 1, 1, 3, 0])
    out = fh_over05(hthg, htag)
    assert out.iloc[:5].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]
    assert np.isnan(out.iloc[5])


def test_target_ignores_full_time_goals():
    # 0-0 at half time, 3-2 at full time -> target 0
    assert fh_over05(pd.Series([0]), pd.Series([0])).iloc[0] == 0.0
