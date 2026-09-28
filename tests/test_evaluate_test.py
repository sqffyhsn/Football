import importlib.util

import pytest

from fh.config import ROOT
from fh.evaluation.report import TEST_INTERPRETATION, test_interpretation as interpret

spec = importlib.util.spec_from_file_location("evaluate_test", ROOT / "scripts" / "evaluate_test.py")
evaluate_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_test)


def test_refuses_second_run(tmp_path):
    evaluate_test.check_not_evaluated(tmp_path)  # no marker -> fine
    evaluate_test.marker_path(tmp_path).write_text("{}")
    with pytest.raises(SystemExit):
        evaluate_test.check_not_evaluated(tmp_path)


@pytest.mark.parametrize("lo,hi,key", [(-0.002, -0.0001, "below"), (-0.001, 0.001, "spans"), (0.0001, 0.002, "above")])
def test_interpretation_is_chosen_by_ci_only(lo, hi, key):
    assert interpret({"delta": (lo + hi) / 2, "ci_low": lo, "ci_high": hi}) == TEST_INTERPRETATION[key]
