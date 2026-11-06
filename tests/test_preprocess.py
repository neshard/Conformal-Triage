import numpy as np
import pandas as pd
import pytest

from src.catalog import group_ids
from src.data import PROVISIONAL, PROCESSED, load
from src.preprocess import GroupPreprocessor, LeakageError, check_no_leakage, feature_columns

DATA = PROCESSED if PROCESSED.exists() else PROVISIONAL
needs_data = pytest.mark.skipif(not DATA.exists(), reason="no analysis table built")


def _frame(target):
    a = load(provisional=DATA == PROVISIONAL)
    a = a[a[f"in_{'anemia' if target == 'anemia' else 'ced'}"] == 1]
    return a[feature_columns(a, target)]


def test_leakage_hemoglobin_and_muac():
    with pytest.raises(LeakageError):
        check_no_leakage(["c014", "c047"], "anemia")
    with pytest.raises(LeakageError):
        check_no_leakage(["c014", "anemia"], "anemia")
    with pytest.raises(LeakageError):
        check_no_leakage(["c014", "c041"], "ced")
    check_no_leakage(["c014", "c041"], "anemia")


@needs_data
def test_fit_raises_when_muac_given_to_ced():
    X = _frame("anemia")                       # contains c041
    with pytest.raises(LeakageError):
        GroupPreprocessor("ced").fit(X)


@needs_data
@pytest.mark.parametrize("target", ["anemia", "ced"])
def test_slice_equals_subset_fit(target):
    X = _frame(target)
    tr, te = X.iloc[: len(X) // 2], X.iloc[len(X) // 2:]
    full = GroupPreprocessor(target).fit(tr)
    rng = np.random.default_rng(0)
    subset = sorted(rng.choice(group_ids(target), 7, replace=False))
    sub = GroupPreprocessor(target, groups=subset).fit(tr)
    idx = full.columns_of(subset)
    np.testing.assert_allclose(full.transform(te)[:, idx], sub.transform(te))
    assert set(sub.feature_groups_) == set(subset)


@needs_data
def test_plausibility_and_no_nan():
    X = _frame("anemia").copy()
    X.iloc[0, X.columns.get_loc("c014")] = 3          # implausible age
    pp = GroupPreprocessor("anemia", groups=["G003"], scale=False).fit(X)
    out = pp.transform(X)
    assert out[0, 0] == pd.to_numeric(X.c014.where(X.c014.between(15, 50))).median()
    assert not np.isnan(GroupPreprocessor("anemia").fit(X).transform(X)).any()
