import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate import Prepared, clopper_pearson, three_way


def _arms(seed, n=120, reps=2, ties=False):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    rows = []
    for r in range(reps):
        p = np.clip(0.3 * y + rng.random(n) * 0.7, 0, 1)
        if ties:
            p = np.round(p, 1)
        sets = rng.choice(["{0}", "{1}", "{0,1}", "{}"], n)
        rows.append(pd.DataFrame(dict(row_id=np.arange(n), rep=r, fold=np.arange(n) % 5,
                                      arm="A1", draw=0, p_hat=p, set=sets, label=y)))
    return pd.concat(rows)


@pytest.mark.parametrize("ties", [False, True])
def test_weighted_metrics_equal_sklearn_on_resample(ties):
    arms = _arms(0, ties=ties)
    prep = Prepared(arms)
    rng = np.random.default_rng(1)
    w = rng.multinomial(prep.n, np.full(prep.n, 1 / prep.n))
    m = prep.metrics(w)
    for r in range(2):
        g = arms[arms.rep == r].sort_values("row_id")
        idx = np.repeat(np.arange(prep.n), w)                 # explicit resample
        yy, pp = g.label.to_numpy()[idx], g.p_hat.to_numpy()[idx]
        assert np.isclose(m["auprc"][0, r], average_precision_score(yy, pp))
        assert np.isclose(m["auroc"][0, r], roc_auc_score(yy, pp))
        assert np.isclose(m["brier"][0, r], np.mean((pp - yy) ** 2))
        s = g.set.to_numpy()[idx]
        assert np.isclose(m["cov1"][0, r], np.isin(s[yy == 1], ["{1}", "{0,1}"]).mean())
        assert np.isclose(m["routine"][0, r], (s == "{0}").mean())
        assert np.isclose(m["miss"][0, r], (s[yy == 1] == "{0}").mean())


def test_clopper_pearson_integer_matches_scipy():
    from scipy.stats import binomtest
    ci = binomtest(80, 85).proportion_ci(0.975, method="exact")
    assert np.allclose(clopper_pearson(80, 85, 0.975), (ci.low, ci.high))
    lo, hi = clopper_pearson(80.4, 85, 0.975)                  # averaged count is allowed
    assert 0 < lo < hi < 1


def test_three_way():
    assert three_way(-0.04, 0.01, -0.05) == "supported"
    assert three_way(-0.09, -0.06, -0.05) == "not supported"
    assert three_way(-0.07, 0.01, -0.05) == "inconclusive"
