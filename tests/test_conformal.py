import math

import numpy as np
import pytest

from src.conformal import (class_thresholds, coverage_distribution, marginal_threshold,
                           prediction_sets, quantile, set_labels, threshold_sets, triage,
                           tuned_threshold)


def brute_force_sets(p1_cal, y_cal, p1_test, alpha):
    """Direct reading of Sec 2.5: sort each class's scores and take rank ceil((n+1)(1-a))."""
    out = []
    for p in p1_test:
        members = set()
        for y in (0, 1):
            cal = [1 - q if y == 1 else q for q, t in zip(p1_cal, y_cal) if t == y]
            rank = math.ceil((len(cal) + 1) * (1 - alpha))
            thr = float("inf") if rank > len(cal) else sorted(cal)[rank - 1]
            s = 1 - p if y == 1 else p
            if s <= thr:
                members.add(y)
        out.append(members)
    return out


@pytest.mark.parametrize("seed", range(20))
def test_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 120))
    y = rng.integers(0, 2, n)
    p = np.clip(rng.beta(2, 5, n) + 0.3 * y, 0, 1)
    pt = rng.random(50)
    alpha = float(rng.choice([0.05, 0.1, 0.2, 0.3]))
    q0, q1 = class_thresholds(p, y, alpha)
    in0, in1 = prediction_sets(pt, q0, q1)
    got = [{k for k, v in ((0, a), (1, b)) if v} for a, b in zip(in0, in1)]
    assert got == brute_force_sets(p, y, pt, alpha)


def test_infinite_threshold_anemia_alpha_005():
    y = np.r_[np.ones(17), np.zeros(61)].astype(int)
    p = np.linspace(0.05, 0.95, 78)
    q0, q1 = class_thresholds(p, y, 0.05)
    assert np.isinf(q1) and np.isfinite(q0)
    in0, in1 = prediction_sets(np.linspace(0, 1, 11), q0, q1)
    assert in1.all()                                # every set contains the positive label
    assert "routine" not in triage(set_labels(in0, in1))


def test_quantile_rank():
    s = np.arange(1, 18) / 100                       # 17 scores
    assert quantile(s, 0.10) == s[16]                # rank 17 = largest
    assert np.isinf(quantile(s, 0.05))


@pytest.mark.parametrize("n,alpha,l,mean,sd,p", [
    (17, 0.10, 17, 0.944, 0.053, 0.063),
    (22, 0.10, 21, 0.913, 0.058, 0.137),
    (22, 0.05, 22, 0.957, 0.042, 0.028),
])
def test_table_3_4(n, alpha, l, mean, sd, p):
    d = coverage_distribution(n, alpha)
    assert d["l"] == l
    assert round(d["mean"], 3) == mean and round(d["sd"], 3) == sd and round(d["p_below"], 3) == p


def test_table_3_4_infinite():
    assert not coverage_distribution(17, 0.05)["finite"]


def test_tuned_threshold_definition():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    t = tuned_threshold(p, y, 0.10)
    pos = p[y == 1]
    assert (pos >= t).mean() >= 0.90
    assert not any((pos >= u).mean() >= 0.90 for u in pos[pos > t])   # t is the largest such value
    in0, in1 = threshold_sets(p, t)
    assert (in0 ^ in1).all()                          # no abstention


def test_marginal_single_threshold():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.9])
    q0, q1 = marginal_threshold(p, y, 0.2)
    assert q0 == q1
