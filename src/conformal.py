"""Sections 2.5 and 3.7: conformal sets, triage rule (Table 3.5), tuned threshold."""
import math

import numpy as np
from scipy import stats

SETS = {(False, False): "{}", (True, False): "{0}", (False, True): "{1}", (True, True): "{0,1}"}
ACTION = {"{1}": "refer", "{0}": "routine", "{0,1}": "confirm", "{}": "confirm"}


def scores(p1, y):
    """Eq 2.1: s(x, y) = 1 - p_hat_y(x)."""
    p1, y = np.asarray(p1, float), np.asarray(y, int)
    return np.where(y == 1, 1 - p1, p1)


def quantile(s, alpha):
    """The ceil((n+1)(1-alpha))-th smallest score; +inf if that rank exceeds n."""
    s = np.sort(np.asarray(s, float))
    rank = math.ceil((len(s) + 1) * (1 - alpha) - 1e-12)
    return np.inf if rank > len(s) else s[rank - 1]


def class_thresholds(p1_cal, y_cal, alpha):
    """Mondrian thresholds (q0, q1), one per class, from the calibration set."""
    s, y = scores(p1_cal, y_cal), np.asarray(y_cal, int)
    return quantile(s[y == 0], alpha), quantile(s[y == 1], alpha)


def marginal_threshold(p1_cal, y_cal, alpha):
    q = quantile(scores(p1_cal, y_cal), alpha)
    return q, q


def prediction_sets(p1, q0, q1):
    """Eq 2.2: C(x) = {y : s(x, y) <= q_y}. Returns (contains 0, contains 1)."""
    p1 = np.asarray(p1, float)
    return p1 <= q0, (1 - p1) <= q1


def set_labels(in0, in1):
    return np.array([SETS[(bool(a), bool(b))] for a, b in zip(in0, in1)])


def triage(sets):
    return np.array([ACTION[s] for s in sets])


def tuned_threshold(p1_cal, y_cal, alpha):
    """Largest t with at least (1-alpha) of positive calibration women at p1 >= t."""
    pos = np.sort(np.asarray(p1_cal, float)[np.asarray(y_cal) == 1])[::-1]
    m = math.ceil((1 - alpha) * len(pos) - 1e-12)
    return pos[m - 1] if m >= 1 else np.inf


def threshold_sets(p1, t):
    """Tuned threshold layer: p1 >= t -> {1} (refer), else {0} (routine). No abstention."""
    refer = np.asarray(p1, float) >= t
    return ~refer, refer


def coverage_distribution(n, alpha, below=0.85):
    """Table 3.4: coverage of a fixed calibration set ~ Beta(l, n + 1 - l)."""
    l = math.ceil((n + 1) * (1 - alpha) - 1e-12)
    if l > n:
        return dict(l=l, finite=False, mean=np.nan, sd=np.nan, p_below=np.nan)
    b = stats.beta(l, n + 1 - l)
    return dict(l=l, finite=True, mean=b.mean(), sd=b.std(), p_below=b.cdf(below))
