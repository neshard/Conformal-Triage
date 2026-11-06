"""Section 3.5: cost-aware greedy selection (Algorithm 3.1, Eq 3.1), one-SE rule,
and the lasso comparator (group ranking by first entry along the L1 path)."""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.catalog import CATALOG, group_ids
from src.preprocess import GroupPreprocessor

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache" / "selection"
N_INNER = 5
FIT_COUNT = [0]          # LR fits done in this process (for the runtime log)


def _lr(C=1.0):
    return LogisticRegression(C=C, max_iter=5000)


def inner_folds(X, y, target, seed):
    """Inner stratified 5-fold CV on D_fit. Preprocessing is refitted on each inner
    training part; columns of a candidate set are sliced from it (see preprocess.py)."""
    folds = []
    for tr, va in StratifiedKFold(N_INNER, shuffle=True, random_state=seed).split(X, y):
        pp = GroupPreprocessor(target).fit(X.iloc[tr])
        folds.append((pp, pp.transform(X.iloc[tr]), y[tr], pp.transform(X.iloc[va]), y[va]))
    return folds


def fold_scores(folds, groups):
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for pp, Xtr, ytr, Xva, yva in folds:
            idx = pp.columns_of(groups)
            m = _lr().fit(Xtr[:, idx], ytr)
            out.append(average_precision_score(yva, m.predict_proba(Xva[:, idx])[:, 1]))
    FIT_COUNT[0] += len(folds)
    return out


def one_se_rule(A, SE):
    """j = argmax_k A(S_k); k* = min{k : A(S_k) >= A(S_j) - SE_j} (k is 1-based)."""
    A, SE = np.asarray(A), np.asarray(SE)
    j = int(np.argmax(A))
    return int(np.flatnonzero(A >= A[j] - SE[j])[0]) + 1


def _key(prefix, X, y, costs, k_max, seed):
    h = hashlib.sha256()
    h.update(json.dumps([list(map(int, X.index)), list(map(int, y)), sorted(costs.items()),
                         k_max, seed, list(X.columns)]).encode())
    h.update(CATALOG.read_bytes())
    return f"{prefix}_{h.hexdigest()[:16]}"


def greedy_select(X, y, target, costs, k_max=20, seed=0, n_jobs=1, cache_prefix=None):
    """Algorithm 3.1 on D_fit only. Returns the full path and the one-SE short form.
    Progress is cached after every step so an interrupted run resumes."""
    y = np.asarray(y, int)
    path_file = None
    state = dict(path=[], A=[], SE=[], fold_scores=[], A0=float(y.mean()))
    if cache_prefix:
        CACHE.mkdir(parents=True, exist_ok=True)
        path_file = CACHE / (_key(cache_prefix, X, y, costs, k_max, seed) + ".json")
        if path_file.exists():
            state = json.loads(path_file.read_text())
    folds = None
    candidates = group_ids(target)
    while len(state["path"]) < k_max:
        folds = folds or inner_folds(X, y, target, seed)
        S = state["path"]
        A_prev = state["A"][-1] if S else state["A0"]
        rest = [g for g in candidates if g not in S]          # sorted by group_id
        res = Parallel(n_jobs=n_jobs)(delayed(fold_scores)(folds, S + [g]) for g in rest)
        A = np.array([np.mean(r) for r in res])
        gain = (A - A_prev) / np.array([costs[g] for g in rest])
        b = int(np.argmax(gain))                              # first max = lowest group_id
        state["path"].append(rest[b])
        state["A"].append(float(A[b]))
        state["SE"].append(float(np.std(res[b], ddof=1) / np.sqrt(N_INNER)))
        state["fold_scores"].append([float(v) for v in res[b]])
        if path_file:
            path_file.write_text(json.dumps(state))
    k = one_se_rule(state["A"], state["SE"])
    return dict(state, k_star=k, selected=state["path"][:k])


def lasso_rank(X, y, target, n_C=100):
    """Rank groups by the smallest C at which any of their columns enters the
    L1-penalized logistic path (ties and never-entering groups: lowest group_id)."""
    pp = GroupPreprocessor(target).fit(X)
    Z = pp.transform(X)
    first = {g: np.inf for g in group_ids(target)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for C in np.logspace(-4, 2, n_C):
            m = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C, max_iter=5000).fit(Z, y)
            for g in set(pp.feature_groups_[np.abs(m.coef_[0]) > 0]):
                first[g] = min(first[g], C)
    return sorted(first, key=lambda g: (first[g], g))
