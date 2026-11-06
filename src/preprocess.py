"""Section 3.4: fold-fitted preprocessing over question groups.

Every operation is per column (plausibility, median/mode imputation, one-hot
categories, standardization), so fitting on all groups and slicing the output
columns of a subset S gives exactly the matrix that fitting on S alone would.
selection.py relies on this to fit once per inner fold; tests check it.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.catalog import CATALOG_OF, feature_name, included

PLAUSIBLE = {"c014": (15, 50), "c043": (30, 120), "c045": (130, 190), "c040": (1, 42)}
HB, MUAC = "c047", "c041"
LABELS = {"anemia", "anemia_who", "ced", "ced22", "ced24", "hb", "hemoglobin"}


class LeakageError(AssertionError):
    pass


def check_no_leakage(columns, target):
    cols = {str(c).lower() for c in columns}
    bad = {c for c in cols if c == HB or c.startswith(HB + "__") or c in LABELS or "hb" in c}
    if CATALOG_OF[target] == "ced":
        bad |= {c for c in cols if c == MUAC or c.startswith(MUAC + "__")}
    if bad:
        raise LeakageError(f"forbidden columns in {target} matrix: {sorted(bad)}")


def feature_spec(target):
    """List of (processed column, group_id, kind) for every predictor of a target."""
    spec = []
    for _, r in included(target).iterrows():
        name = feature_name(r.col_index)
        if r.encoding == "multibinary":
            spec.append((name + "__", r.group_id, "multibinary"))   # prefix, expanded on fit
        else:
            spec.append((name, r.group_id, r.encoding))
    return spec


def feature_columns(frame, target):
    """Processed columns that form the predictor matrix for a target."""
    cols = []
    for name, _, kind in feature_spec(target):
        cols += [c for c in frame.columns if c.startswith(name)] if kind == "multibinary" else [name]
    return cols


def _mode(s):
    vc = s.dropna().value_counts()
    if vc.empty:
        return np.nan
    top = vc[vc == vc.max()].index
    return sorted(top, key=str)[0]


class GroupPreprocessor(BaseEstimator, TransformerMixin):
    """Fit on the training part of a fold only. `groups` restricts the output to those
    question groups (None = all groups of the target)."""

    def __init__(self, target="anemia", groups=None, scale=True):
        self.target = target
        self.groups = groups
        self.scale = scale

    def fit(self, X, y=None):
        check_no_leakage(X.columns, self.target)
        keep = None if self.groups is None else set(self.groups)
        unknown = (keep or set()) - {g for _, g, _ in feature_spec(self.target)}
        if unknown:
            raise ValueError(f"groups not in the {self.target} catalog: {sorted(unknown)}")
        self.plan_ = []                       # (column, group, kind, fill, categories)
        for name, g, kind in feature_spec(self.target):
            if keep is not None and g not in keep:
                continue
            cols = [c for c in X.columns if c.startswith(name)] if kind == "multibinary" else [name]
            for c in cols:
                s = self._plausible(X[c], c)
                if kind == "numeric":
                    fill = s.median()
                    fill = 0.0 if np.isnan(fill) else fill
                    self.plan_.append((c, g, kind, fill, None))
                else:
                    fill = _mode(s)
                    if kind == "onehot":
                        cats = sorted(s.fillna(fill).dropna().unique(), key=str)
                        self.plan_.append((c, g, kind, fill, cats))
                    else:
                        self.plan_.append((c, g, kind, 0.0 if pd.isna(fill) else fill, None))
        raw = self._encode(X)
        self.feature_groups_ = np.array(self._groups_out)
        self.feature_names_ = list(self._names_out)
        if self.scale:
            self.mean_ = raw.mean(axis=0)
            sd = raw.std(axis=0)
            self.scale_ = np.where(sd > 0, sd, 1.0)
        return self

    @staticmethod
    def _plausible(s, c):
        if c in PLAUSIBLE:
            lo, hi = PLAUSIBLE[c]
            s = s.where(s.between(lo, hi))
        return s

    def _encode(self, X):
        blocks, self._groups_out, self._names_out = [], [], []
        for c, g, kind, fill, cats in self.plan_:
            s = self._plausible(X[c], c).fillna(fill)
            if kind == "onehot":
                vals = s.to_numpy(dtype=object)
                blocks.append(np.column_stack([(vals == k).astype(float) for k in cats]))
                self._groups_out += [g] * len(cats)
                self._names_out += [f"{c}={k}" for k in cats]
            else:
                blocks.append(s.to_numpy(dtype=float)[:, None])
                self._groups_out.append(g)
                self._names_out.append(c)
        return np.hstack(blocks) if blocks else np.empty((len(X), 0))

    def transform(self, X):
        check_no_leakage(X.columns, self.target)
        out = self._encode(X)
        if self.scale:
            out = (out - self.mean_) / self.scale_
        return out

    def columns_of(self, groups):
        return np.flatnonzero(np.isin(self.feature_groups_, list(groups)))
