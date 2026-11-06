"""Sections 3.1, 3.5-3.8: outer CV, D_fit/D_cal split, the six arms of Table 3.6
and the no-verdict comparisons. Every out-of-fold prediction is stored so that
metrics, bootstraps and other alpha levels are recomputed without refitting."""
import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split

from src import conformal as cf
from src import data
from src.catalog import CATALOG, CATALOG_OF, group_costs, group_ids
from src.models import tune_lr, tune_xgb
from src.preprocess import feature_columns
from src.selection import greedy_select, lasso_rank

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "results" / "predictions"
SEL_DIR = ROOT / "results" / "selection"
FOLD_CACHE = ROOT / "results" / "cache" / "folds"
SUBDISTRICTS = ["gangga", "pemenang", "tanjung", "bayan", "kayangan"]


@dataclass(frozen=True)
class Setting:
    name: str
    label: str                        # anemia | ced | anemia_who | ced22 | ced24
    scheme: str = "main"
    alpha: float = 0.10
    cal_share: float = 0.25
    n_reps: int = 5
    n_folds: int = 5
    outer: str = "repeated"           # repeated | loso
    k_max: int = 20
    n_draws: int = 50
    extras: bool = False              # lasso, marginal conformal, XGBoost
    drop_ambiguous: bool = False
    max_hb_days: Optional[float] = None
    muac_clean: bool = False          # post-hoc: MUAC outside 15-45 cm -> missing as an anemia predictor
    provisional: bool = False

    @property
    def target(self):
        return CATALOG_OF[self.label]


def main_settings(provisional=False, **kw):
    return [Setting(f"main_{t}", t, extras=True, provisional=provisional, **kw)
            for t in ("anemia", "ced")]


# ---------------------------------------------------------------- data
def sample(setting):
    a = data.load(provisional=setting.provisional)
    a = a[a[f"in_{setting.target}"] == 1]
    a = a[a[setting.label].notna()]
    if setting.drop_ambiguous:
        a = a[a.ambiguous_dup_later == 0]
    if setting.max_hb_days is not None:      # unknown dates cannot be shown to exceed the limit
        a = a[~(a.hb_interview_days > setting.max_hb_days)]
    if setting.muac_clean:
        a = a.copy()
        a["c041"] = a["c041"].where(a["c041"].between(15, 45))
    return a


def outer_splits(setting, frame):
    y = frame[setting.label].astype(int).to_numpy()
    ids = frame.index.to_numpy()
    if setting.outer == "loso":
        sub = frame["c006"].to_numpy()
        for f, s in enumerate(SUBDISTRICTS):
            yield 0, f, ids[sub != s], ids[sub == s]
        return
    for rep in range(setting.n_reps):
        skf = StratifiedKFold(setting.n_folds, shuffle=True, random_state=rep)
        for f, (tr, te) in enumerate(skf.split(ids, y)):
            yield rep, f, ids[tr], ids[te]


# ---------------------------------------------------------------- one fold
def _fold_key(setting, frame, rep, fold):
    h = hashlib.sha256(json.dumps([asdict(setting), list(map(int, frame.index)),
                                   list(map(float, frame[setting.label]))]).encode())
    h.update(CATALOG.read_bytes())
    return FOLD_CACHE / setting.name / f"r{rep}_f{fold}_{h.hexdigest()[:12]}.pkl"


def run_fold(setting, frame, rep, fold, train_ids, test_ids):
    cache = _fold_key(setting, frame, rep, fold)
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    t0 = time.time()
    target, label = setting.target, setting.label
    seed = 1000 * rep + fold
    y = frame[label].astype(int)
    X = frame[feature_columns(frame, target)]
    fit_ids, cal_ids = train_test_split(train_ids, test_size=setting.cal_share,
                                        stratify=y.loc[train_ids], random_state=seed)
    Xfit, yfit = X.loc[fit_ids], y.loc[fit_ids].to_numpy()
    used_for_fitting = set(Xfit.index)                    # everything selection/tuning sees

    # ---- fitting branch: D_fit only
    groups = group_ids(target)
    sel = greedy_select(Xfit, yfit, target, group_costs(target, setting.scheme), setting.k_max,
                        seed, cache_prefix=f"{setting.name}_r{rep}_f{fold}")
    k = sel["k_star"]
    rng = np.random.default_rng([rep, fold, 2026])
    itemsets = {("full", 0): groups, ("short", 0): sel["selected"]}
    for d in range(setting.n_draws):
        itemsets[("random", d)] = sorted(rng.choice(groups, k, replace=False).tolist())
    lasso_order = None
    if setting.extras:
        lasso_order = lasso_rank(Xfit, yfit, target)
        itemsets[("lasso", 0)] = sorted(lasso_order[:k])
    models, tuned = {}, {}
    for key, gs in itemsets.items():
        models[key], tuned[key] = tune_lr(Xfit, yfit, target, gs, seed)
    if setting.extras:
        for name in ("full", "short"):
            models[("xgb_" + name, 0)], tuned[("xgb_" + name, 0)] = tune_xgb(
                Xfit, yfit, target, itemsets[(name, 0)], seed)

    # ---- calibration branch: only after every model is fixed
    assert not used_for_fitting & set(cal_ids), "D_cal leaked into selection or tuning"
    assert not used_for_fitting & set(test_ids), "test fold leaked into selection or tuning"
    rows = []
    for (itemset, d), m in models.items():
        for part, ids in (("cal", cal_ids), ("test", test_ids)):
            p = m.predict_proba(X.loc[ids])[:, 1]
            rows.append(pd.DataFrame(dict(row_id=ids, rep=rep, fold=fold, itemset=itemset,
                                          draw=d, part=part, p_hat=p,
                                          label=y.loc[ids].to_numpy())))
    out = dict(
        scores=pd.concat(rows, ignore_index=True),
        selection=dict(rep=rep, fold=fold, k_star=k, selected=sel["selected"], path=sel["path"],
                       A=sel["A"], SE=sel["SE"], A0=sel["A0"], fold_scores=sel["fold_scores"],
                       lasso_order=lasso_order, n_fit=len(fit_ids), n_cal=len(cal_ids),
                       n_cal_pos=int(y.loc[cal_ids].sum()), n_test=len(test_ids),
                       draws=[itemsets[("random", d)] for d in range(setting.n_draws)],
                       tuned={f"{a}:{b}": v for (a, b), v in tuned.items()
                              if a != "random"},
                       seconds=time.time() - t0))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out


# ---------------------------------------------------------------- arms
ARMS = {  # arm -> (itemset, decision layer)
    "A1": ("full", "tuned"), "A2": ("full", "conformal"),
    "A3": ("short", "tuned"), "A4": ("short", "conformal"),
    "A5": ("random", "tuned"), "A6": ("random", "conformal"),
    "lasso_tuned": ("lasso", "tuned"), "lasso_conformal": ("lasso", "conformal"),
    "A4_marginal": ("short", "marginal"),
    "xgb_full_tuned": ("xgb_full", "tuned"), "xgb_full_conformal": ("xgb_full", "conformal"),
    "xgb_short_tuned": ("xgb_short", "tuned"), "xgb_short_conformal": ("xgb_short", "conformal"),
}


def derive_arms(scores, alpha):
    """Prediction sets and triage actions for every arm, from stored scores."""
    out = []
    for (itemset, rep, fold, d), g in scores.groupby(["itemset", "rep", "fold", "draw"], sort=False):
        cal, test = g[g.part == "cal"], g[g.part == "test"]
        for arm, (iset, layer) in ARMS.items():
            if iset != itemset:
                continue
            if layer == "tuned":
                in0, in1 = cf.threshold_sets(test.p_hat, cf.tuned_threshold(cal.p_hat, cal.label, alpha))
            else:
                thr = cf.class_thresholds if layer == "conformal" else cf.marginal_threshold
                in0, in1 = cf.prediction_sets(test.p_hat, *thr(cal.p_hat, cal.label, alpha))
            sets = cf.set_labels(in0, in1)
            out.append(pd.DataFrame(dict(row_id=test.row_id.to_numpy(), rep=rep, fold=fold, arm=arm,
                                         draw=d, p_hat=test.p_hat.to_numpy(), set=sets,
                                         action=cf.triage(sets), label=test.label.to_numpy())))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- runner
def run(setting, n_jobs=-1, log=print):
    frame = sample(setting)
    jobs = list(outer_splits(setting, frame))
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run_fold)(setting, frame, rep, f, tr, te) for rep, f, tr, te in jobs)
    scores = pd.concat([r["scores"] for r in res], ignore_index=True)
    arms = derive_arms(scores, setting.alpha)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    SEL_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(PRED_DIR / f"{setting.name}_scores.parquet", index=False)
    arms.to_parquet(PRED_DIR / f"{setting.name}.parquet", index=False)
    meta = dict(setting=asdict(setting), n=len(frame), n_pos=int(frame[setting.label].sum()),
                folds=[r["selection"] for r in res], wall_seconds=time.time() - t0)
    (SEL_DIR / f"{setting.name}.json").write_text(json.dumps(meta, default=_json))
    log(f"{setting.name}: n={meta['n']} pos={meta['n_pos']} folds={len(jobs)} "
        f"wall={meta['wall_seconds']:.0f}s")
    return arms, meta


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_results(name):
    arms = pd.read_parquet(PRED_DIR / f"{name}.parquet")
    meta = json.loads((SEL_DIR / f"{name}.json").read_text())
    return arms, meta


def load_scores(name):
    return pd.read_parquet(PRED_DIR / f"{name}_scores.parquet")


def n_jobs_default():
    return max(1, (os.cpu_count() or 2) - 1)
