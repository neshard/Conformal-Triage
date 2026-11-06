"""Section 3.9 and Table 1.1: metrics (Table 3.7), point estimates (Eq 3.2),
paired percentile bootstrap over women, Clopper-Pearson intervals, verdicts.

A bootstrap resample is applied as integer weights over women (the number of
times each woman is drawn), so every metric is a weighted metric. All
repetitions and arms share the same weights, which makes differences paired.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.catalog import group_costs, group_ids, group_info

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
SEED = 20260102
RANDOM_ARMS = ("A5", "A6")
METRICS = ["auprc", "auroc", "brier", "cov1", "cov0", "set_size", "singleton", "empty",
           "refer", "routine", "confirm", "miss"]
PROPORTIONS = {"cov1": "pos", "cov0": "neg", "miss": "pos", "routine": "all", "refer": "all",
               "confirm": "all"}


# ---------------------------------------------------------------- arrays
class Prepared:
    """Predictions as arrays of shape (K arm-draws, R repetitions, n women)."""

    def __init__(self, arms):
        self.women = np.sort(arms.row_id.unique())
        n, pos = len(self.women), {r: i for i, r in enumerate(np.sort(arms.row_id.unique()))}
        self.reps = sorted(arms.rep.unique())
        self.keys = sorted(arms.groupby(["arm", "draw"]).size().index, key=lambda k: (k[0], k[1]))
        K, R = len(self.keys), len(self.reps)
        self.p = np.full((K, R, n), np.nan)
        self.in0 = np.zeros((K, R, n), bool)
        self.in1 = np.zeros((K, R, n), bool)
        self.fold = np.zeros((K, R, n), int)
        y = np.full(n, -1)
        kidx = {k: i for i, k in enumerate(self.keys)}
        ridx = {r: i for i, r in enumerate(self.reps)}
        for (arm, d, rep), g in arms.groupby(["arm", "draw", "rep"]):
            k, r = kidx[(arm, d)], ridx[rep]
            j = g.row_id.map(pos).to_numpy()
            if len(np.unique(j)) != n or len(j) != n:
                raise ValueError(f"{arm}/{d}/rep {rep}: each woman must appear exactly once")
            self.p[k, r, j] = g.p_hat
            self.in0[k, r, j] = g.set.isin(["{0}", "{0,1}"])
            self.in1[k, r, j] = g.set.isin(["{1}", "{0,1}"])
            self.fold[k, r, j] = g.fold
            y[j] = g.label
        self.y = y.astype(float)
        self.n_pos, self.n = int(self.y.sum()), n
        # tie-aware sort (descending p) shared by every resample
        self.order = np.argsort(-self.p, axis=2, kind="stable")
        ps = np.take_along_axis(self.p, self.order, 2)
        new_block = np.ones_like(ps, bool)
        new_block[..., 1:] = ps[..., 1:] != ps[..., :-1]
        idx = np.broadcast_to(np.arange(n), ps.shape)
        self.bstart = np.maximum.accumulate(np.where(new_block, idx, 0), axis=2)
        last = np.ones_like(ps, bool)
        last[..., :-1] = new_block[..., 1:]
        end = np.where(last, idx, n)
        self.bend = np.minimum.accumulate(end[..., ::-1], axis=2)[..., ::-1]

    def index(self, arm, draw=None):
        return [i for i, (a, d) in enumerate(self.keys) if a == arm and (draw is None or d == draw)]

    def metrics(self, w=None):
        """Every metric for every arm-draw and repetition: dict name -> (K, R)."""
        w = np.ones(self.n) if w is None else np.asarray(w, float)
        y = self.y
        ws, ys = w[self.order], y[self.order]
        tp, fp = np.cumsum(ws * ys, 2), np.cumsum(ws * (1 - ys), 2)
        TP, FP = tp[..., -1:], fp[..., -1:]
        tp_e, fp_e = np.take_along_axis(tp, self.bend, 2), np.take_along_axis(fp, self.bend, 2)
        prec = np.divide(tp_e, tp_e + fp_e, out=np.zeros_like(tp_e), where=(tp_e + fp_e) > 0)
        auprc = (ws * ys * prec).sum(2) / TP[..., 0]
        fp_before = np.where(self.bstart > 0,
                             np.take_along_axis(fp, np.maximum(self.bstart - 1, 0), 2), 0.0)
        below, tie = FP - fp_e, fp_e - fp_before
        auroc = (ws * ys * (below + 0.5 * tie)).sum(2) / (TP[..., 0] * FP[..., 0])
        W, Wp, Wn = w.sum(), (w * y).sum(), (w * (1 - y)).sum()
        in0, in1 = self.in0, self.in1
        only0, only1, both, empty = in0 & ~in1, in1 & ~in0, in0 & in1, ~in0 & ~in1
        s = lambda m, v=w: (m * v).sum(2)
        return dict(
            auprc=auprc, auroc=auroc, brier=s((self.p - y) ** 2) / W,
            cov1=s(in1, w * y) / Wp, cov0=s(in0, w * (1 - y)) / Wn,
            set_size=s(in0.astype(float) + in1) / W, singleton=s(only0 | only1) / W,
            empty=s(empty) / W, refer=s(only1) / W, routine=s(only0) / W,
            confirm=s(both | empty) / W, miss=s(only0, w * y) / Wp)

    def counts(self):
        """Repetition-averaged counts for Clopper-Pearson: dict name -> (K,)."""
        y = self.y.astype(bool)
        only0 = self.in0 & ~self.in1
        only1 = self.in1 & ~self.in0
        conf = ~(only0 | only1)
        return dict(cov1=self.in1[..., y].sum(2).mean(1), cov0=self.in0[..., ~y].sum(2).mean(1),
                    miss=only0[..., y].sum(2).mean(1), routine=only0.sum(2).mean(1),
                    refer=only1.sum(2).mean(1), confirm=conf.sum(2).mean(1))


def arm_values(prep, m, arm, how="median"):
    """(R,) per-repetition value of an arm; random arms: median (or mean) over draws."""
    v = m[prep.index(arm)]
    if arm in RANDOM_ARMS:
        return np.median(v, 0) if how == "median" else v.mean(0)
    return v[0]


# ---------------------------------------------------------------- intervals
def clopper_pearson(x, n, level):
    """Exact interval via Beta quantiles; x may be a non-integer averaged count."""
    a = 1 - level
    lo = 0.0 if x <= 0 else stats.beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x >= n else stats.beta.ppf(1 - a / 2, x + 1, n - x)
    return float(lo), float(hi)


def pct(dist, level):
    a = (1 - level) / 2 * 100
    return [float(v) for v in np.nanpercentile(dist, [a, 100 - a])]


# ---------------------------------------------------------------- evaluation
def evaluate(arms, meta, B=2000, seed=SEED):
    prep = Prepared(arms)
    present = [a for a in dict.fromkeys(k[0] for k in prep.keys)]
    m0 = prep.metrics()
    rng = np.random.default_rng(seed)
    boot = {a: {k: np.empty(B) for k in METRICS} for a in present}
    diffs = {k: np.empty(B) for k in ("P1_auprc_A3_minus_A1", "S1_auprc_A3_minus_medianA5",
                                      "lasso_minus_A3_auprc", "cov1_A4_minus_A4marginal")}
    for b in range(B):
        w = rng.multinomial(prep.n, np.full(prep.n, 1 / prep.n))
        m = prep.metrics(w)
        for a in present:
            for k in METRICS:
                boot[a][k][b] = arm_values(prep, m[k], a).mean()
        _diffs(prep, m, diffs, b, present)
    cnt = prep.counts()
    n_cls = dict(pos=prep.n_pos, neg=prep.n - prep.n_pos, all=prep.n)
    rows = []
    for a in present:
        for k in METRICS:
            v = arm_values(prep, m0[k], a)
            row = dict(arm=a, metric=k, estimate=float(v.mean()), rep_min=float(v.min()),
                       rep_max=float(v.max()), boot95=pct(boot[a][k], .95),
                       boot975=pct(boot[a][k], .975))
            if k in PROPORTIONS and a not in RANDOM_ARMS:
                x, n = float(cnt[k][prep.index(a)][0]), n_cls[PROPORTIONS[k]]
                row.update(count=x, n=n, cp95=clopper_pearson(x, n, .95),
                           cp975=clopper_pearson(x, n, .975))
            if a in RANDOM_ARMS:
                row["draw_mean"] = float(arm_values(prep, m0[k], a, "mean").mean())
            rows.append(row)
    d0 = {}
    _diffs(prep, m0, d0, None, present)
    diff_rows = {k: dict(estimate=float(d0[k]), boot95=pct(diffs[k], .95),
                         boot975=pct(diffs[k], .975)) for k in d0}
    return dict(metrics=rows, diffs=diff_rows, per_fold=per_fold_coverage(arms),
                instrument=instrument(meta), n=prep.n, n_pos=prep.n_pos, B=B)


def _diffs(prep, m, out, b, present):
    a3 = arm_values(prep, m["auprc"], "A3")
    vals = {"P1_auprc_A3_minus_A1": (a3 - arm_values(prep, m["auprc"], "A1")).mean()}
    if "A5" in present:
        vals["S1_auprc_A3_minus_medianA5"] = (a3 - arm_values(prep, m["auprc"], "A5")).mean()
    if "lasso_tuned" in present:
        vals["lasso_minus_A3_auprc"] = (arm_values(prep, m["auprc"], "lasso_tuned") - a3).mean()
    if "A4_marginal" in present:
        vals["cov1_A4_minus_A4marginal"] = (arm_values(prep, m["cov1"], "A4")
                                            - arm_values(prep, m["cov1"], "A4_marginal")).mean()
    for k, v in vals.items():
        if b is None:
            out[k] = v
        else:
            out[k][b] = v
    if b is not None:
        for k in out:
            if k not in vals:
                out[k][b] = np.nan


def per_fold_coverage(arms, arm="A4"):
    g = arms[(arms.arm == arm) & (arms.label == 1)]
    if g.empty:
        return {}
    c = g.assign(covered=g["set"].isin(["{1}", "{0,1}"])).groupby(["rep", "fold"])["covered"].mean()
    q = c.quantile([0, .25, .5, .75, 1]).tolist()
    return dict(arm=arm, n_folds=len(c), min=q[0], q1=q[1], median=q[2], q3=q[3], max=q[4],
                share_below_090=float((c < 0.90).mean()), share_below_085=float((c < 0.85).mean()),
                values=c.round(4).tolist())


def instrument(meta):
    st = meta["setting"]
    target = {"anemia": "anemia", "anemia_who": "anemia"}.get(st["label"], "ced")
    costs = group_costs(target, st["scheme"])
    main_costs = group_costs(target, "main")
    folds = meta["folds"]
    ks = [f["k_star"] for f in folds]
    freq = pd.Series(0, index=group_ids(target), dtype=float)
    for f in folds:
        freq[f["selected"]] += 1
    freq /= len(folds)
    reported = sorted(freq[freq >= 0.60].index)
    info = group_info(target)
    return dict(
        k_star=dict(mean=float(np.mean(ks)), min=int(min(ks)), max=int(max(ks)), values=ks),
        fold_cost_main=dict(mean=float(np.mean([sum(main_costs[g] for g in f["selected"])
                                                for f in folds]))),
        fold_cost_scheme=dict(mean=float(np.mean([sum(costs[g] for g in f["selected"])
                                                  for f in folds]))),
        reported_short_form=reported, reported_size=len(reported),
        reported_cost_main=int(sum(main_costs[g] for g in reported)),
        full_groups=len(costs), full_cost_main=int(sum(main_costs.values())),
        selection_frequency=[dict(group_id=g, group_name=info.group_name[g],
                                  domain=info.domain[g], tier=int(info.tier[g]),
                                  frequency=float(freq[g])) for g in freq.index])


# ---------------------------------------------------------------- verdicts
def three_way(lo, hi, threshold, above=True):
    """Supported if the whole interval satisfies the criterion, not supported if the
    whole interval violates it, inconclusive otherwise."""
    if above:
        return "supported" if lo > threshold else "not supported" if hi < threshold else "inconclusive"
    raise ValueError


def verdicts(ev):
    m = {(r["arm"], r["metric"]): r for r in ev["metrics"]}
    d = ev["diffs"]
    ins = ev["instrument"]
    out = {}
    # Primary 1: short form <= 20 groups and 97.5% CI of AUPRC(A3)-AUPRC(A1) above -0.05
    lo, hi = d["P1_auprc_A3_minus_A1"]["boot975"]
    size_ok = ins["k_star"]["max"] <= 20 and ins["reported_size"] <= 20
    v = three_way(lo, hi, -0.05)
    out["primary1"] = dict(verdict=v if size_ok else "not supported", criterion=
                           "short form <= 20 groups and 97.5% bootstrap CI of AUPRC(A3)-AUPRC(A1) > -0.05",
                           estimate=d["P1_auprc_A3_minus_A1"]["estimate"], ci975=[lo, hi],
                           k_star_max=ins["k_star"]["max"], reported_size=ins["reported_size"])
    # Primary 2: A4 positive-class coverage, 97.5% Clopper-Pearson vs 0.80
    c = m[("A4", "cov1")]
    lo, hi = c["cp975"]
    out["primary2"] = dict(
        verdict="supported" if lo >= 0.80 else "not supported" if hi < 0.80 else "inconclusive",
        criterion="lower bound of 97.5% Clopper-Pearson interval of A4 positive-class coverage >= 0.80",
        estimate=c["estimate"], count=c["count"], n=c["n"], cp975=[lo, hi],
        below_nominal_flag=bool(hi < 0.90), negative_class_coverage=m[("A4", "cov0")]["estimate"],
        negative_class_cp95=m[("A4", "cov0")]["cp95"])
    # Secondary 1: 95% CI of AUPRC(A3) - median AUPRC(A5 draws) above 0
    if "S1_auprc_A3_minus_medianA5" in d:
        lo, hi = d["S1_auprc_A3_minus_medianA5"]["boot95"]
        out["secondary1"] = dict(verdict=three_way(lo, hi, 0.0), criterion=
                                 "95% bootstrap CI of AUPRC(cost-aware) - median AUPRC(50 random) > 0",
                                 estimate=d["S1_auprc_A3_minus_medianA5"]["estimate"], ci95=[lo, hi])
    # Secondary 2: A4 routine share >= 0.20, 95% Clopper-Pearson
    r = m[("A4", "routine")]
    lo, hi = r["cp95"]
    out["secondary2"] = dict(
        verdict="supported" if lo >= 0.20 else "not supported" if hi < 0.20 else "inconclusive",
        criterion="95% Clopper-Pearson interval of A4 routine share >= 0.20",
        estimate=r["estimate"], count=r["count"], n=r["n"], cp95=[lo, hi])
    return out


def metrics_table(ev, name):
    rows = []
    for r in ev["metrics"]:
        rows.append(dict(setting=name, arm=r["arm"], metric=r["metric"], estimate=r["estimate"],
                         boot95_lo=r["boot95"][0], boot95_hi=r["boot95"][1],
                         boot975_lo=r["boot975"][0], boot975_hi=r["boot975"][1],
                         cp95_lo=r.get("cp95", [np.nan] * 2)[0], cp95_hi=r.get("cp95", [np.nan] * 2)[1],
                         cp975_lo=r.get("cp975", [np.nan] * 2)[0],
                         cp975_hi=r.get("cp975", [np.nan] * 2)[1],
                         rep_min=r["rep_min"], rep_max=r["rep_max"],
                         count=r.get("count", np.nan), n=r.get("n", np.nan),
                         draw_mean=r.get("draw_mean", np.nan)))
    return pd.DataFrame(rows)


def save(ev, name):
    TABLES.mkdir(parents=True, exist_ok=True)
    metrics_table(ev, name).to_csv(TABLES / f"{name}_metrics.csv", index=False)
    pd.DataFrame(ev["instrument"]["selection_frequency"]).to_csv(
        TABLES / f"{name}_selection_frequency.csv", index=False)
    (TABLES / f"{name}_evaluation.json").write_text(json.dumps(ev, indent=1, default=float))
