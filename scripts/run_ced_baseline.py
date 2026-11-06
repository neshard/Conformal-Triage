"""Post-hoc: simple CED baselines on the main CED folds (same outer folds and D_fit / D_cal splits).
B2 = pre-pregnancy weight + maternal age; B3 = B2 + height. Logistic regression (C tuned on D_fit),
class-conditional conformal at alpha = 0.10. Writes results/tables/ced_baseline.csv.
Run: python -m scripts.run_ced_baseline"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src import conformal as cf
from src.experiment import main_settings, outer_splits, sample
from src.models import tune_lr
from src.preprocess import feature_columns

SETS = {"B2": ["G089", "G003"], "B3": ["G089", "G090", "G003"]}
ALPHA = 0.10

s = {x.target: x for x in main_settings()}["ced"]
frame = sample(s)
y = frame[s.label].astype(int)
X = frame[feature_columns(frame, s.target)]
rows = []
for rep, fold, tr, te in outer_splits(s, frame):
    seed = 1000 * rep + fold                      # as in run_fold
    fit_ids, cal_ids = train_test_split(tr, test_size=s.cal_share, stratify=y.loc[tr], random_state=seed)
    for name, gs in SETS.items():
        m, _ = tune_lr(X.loc[fit_ids], y.loc[fit_ids].to_numpy(), s.target, gs, seed)
        pc, pt = m.predict_proba(X.loc[cal_ids])[:, 1], m.predict_proba(X.loc[te])[:, 1]
        q0, q1 = cf.class_thresholds(pc, y.loc[cal_ids], ALPHA)
        in0, in1 = cf.prediction_sets(pt, q0, q1)
        rows.append(pd.DataFrame(dict(name=name, rep=rep, fold=fold, p=pt, in0=in0, in1=in1,
                                      y=y.loc[te].to_numpy(), row=te)))
d = pd.concat(rows, ignore_index=True)
out = []
for name, g in d.groupby("name"):
    per = []
    for rep, r in g.groupby("rep"):
        pos = r.y == 1
        routine = (r.in0 & ~r.in1)
        per.append(dict(auprc=average_precision_score(r.y, r.p), auroc=roc_auc_score(r.y, r.p),
                        cov1=r.in1[pos].mean(), routine=routine.mean(), miss=routine[pos].mean(),
                        empty=(~r.in0 & ~r.in1).mean(), set_size=(r.in0.astype(int) + r.in1.astype(int)).mean()))
    out.append(dict(name=name, groups="+".join(SETS[name]), **pd.DataFrame(per).mean().to_dict()))
res = pd.DataFrame(out)
res.to_csv("results/tables/ced_baseline.csv", index=False)
print(res.to_string())

# paired bootstrap over women: AUPRC(B) - AUPRC(A4), 2,000 resamples, averaged over repetitions
a4 = pd.read_parquet("results/predictions/main_ced.parquet")
a4 = a4[a4.arm == "A4"][["row_id", "rep", "p_hat", "label"]]
rng = np.random.default_rng(0)
ids = np.array(sorted(frame.index))
rep_of = {}
for name in SETS:
    b = d[d.name == name][["row", "rep", "p", "y"]].rename(columns={"row": "row_id"})
    mrg = b.merge(a4, on=["row_id", "rep"], )
    assert (mrg.y == mrg.label).all()
    reps = sorted(mrg.rep.unique())
    by_rep = {r: mrg[mrg.rep == r].set_index("row_id") for r in reps}
    diffs = []
    for _ in range(2000):
        samp = rng.choice(ids, len(ids), replace=True)
        v = []
        for r in reps:
            t = by_rep[r].reindex(samp).dropna()
            v.append(average_precision_score(t.y, t.p) - average_precision_score(t.y, t.p_hat))
        diffs.append(np.mean(v))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(name, "AUPRC diff vs A4: mean %.3f, 95%% CI [%.3f, %.3f]" % (np.mean(diffs), lo, hi))
    res.loc[res.name == name, ["diff_vs_A4_lo", "diff_vs_A4_hi"]] = [lo, hi]
res.to_csv("results/tables/ced_baseline.csv", index=False)
