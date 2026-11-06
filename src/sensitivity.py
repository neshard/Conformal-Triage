"""Section 3.9: the eight sensitivity analyses and subgroup coverage.
Run only after the main results are saved."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from src import evaluate as ev_mod
from src.evaluate import TABLES, Prepared, clopper_pearson
from src.experiment import (Setting, derive_arms, load_results, load_scores, run, sample)

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
INCOME = {0: "<1 juta", 1: "1-2 juta", 2: "2-3 juta", 3: ">3 juta"}


def settings(main, prefix=""):
    """main: dict target -> main Setting. Returns the sensitivity Settings (analyses 1, 3-8);
    analysis 2 (alpha grid) is recomputed from stored scores."""
    out = []
    for t, s in main.items():
        base = replace(s, extras=False)
        out += [replace(base, name=f"{prefix}uniform_{t}", scheme="uniform"),
                replace(base, name=f"{prefix}steep_{t}", scheme="steep"),
                replace(base, name=f"{prefix}loso_{t}", outer="loso", n_reps=1),
                replace(base, name=f"{prefix}noamb_{t}", drop_ambiguous=True),
                replace(base, name=f"{prefix}cal35_{t}", cal_share=0.35)]
    a, c = replace(main["anemia"], extras=False), replace(main["ced"], extras=False)
    out += [replace(a, name=f"{prefix}who_anemia", label="anemia_who"),
            replace(a, name=f"{prefix}hb90_anemia", max_hb_days=90),
            replace(c, name=f"{prefix}ced22", label="ced22"),
            replace(c, name=f"{prefix}ced24", label="ced24")]
    return out


ANALYSIS = {"uniform": "1 cost scheme", "steep": "1 cost scheme", "loso": "3 leave-one-subdistrict-out",
            "who": "4 WHO trimester Hb cutoffs", "ced22": "5 CED cutoff 22 cm",
            "ced24": "5 CED cutoff 24 cm", "noamb": "6 ambiguous duplicates removed",
            "hb90": "7 Hb > 90 days removed", "cal35": "8 calibration share 35%"}


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def run_all(main, n_jobs, B, prefix="", log=print):
    rows = []
    main_forms = {t: ev_mod_load(s.name)["instrument"]["reported_short_form"] for t, s in main.items()}
    for s in settings(main, prefix):
        arms, meta = run(s, n_jobs=n_jobs, log=log)
        e = ev_mod.evaluate(arms, meta, B=B)
        e["verdicts"] = ev_mod.verdicts(e)
        ev_mod.save(e, s.name)
        rows.append(summary_row(s, e, main_forms[s.target]))
        if s.outer == "loso":
            per_group(arms, sample(s), "c006", f"{s.name}_by_subdistrict")
    tab = pd.DataFrame(rows)
    tab.to_csv(TABLES / f"{prefix}sensitivity_summary.csv", index=False)
    for t, s in main.items():
        alpha_grid(s).to_csv(TABLES / f"{s.name}_alpha_grid.csv", index=False)
        arms, _ = load_results(s.name)
        frame = sample(s)
        per_group(arms, frame, "c006", f"{s.name}_by_subdistrict")
        per_group(arms, frame[["c023"]].assign(income=frame.c023.map(INCOME)), "income",
                  f"{s.name}_by_income")
    return tab


def ev_mod_load(name):
    return json.loads((TABLES / f"{name}_evaluation.json").read_text())


def summary_row(s, e, main_form):
    m = {(r["arm"], r["metric"]): r for r in e["metrics"]}
    ins, v = e["instrument"], e["verdicts"]
    kind = next(k for k in ANALYSIS if k in s.name)
    return dict(
        analysis=ANALYSIS[kind], setting=s.name, label=s.label, scheme=s.scheme, n=e["n"],
        n_pos=e["n_pos"], k_star_mean=ins["k_star"]["mean"], reported_size=ins["reported_size"],
        reported_cost_main=ins["reported_cost_main"],
        reported_short_form=" ".join(ins["reported_short_form"]),
        jaccard_vs_main=jaccard(ins["reported_short_form"], main_form),
        auprc_A1=m[("A1", "auprc")]["estimate"], auprc_A3=m[("A3", "auprc")]["estimate"],
        auprc_diff=e["diffs"]["P1_auprc_A3_minus_A1"]["estimate"],
        auprc_diff_ci975=e["diffs"]["P1_auprc_A3_minus_A1"]["boot975"],
        cov1_A4=m[("A4", "cov1")]["estimate"], cov1_A4_cp975=m[("A4", "cov1")]["cp975"],
        routine_A4=m[("A4", "routine")]["estimate"], routine_A4_cp95=m[("A4", "routine")]["cp95"],
        **{f"verdict_{k}": x["verdict"] for k, x in v.items()})


def alpha_grid(s):
    """Analysis 2: alpha 0.05-0.30 from stored calibration and test scores (no refit)."""
    scores = load_scores(s.name)
    scores = scores[scores.itemset.isin(["full", "short"])]
    rows = []
    for a in ALPHAS:
        arms = derive_arms(scores, a)
        arms = arms[arms.arm.isin(["A1", "A2", "A3", "A4"])]
        prep = Prepared(arms)
        m, cnt = prep.metrics(), prep.counts()
        for arm in ["A1", "A2", "A3", "A4"]:
            k = prep.index(arm)[0]
            row = dict(alpha=a, arm=arm)
            for name in ("cov1", "cov0", "routine", "refer", "confirm", "miss", "empty", "set_size"):
                row[name] = float(m[name][k].mean())
            for name, n in (("cov1", prep.n_pos), ("routine", prep.n)):
                row[f"{name}_cp95"] = clopper_pearson(float(cnt[name][k]), n, .95)
            rows.append(row)
    return pd.DataFrame(rows)


def per_group(arms, frame, col, name, arm="A4"):
    """Coverage of positives and routine share per subgroup (repetition-averaged counts,
    95% Clopper-Pearson at the number of distinct women in the subgroup)."""
    a = arms[arms.arm == arm].merge(frame[[col]], left_on="row_id", right_index=True)
    rows = []
    for g, d in a.groupby(col):
        reps = d.rep.nunique()
        pos = d[d.label == 1]
        n_pos, n = pos.row_id.nunique(), d.row_id.nunique()
        x_cov = pos.set.isin(["{1}", "{0,1}"]).sum() / reps
        x_rout = (d.set == "{0}").sum() / reps
        rows.append(dict(group=g, n=n, n_pos=n_pos,
                         coverage=x_cov / n_pos if n_pos else np.nan,
                         coverage_cp95=clopper_pearson(x_cov, n_pos, .95) if n_pos else None,
                         routine=x_rout / n, routine_cp95=clopper_pearson(x_rout, n, .95)))
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / f"{name}.csv", index=False)
    return out
