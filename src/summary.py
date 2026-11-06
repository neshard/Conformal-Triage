"""results/SUMMARY.md from stored evaluations (no recomputation)."""
import json
from pathlib import Path

import pandas as pd

from src.data import REPORT
from src.evaluate import TABLES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "SUMMARY.md"
TARGET_NAME = {"anemia": "Anemia (Hb < 11.0 g/dL)", "ced": "CED risk (MUAC <= 23 cm)"}
ARM_ROWS = ["A1", "A2", "A3", "A4", "A5", "A6", "lasso_tuned", "lasso_conformal", "A4_marginal",
            "xgb_full_tuned", "xgb_full_conformal", "xgb_short_tuned", "xgb_short_conformal"]

DEVIATIONS = """\
- **Deduplication.** The raw export has no respondent-name column. Duplicates were resolved once from
  the named export through `scripts/build_dedup_decisions.py`, which writes only
  `config/dedup_decisions.yaml`. The pipeline reads the raw export plus that YAML. The proposal's rule
  (same normalised name, age, enumerator and subdistrict, at most 20 of 144 items differing) gave **4**
  duplicate pairs, not 5, and **1** ambiguous pair (59 differing items), not 2. Item comparison ignores
  case and repeated whitespace and treats a blank answer as equal to a blank answer. The final samples are
  therefore 393 (anemia) and 407 (CED), one woman more than the proposal's 392 / 406; the CED positives are
  109, not 108. Analysis 6 removes the single ambiguous pair (392 / 406).
- **Row 240.** Col 14 age is 3; the plausibility rule sets it to missing. Col 3 (an admin-block copy
  of age, 30) is not used as a fallback.
- **Gestational age.** Text answers are parsed as weeks ("24 minggu" -> 24, "8bln (32minggu)" -> 32).
  "6 bulan" (months only) and "A" are missing.
- **Dates (sensitivity 7 only).** Dates are DD/MM/YYYY; entries with a 2-digit year are MM/DD/YY.
  Years that are typos (2015, 0225, 1015) are left unparsed, and those records are kept in
  analysis 7 because they cannot be shown to exceed 90 days. The gap is the absolute difference.
- **Tuned threshold.** "Women above t are referred" is read as p_hat >= t, which is the condition
  that defines t; with continuous scores ties at t essentially never occur on test women.
- **CED cutoffs 22 and 24 cm** are applied as MUAC <= cutoff, parallel to the main rule (<= 23).
- **Analysis 6** drops the later record (by timestamp) of each ambiguous pair.
- **Multi-answer free text.** Five "other" answers to the anemia-symptom question (col 68) are not
  form options and get no column (decision: kept ignored, since a column that is 1 for five women cannot
  be learned or selected).
- **MUAC as an anemia predictor** keeps its recorded values; Sec 3.4 lists no plausibility range
  for it, so out-of-range values (0, 10, 12, 66, 77) in the anemia sample are used as recorded. The main
  analysis was not changed after the results were known. This is reported as a limitation: the values
  are implausible for a pregnant woman, and a post-hoc run with them set to missing was not performed.
- **Implementation details not fixed by the proposal:** SE_k uses the sample SD (ddof = 1);
  the lasso path uses 100 C values in [1e-4, 1e2] (liblinear, L1); the XGBoost search space is
  listed in `src/models.py`. Preprocessing is fitted once per inner fold on all groups and the
  columns of each candidate set are sliced; every step is per column, so this equals refitting
  on the candidate set (tested in `tests/test_preprocess.py`).
"""


def _ev(name):
    return json.loads((TABLES / f"{name}_evaluation.json").read_text())


def _ci(lo_hi):
    return f"[{lo_hi[0]:.3f}, {lo_hi[1]:.3f}]"


def write(main, prefix=""):
    rep = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    L = ["# Results summary", ""]
    if prefix:
        L += ["**SMOKE TEST OUTPUT. Not results.**", ""]
    L += ["## Sample sizes (Table 3.1)", "",
          "| Step | Records |", "|---|---|",
          f"| Raw data | {rep.get('raw')} |",
          f"| Duplicates removed (earliest kept) | {rep.get('duplicates_removed')} -> {rep.get('after_dedup')} |",
          f"| Hb = 0 treated as not measured | {rep.get('hb_zero')} -> {rep.get('hb_measured')} |",
          f"| Hb > 20 g/dL removed | {rep.get('hb_above_20')} -> {rep.get('anemia_n')} (anemia) |",
          f"| MUAC outside 15-45 cm removed | {rep.get('muac_out')} -> {rep.get('ced_n')} (CED) |",
          "", f"Positives: anemia {rep.get('anemia_pos')} of {rep.get('anemia_n')}, "
              f"CED {rep.get('ced_pos')} of {rep.get('ced_n')}.", ""]
    verdicts = {}
    for t, s in main.items():
        e = _ev(s.name)
        verdicts[t] = e["verdicts"]
        m = {(r["arm"], r["metric"]): r for r in e["metrics"]}
        ins = e["instrument"]
        L += [f"## {TARGET_NAME[t]}", "",
              f"n = {e['n']}, positives = {e['n_pos']}. Alpha = {s.alpha}, cost scheme {s.scheme}, "
              f"calibration share {s.cal_share}. Bootstrap B = {e['B']}.", "",
              f"Short form: k* per fold mean {ins['k_star']['mean']:.1f} "
              f"(range {ins['k_star']['min']}-{ins['k_star']['max']}); reported short form "
              f"(selected in >= 60% of folds): {ins['reported_size']} groups, field cost "
              f"{ins['reported_cost_main']} of {ins['full_cost_main']}: "
              f"{', '.join(ins['reported_short_form']) or '(none)'}.", "",
              "| Arm | AUPRC [95%] | AUROC | Brier | Coverage pos [97.5% CP] | Coverage neg | "
              "Set size | Singleton | Empty | Refer | Routine [95% CP] | Confirm | Miss |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in ARM_ROWS:
            if (a, "auprc") not in m:
                continue
            g = lambda k: m[(a, k)]["estimate"]
            cov = m[(a, "cov1")]
            rout = m[(a, "routine")]
            L.append(f"| {a} | {g('auprc'):.3f} {_ci(m[(a, 'auprc')]['boot95'])} | {g('auroc'):.3f} | "
                     f"{g('brier'):.3f} | {g('cov1'):.3f} {_ci(cov['cp975']) if 'cp975' in cov else ''} | "
                     f"{g('cov0'):.3f} | {g('set_size'):.3f} | {g('singleton'):.3f} | {g('empty'):.3f} | "
                     f"{g('refer'):.3f} | "
                     f"{g('routine'):.3f} {_ci(rout['cp95']) if 'cp95' in rout else ''} | "
                     f"{g('confirm'):.3f} | {g('miss'):.3f} |")
        pf = e["per_fold"]
        L += ["", "A5/A6 rows are the median over the 50 random draws.",
              f"Per-fold positive-class coverage of A4 over {pf['n_folds']} folds: min {pf['min']:.3f}, "
              f"median {pf['median']:.3f}, max {pf['max']:.3f}; share of folds below 0.90: "
              f"{pf['share_below_090']:.2f}, below 0.85: {pf['share_below_085']:.2f}.", "",
              "### Verdicts (Table 1.1)", ""]
        for k, v in e["verdicts"].items():
            ci = v.get("ci975") or v.get("cp975") or v.get("ci95") or v.get("cp95")
            extra = ""
            if k == "primary2":
                extra = (f"; below-nominal flag: {v['below_nominal_flag']}; negative-class coverage "
                         f"{v['negative_class_coverage']:.3f}")
            L.append(f"- **{k}: {v['verdict']}** - estimate {v['estimate']:.3f}, interval "
                     f"{_ci(ci)}{extra}. Criterion: {v['criterion']}.")
        L.append("")
    sens = TABLES / f"{prefix}sensitivity_summary.csv"
    if sens.exists():
        tab = pd.read_csv(sens)
        L += ["## Sensitivity analyses", "",
              "| Analysis | Setting | n (pos) | Reported size / cost | Jaccard vs main | AUPRC A3-A1 | "
              "Coverage A4 | Routine A4 | P1 | P2 | S1 | S2 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in tab.iterrows():
            L.append(f"| {r.analysis} | {r.setting} | {r.n} ({r.n_pos}) | {r.reported_size} / "
                     f"{r.reported_cost_main} | {r.jaccard_vs_main:.2f} | {r.auprc_diff:.3f} | "
                     f"{r.cov1_A4:.3f} | {r.routine_A4:.3f} | {r.verdict_primary1} | "
                     f"{r.verdict_primary2} | {r.get('verdict_secondary1', '')} | {r.verdict_secondary2} |")
        L += ["", "Alpha grid (analysis 2) and subgroup tables are in `results/tables/`.", ""]
    L += ["## Deviations and interpretations", "", DEVIATIONS]
    OUT.write_text("\n".join(L), encoding="utf-8")
    return OUT
