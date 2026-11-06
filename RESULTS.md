# Conformal Triage: Final Report of Experiments and Methodology

Project: *Fewer Questions, Guaranteed Coverage: Conformal Triage for Maternal Anemia and Chronic Energy
Deficiency Screening in North Lombok, Indonesia* (specification: `docs/proposal.pdf`).

> **Status of the numbers.** These are the **final results of the full run** (5 repetitions x 5 outer folds,
> K_max = 20, 50 random draws, 2,000 bootstrap resamples) on the deduplicated sample of **393 women for anemia
> (85 positive) and 407 for CED risk (109 positive)**. The sample differs by one woman from the proposal
> (392 / 406, 108 CED positives) because the named dedup rule found 4 duplicate pairs, not 5 (Section 2).
> All results are internal validation on one regional dataset (Section 11).

---

## 1. Goal and research questions

Screening for maternal anemia (Hb < 11.0 g/dL) and chronic energy deficiency (CED, MUAC <= 23 cm) in North
Lombok currently relies on a 112-group questionnaire plus a blood test or tape measurement. The study asks:

| RQ | Question |
|---|---|
| 1 | How many question groups does a cost-aware short form need to stay within 0.05 AUPRC of the full questionnaire? |
| 2 | Does a class-conditional conformal layer on the short form keep positive-class coverage near the nominal 1 - alpha? |
| 3 | At alpha = 0.10, what share of women can go to routine follow-up without a confirmatory test, compared with a tuned threshold at the same sensitivity? |
| 4 | Does cost-aware selection beat random selection of the same size? |

Each question is answered separately for anemia and for CED risk. The contribution is the *combination*
(question-collection cost + class-conditional conformal triage) on a community questionnaire, not a new algorithm.

## 2. Data

- **Source:** one questionnaire wave, five subdistricts of North Lombok (Gangga, Pemenang, Tanjung, Bayan,
  Kayangan). Raw export is 420 records x 166 columns, latin-1, `;`-separated (`data/raw/kuesioner_raw.csv`, git-ignored).
- **Targets:** anemia = Hb < 11.0 g/dL. CED risk = MUAC <= 23 cm (an operational label, not a diagnosis).
  The two targets are modelled separately.
- **Cleaning (Table 3.1):** remove duplicate submissions (earliest kept); Hb = 0 treated as not measured;
  Hb > 20 removed; MUAC outside 15 to 45 cm removed. No label is imputed.

| Step | Proposal | Final |
|---|---|---|
| Raw | 420 | 420 |
| Duplicates removed (earliest kept) | 5 -> 415 | **4 -> 416** |
| Hb = 0 treated as not measured | 16 -> 399 | 16 -> 400 |
| Hb > 20 removed (anemia) | 7 -> 392 (85 positive) | 7 -> **393** (85 positive) |
| MUAC outside 15-45 removed (CED) | 9 -> 406 (108 positive) | 9 -> **407** (**109** positive) |

**Deduplication.** The named export aligns 1:1 with the raw file (420 rows, matched on timestamp, age, enumerator
and subdistrict). Under the proposal's rule (same normalised name, age, enumerator and subdistrict, at most 20 of
144 items differing) it finds **4** duplicate pairs (rows 21-344, 30-37, 66-147, 109-288), not 5, and **one**
ambiguous pair (rows 258-372, 59 differing items), not two. Checks made before the count was fixed: blank answers are
stored as empty strings and already count as equal; item comparison now ignores case and repeated whitespace; no pair
of same-name women falls between 21 and 30 differing items; no near-identical names (highest similarity 0.61) were
missed. The three records that a name-free rule had grouped as a triple (rows 190, 194, 215) have different names and
are different women. The data count (416) is therefore final, and the expected counts in `src/data.py` were updated
from the proposal's. Analysis 6 drops the later record of the ambiguous pair (392 anemia / 406 CED, 85 / 108 positive).
- **Privacy:** names, phone values, respondent codes, enumerator names, villages, health-centre names and GPS are
  never written outside `data/raw/`. Only `scripts/build_dedup_decisions.py` may open the named export, and it
  writes row indices only.

**Labels in the final data.** Anemia: 85 of 393 women (21.63%) are below 11.0 g/dL, 57 mild and 28 moderate (no
severe); Hb is recorded in whole numbers and 104 women sit exactly at 11 g/dL. CED: 109 of 407 women (26.78%) have MUAC
<= 23 cm, 53 of them exactly at 23 cm. Among the 385 women valid for both targets, 21 have both conditions, 63 anemia only,
83 CED risk only and 218 neither.

## 3. Question catalog and field costs

- Every candidate predictor is a **question group** (one question plus every column it produces after encoding;
  a food-frequency item is grouped with its portion question). 112 groups for anemia, 111 for CED (MUAC is
  excluded from CED). Catalog frozen on 2026-11-01 (`config/question_catalog.csv`, SHA-256 recorded in `PLAN.md`).
- **Cost tiers** (ordinal judgements made before any modelling, reported as assumptions):

| Tier | What it covers | Cost (main) |
|---|---|---|
| 0 | context questions answered in seconds (age, education, subdistrict) | 1 |
| 1 | recall or linked-answer interview items (HFIAS, FFQ, IFA, knowledge) | 2 |
| 2 | needs an instrument or written record (weight, height, MUAC) | 5 |

  Total field cost: 214 (anemia), 209 (CED). Alternative schemes for sensitivity: **uniform** 1/1/1, **steep** 1/3/10.
  Only the ratios matter because cost enters selection only as a denominator.
- **Excluded:** identifiers, village / health-centre names, the gravida-para-abortus field, measurement dates,
  empty columns, open-text follow-ups, and the "was a Hb test ever done" item (it describes how the label was produced).
- **Preprocessing** (fitted on the training part of each fold only): leakage assertion (Hb never in either matrix,
  MUAC never in the CED matrix), plausibility rules (age 15-50, weight 30-120 kg, height 130-190 cm, gestational
  age 1-42 weeks -> missing), ordinal/one-hot/multi-binary encoding, median/mode imputation, standardisation for
  logistic regression. Multi-answer items are split against explicit option lists, not on commas.

## 4. Methodology

### 4.1 Pipeline overview

```
cleaned data -> repeated stratified 5x5 outer CV
  each outer training fold -> split by class into D_fit (75%) and D_cal (25%)
     fitting branch (D_fit only):  cost-aware greedy selection (inner 5-fold CV)
                                   -> one-standard-error rule -> short form S_k*
                                   -> tune + fit model on S_k*
     calibration branch (D_cal):   untouched until the model is fixed
                                   -> class-conditional thresholds q0, q1
  outer test fold -> prediction set C(x) for every woman -> triage action
```

Keeping D_cal out of selection and tuning is what keeps the coverage guarantee valid. The loop uses 5
repetitions (seeds 0 to 4) of stratified 5-fold CV, so every woman gets one out-of-fold prediction per
repetition from a model that never saw her. All arms share the same folds and the same D_fit / D_cal splits.

### 4.2 Cost-aware greedy selection (Algorithm 3.1)

- A(S) = mean inner-CV AUPRC of logistic regression (C = 1) on the groups in S; A(empty) = prevalence.
- At each step add the group with the largest gain per unit cost,
  `g* = argmax_g [A(S + g) - A(S)] / c_g`, up to K_max = 20 groups.
- **One-standard-error rule:** let j = argmax_k A(S_k); choose the smallest k with `A(S_k) >= A(S_j) - SE_j`.
- The rule is applied in every outer fold, so reported performance includes the choice of length. The *reported*
  short form is the set of groups selected in at least 60% of outer folds; selection frequency of every group is reported.
- Comparator: lasso (L1 logistic) ranking of groups, keeping the same k*.

### 4.3 Models

- **Main:** logistic regression, C tuned on D_fit over 10 values in [1e-3, 1e2] (inner 3-fold CV, AUPRC).
- **Exploratory (no verdict):** XGBoost on the short form and on the full questionnaire, 30 random
  hyper-parameter configurations under the same inner CV.

### 4.4 Class-conditional conformal triage

- Non-conformity score from the fitted model (Eq. 2.1), thresholds `q0` and `q1` computed separately within each class
  on D_cal, prediction set C(x) from Eq. 2.2. Under exchangeability, a positive woman receives set {0} with
  probability at most alpha.
- **Triage rule (Table 3.5):** {1} -> refer; {0} -> routine follow-up; {0,1} or empty -> confirmatory test.
- Main alpha = 0.10. A positive woman is *missed* only when her set is {0}.
- **Calibration size caveat (Table 3.4):** with ~17 positive calibration women (anemia), alpha = 0.05 gives an infinite
  threshold (no routine routing is possible), and alpha = 0.10 is conservative (expected coverage about 0.944, 6% chance
  that one calibration set gives coverage below 0.85). CED has ~22 positives (expected coverage 0.913, 14% chance below 0.85).
- **Tuned threshold:** the largest t on D_cal such that at least 1 - alpha of positive calibration women have p_hat >= t.
  It targets the same sensitivity but has no finite-sample correction and no abstention.

### 4.5 Experimental design: 3 x 2 factorial

| Arm | Item set | Decision layer |
|---|---|---|
| A1 | Full questionnaire | Tuned threshold |
| A2 | Full questionnaire | Conformal |
| A3 | Cost-aware short form | Tuned threshold |
| A4 | Cost-aware short form | **Conformal (proposed)** |
| A5 | Random short form, 50 draws of size k* | Tuned threshold |
| A6 | Random short form, 50 draws of size k* | Conformal |

Two further comparisons without a verdict: **lasso selection** vs cost-aware selection (same k*), and **marginal
conformal** prediction (a single threshold for all women) vs class-conditional, to show how much positive-class
coverage is lost without the class-conditional guarantee.

### 4.6 Hypotheses and verdict rule (Table 1.1)

A hypothesis is *supported* when its whole interval satisfies the criterion, *not supported* when the whole
interval violates it, and *inconclusive* otherwise. Primary hypotheses use 97.5% intervals (Bonferroni over two
primaries), secondary ones use 95%.

| Hypothesis | Criterion | Comparison |
|---|---|---|
| Primary 1: short form non-inferior | short form <= 20 groups and paired AUPRC(A3) - AUPRC(A1) > -0.05 | A1 vs A3 |
| Primary 2: coverage not materially below nominal | lower 97.5% Clopper-Pearson bound of A4 positive-class coverage >= 0.80 | A4 |
| Secondary 1: selection beats chance | paired AUPRC(A4 short form) minus median of 50 random sets > 0 | A4 vs A6 |
| Secondary 2: triage avoids tests | routine-follow-up share >= 0.20 at alpha = 0.10 | A4 |

The 0.05 / 0.10 / 0.20 margins are feasibility criteria, not clinical standards.

### 4.7 Evaluation and inference

- **Metrics (Table 3.7):** AUPRC (primary), AUROC, Brier; coverage per class, mean set size, share of singleton and
  empty sets; refer / routine / confirm shares and miss rate among positives; number of groups, field cost, selection frequency.
- **Point estimate:** metric computed on the pooled out-of-fold predictions of each repetition, then averaged
  over the 5 repetitions (Eq. 3.2).
- **Intervals:** percentile bootstrap over women (2,000 resamples; the same resampled women are used in all
  repetitions and in both arms of any difference, so differences are paired). Proportions use exact Clopper-Pearson
  at the number of distinct women in the class (85 anemic, 109 at risk of CED).
- **Honest limits stated up front:** the intervals treat the fold models as fixed (no refitting of selection inside
  each resample), so they can be narrower than true uncertainty. The range of M_r across repetitions and per-fold
  coverage are reported beside every interval.

### 4.8 Sensitivity analyses

| # | Analysis |
|---|---|
| 1 | uniform and steep cost schemes (with Jaccard overlap of the short form vs main) |
| 2 | alpha grid 0.05 to 0.30 |
| 3 | leave-one-subdistrict-out validation |
| 4 | WHO trimester-specific Hb cutoffs |
| 5 | CED cutoffs 22 cm and 24 cm |
| 6 | remove the ambiguous duplicate pair |
| 7 | remove Hb measured more than 90 days from the interview |
| 8 | calibration share 35% instead of 25% |
| + | subgroup coverage and routine rate by subdistrict and income band |

### 4.9 Reproducibility and verification

- Fixed seeds for splits, random short forms and models; library versions in `results/environment.txt`
  (Python 3.14.4, numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, xgboost 3.4.1).
- Unit tests (`tests/`): conformal layer against a brute-force computation and Table 3.4; leakage assertion;
  preprocessing slices equal refitting on the candidate set; weighted metrics; one-SE rule; every multi-answer response
  fully consumed by the option list.
- Resumable run with per-fold caching (`results/cache/`), so all arms use the same women.
- Entry point: `python -m src.run --stage all` (add `--smoke` for the reduced run).

## 5. Implementation map

| Module | Role |
|---|---|
| `src/data.py`, `src/preprocess.py`, `src/multianswer.py` | cleaning, labels, encoding, leakage checks |
| `src/catalog.py` | groups, tiers, cost schemes |
| `src/selection.py` | Algorithm 3.1, one-SE rule, lasso comparator |
| `src/models.py` | tuned logistic regression, XGBoost |
| `src/conformal.py` | conformal sets, triage, marginal variant, tuned threshold |
| `src/experiment.py` | outer CV, D_fit / D_cal, arms A1-A6 |
| `src/evaluate.py` | metrics, bootstrap, Clopper-Pearson, verdicts |
| `src/sensitivity.py`, `src/figures.py`, `src/summary.py`, `src/run.py` | sensitivity, figures, report, entry point |

## 6. Decisions and deviations recorded so far

- **Age:** column 14 is authoritative; row 240 has age 3, which the plausibility rule sets to missing (no fallback to column 3).
- **Gestational age** text is parsed as weeks ("8bln (32minggu)" -> 32); month-only answers and invalid text become missing.
- **Tuned threshold** is read as p_hat >= t.
- **MUAC as an anemia predictor** keeps recorded values (the proposal gives no plausibility range for it), so values
  such as 0, 10, 12, 66, 77 are used as recorded in the anemia sample. The main analysis was not changed after the
  results were known; this is a limitation (Section 11), and no post-hoc run with these values set to missing was made.
- **Free-text "other" symptoms** (col 68, five records) get no column (decision 2026-11-02): a column equal to 1 for five women cannot be learned or selected.
- **Deduplication.** The same rule as the proposal gives 4 duplicate pairs (not 5) and 1 ambiguous pair (not 2);
  item comparison ignores case and repeated whitespace and treats blank = blank. Samples are 393 / 407, not 392 / 406.
- **Not fixed by the proposal:** SE_k uses sample SD (ddof = 1); the lasso path uses 100 values of C in [1e-4, 1e2]
  (liblinear, L1); the XGBoost search space is in `src/models.py`.
- **Analysis 6** drops the later record of each ambiguous pair; **CED cutoffs** use MUAC <= cutoff.

## 7. Outputs

`results/predictions/` (all out-of-fold predictions), `results/selection/` (selected groups and k* per fold),
`results/tables/` (metrics, evaluation JSONs, alpha grid, subgroups, selection curves and frequencies, sensitivity
summary), `results/figures/`, `results/SUMMARY.md` (full tables including every arm), `results/environment.txt`,
`results/run_log.txt`.

## 8. Run configuration

5 repetitions x 5 outer folds (25 folds per setting), K_max = 20, inner CV as in Section 4, 50 random draws for A5/A6
(median reported), B = 2,000 bootstrap resamples, alpha = 0.10, calibration share 0.25, main cost scheme 1/2/5.
Wall time: main anemia 2,040 s, main CED 1,178 s, about 500 s per sensitivity setting; the whole run took 22,619 s
(about 6.3 h, including a period when the machine was suspended). Dedup was applied through
`config/dedup_decisions.yaml`; the 46 unit tests pass.

**Code version.** The full run was produced by git commit `46e509355c6ae66de171c3b602f0d77b4ec7046e`
(`46e5093`, branch `master`), made before any further change. The `results/` outputs, `data/processed/` and
this file are not part of that commit.

## 9. Final results

Point estimates are repetition-averaged (Eq. 3.2); intervals are 95% bootstrap (AUPRC) or Clopper-Pearson
(proportions). Full arm tables with Brier, set size, singleton and empty shares and the negative-class coverage are in
`results/SUMMARY.md`.

**Correction.** The positive-class coverage intervals printed in the 9.1 and 9.2 tables below are 97.5% Clopper-Pearson (the Primary 2 interval), not 95%. The 95% intervals for A4 are anemia [0.859, 0.977] and CED [0.827, 0.949] (`cp95` in `results/tables/*_evaluation.json`; `docs/paper_tables.md`). Routine-share intervals are 95%.

### 9.1 Anemia (Hb < 11.0 g/dL; n = 393, 85 positive, prevalence 0.216)

| Arm | AUPRC [95%] | AUROC | Pos. coverage [CP] | Neg. coverage | Set size | Routine share [CP] | Confirm | Miss |
|---|---|---|---|---|---|---|---|---|
| A1 full + tuned | 0.313 [0.254, 0.393] | 0.598 | 0.873 [0.770, 0.942] | 0.177 | 1.00 | 0.166 [0.131, 0.207] | 0 | 0.127 |
| A2 full + conformal | 0.313 | 0.598 | 0.941 [0.857, 0.984] | 0.904 | 1.79 | 0.090 [0.064, 0.123] | 0.792 | 0.059 |
| A3 short + tuned | 0.278 [0.227, 0.347] | 0.560 | 0.899 [0.801, 0.959] | 0.135 | 1.00 | 0.128 [0.096, 0.165] | 0 | 0.101 |
| **A4 short + conformal** | 0.278 | 0.560 | **0.934 [0.847, 0.980]** | 0.916 | 1.83 | **0.077 [0.053, 0.108]** | 0.827 | 0.066 |
| A5 random + tuned (median) | 0.253 [0.214, 0.305] | 0.544 | 0.900 | 0.122 | 1.00 | 0.118 | 0 | 0.100 |
| A6 random + conformal (median) | 0.253 | 0.544 | 0.954 | 0.908 | 1.84 | 0.059 | 0.841 | 0.046 |
| lasso + conformal | 0.297 [0.242, 0.373] | 0.589 | 0.946 [0.863, 0.986] | 0.899 | 1.80 | 0.079 [0.054, 0.110] | 0.803 | 0.054 |
| A4 marginal | 0.278 | 0.560 | 0.595 [0.468, 0.714] | 0.989 | 1.55 | 0.440 [0.390, 0.491] | 0.546 | 0.405 |
| XGBoost full + conformal | 0.279 [0.227, 0.354] | 0.581 | 0.951 [0.870, 0.988] | 0.908 | 1.82 | 0.075 [0.051, 0.106] | 0.818 | 0.049 |
| XGBoost short + conformal | 0.278 [0.230, 0.346] | 0.562 | 0.929 [0.841, 0.978] | 0.919 | 1.83 | 0.073 [0.050, 0.104] | 0.832 | 0.071 |

- **Discrimination is weak** for every model (AUROC 0.56-0.60, AUPRC about 0.25-0.31 against a prevalence of 0.216),
  including XGBoost on the full questionnaire. The questionnaire carries little information about Hb.
- **Instrument.** k* per fold: mean 8.8, range 4-20. The group selected in at least 60% of folds is a single one,
  household assets (G013, 88% of folds, cost 1 of 214). Next most frequent: maternal occupation (52%), transport to
  health service (40%). The evaluated arms A3/A4 use each fold's own k*, not only this one group. The selected groups are
  cheap tier-0 context questions.

### 9.2 CED risk (MUAC <= 23 cm; n = 407, 109 positive, prevalence 0.268)

| Arm | AUPRC [95%] | AUROC | Pos. coverage [CP] | Neg. coverage | Set size | Routine share [CP] | Confirm | Miss |
|---|---|---|---|---|---|---|---|---|
| A1 full + tuned | 0.457 [0.387, 0.540] | 0.683 | 0.877 [0.789, 0.938] | 0.295 | 1.00 | 0.249 [0.207, 0.294] | 0 | 0.123 |
| A2 full + conformal | 0.457 | 0.683 | 0.908 [0.827, 0.960] | 0.902 | 1.67 | 0.183 [0.147, 0.224] | 0.665 | 0.092 |
| A3 short + tuned | 0.513 [0.436, 0.597] | 0.737 | 0.851 [0.759, 0.919] | 0.441 | 1.00 | 0.363 [0.316, 0.411] | 0 | 0.149 |
| **A4 short + conformal** | 0.513 | 0.737 | **0.899 [0.816, 0.954]** | 0.901 | 1.57 | **0.255 [0.213, 0.300]** | 0.574 | 0.101 |
| A5 random + tuned (median) | 0.340 [0.295, 0.397] | 0.582 | 0.875 | 0.194 | 1.00 | 0.179 | 0 | 0.125 |
| A6 random + conformal (median) | 0.340 | 0.582 | 0.912 | 0.899 | 1.76 | 0.121 | 0.756 | 0.088 |
| lasso + conformal | 0.521 [0.446, 0.607] | 0.744 | 0.892 [0.807, 0.949] | 0.899 | 1.57 | 0.264 [0.222, 0.310] | 0.566 | 0.108 |
| A4 marginal | 0.513 | 0.737 | 0.712 [0.604, 0.805] | 0.968 | 1.38 | 0.550 [0.500, 0.599] | 0.379 | 0.288 |
| XGBoost full + conformal | 0.576 [0.499, 0.663] | 0.787 | 0.903 [0.820, 0.956] | 0.919 | 1.54 | 0.303 [0.259, 0.350] | 0.540 | 0.097 |
| XGBoost short + conformal | 0.509 [0.435, 0.596] | 0.743 | 0.903 [0.820, 0.956] | 0.907 | 1.61 | 0.230 [0.190, 0.274] | 0.608 | 0.097 |

- **Instrument.** k* per fold: mean 9.6, range 4-14. The groups selected in at least 60% of folds are
  pre-pregnancy weight (G089, 100%, tier 2, cost 5), maternal age (G003, 96%) and counselling organiser (G065, 68%):
  3 groups, field cost 8 of 209. The cost-aware short form is better than the full questionnaire in point estimate
  (AUPRC 0.513 vs 0.457) and about equal to lasso selection (0.521). Pre-pregnancy weight is a strong proxy for MUAC.
- **XGBoost** on the full questionnaire reaches AUPRC 0.576, higher than any logistic arm, but it is exploratory
  and receives no verdict; its interval overlaps those of A3 and the lasso.

### 9.3 Verdicts (Table 1.1)

| Hypothesis | Anemia | CED risk |
|---|---|---|
| **Primary 1.** Short form non-inferior (<= 20 groups; AUPRC(A3) - AUPRC(A1) > -0.05) | **inconclusive**: diff -0.035, 97.5% CI [-0.082, 0.013] | **supported**: diff +0.055, 97.5% CI [-0.010, 0.123] |
| **Primary 2.** Positive-class coverage not materially below nominal (lower CP bound >= 0.80) | **supported**: 0.934, CP [0.847, 0.980]; below-nominal flag false; negative-class coverage 0.916 | **supported**: 0.899, CP [0.816, 0.954]; flag false; negative-class coverage 0.901 |
| **Secondary 1.** Cost-aware beats median random set (AUPRC difference > 0) | **inconclusive**: +0.025, 95% CI [-0.004, 0.059] | **supported**: +0.172, 95% CI [0.119, 0.223] |
| **Secondary 2.** At least 20% routed to routine follow-up | **not supported**: 0.077, CP [0.053, 0.108] | **supported**: 0.255, CP [0.213, 0.300] |

How to read them:
- **Anemia, Primary 1.** The interval is not entirely above -0.05 (its lower end is -0.082), so non-inferiority is not shown,
  and it is not refuted either. The short form loses 0.035 AUPRC against a full questionnaire that is itself weak.
- **Anemia, Secondary 1.** The cost-aware set is slightly better than random sets of the same size (+0.025) but the
  interval includes 0, so selection did not clearly beat chance.
- **Anemia, Secondary 2.** Not supported in the main analysis or in any anemia sensitivity analysis: the interval lies
  entirely below 0.20.
- **CED, Secondary 2.** Supported in the main analysis; in most sensitivity analyses it is inconclusive because the
  routine share falls to about 0.20-0.23.

### 9.4 Coverage across folds, subgroups and alpha

- **Pooled coverage is on target but single folds vary a lot.** Per-fold positive-class coverage of A4 (25 folds):
  anemia min 0.647, median 1.000, max 1.000, with 28% of folds below 0.90 and 20% below 0.85. CED min 0.727,
  median 0.909, max 1.000, with 40% of folds below 0.90 and 20% below 0.85. This is the small-calibration-set effect
  predicted by Table 3.4 (about 17 and 22 positive calibration women per fold).
- **Class-conditional vs marginal.** One threshold for all women drops positive-class coverage to 0.595
  (anemia, difference -0.339, 95% CI [-0.390, -0.288]) and 0.712 (CED, difference -0.187, 95% CI [-0.232, -0.144]).
- **Tuned threshold.** A3 reaches 0.899 (anemia) and 0.851 (CED) positive-class coverage, below the conformal A4 at
  0.934 and 0.899, and it never abstains. The conformal layer sends 82.7% (anemia) and 57.4% (CED) to a confirmatory test.
- **Alpha grid, A4 (positive coverage / routine share):**

| alpha | Anemia | CED risk |
|---|---|---|
| 0.05 | 1.000 / 0.000 | 0.952 / 0.150 |
| 0.10 | 0.934 / 0.077 | 0.899 / 0.255 |
| 0.15 | 0.899 / 0.128 | 0.851 / 0.363 |
| 0.20 | 0.861 / 0.177 | 0.806 / 0.422 |
| 0.25 | 0.795 / 0.242 | 0.758 / 0.486 |
| 0.30 | 0.734 / 0.311 | 0.714 / 0.532 |

  At alpha = 0.05 anemia routes no one to routine follow-up (infinite threshold), as Table 3.4 predicted. Anemia
  reaches 20% routine only at an alpha of about 0.22, where coverage is about 0.84 (interpolating between alpha 0.20 and 0.25), below the 0.90 target.
- **Subdistrict (A4 positive coverage):** anemia 0.900-0.978 (bayan 0.978, gangga 0.907, kayangan 0.900,
  pemenang 0.958, tanjung 0.943); CED 0.870-0.944 (bayan 0.886, gangga 0.870, kayangan 0.933, pemenang 0.892,
  tanjung 0.944).
- **Income band:** anemia 0.914-1.000 in all bands. **CED coverage is lower in point estimate in two bands:** 0.723 for
  2-3 juta (13 positive) and 0.711 for > 3 juta (9 positive), against 0.960 for < 1 juta and 0.919 for 1-2 juta.
  The cells are small and their 95% CP intervals ([0.415, 0.926] and [0.338, 0.946]) include 0.90, so this is a flag,
  not evidence that the guarantee fails in those bands. Full subgroup and alpha tables with CP intervals: `docs/paper_tables.md`.

### 9.5 Sensitivity analyses

| Analysis | n (pos) | Reported size / cost | P1 | P2 | S1 | S2 |
|---|---|---|---|---|---|---|
| **Anemia** | | | | | | |
| 1 uniform cost | 393 (85) | 0 / 0 | inconclusive | supported | inconclusive | not supported |
| 1 steep cost | 393 (85) | 2 / 2 | inconclusive | supported | supported | not supported |
| 3 leave-one-subdistrict-out | 393 (85) | 5 / 6 | inconclusive | supported | inconclusive | not supported |
| 4 WHO trimester cutoffs | 391 (85) | 1 / 1 | supported | supported | supported | not supported |
| 6 ambiguous pair removed | 392 (85) | 1 / 1 | inconclusive | supported | inconclusive | not supported |
| 7 Hb > 90 days removed | 305 (79) | 1 / 1 | inconclusive | supported | inconclusive | not supported |
| 8 calibration share 35% | 393 (85) | 1 / 1 | inconclusive | supported | inconclusive | not supported |
| **CED risk** | | | | | | |
| 1 uniform cost | 407 (109) | 2 / 7 | supported | supported | supported | inconclusive |
| 1 steep cost | 407 (109) | 2 / 6 | supported | supported | supported | inconclusive |
| 3 leave-one-subdistrict-out | 407 (109) | 4 / 10 | supported | inconclusive | supported | supported |
| 5 cutoff 22 cm | 407 (56) | 3 / 7 | supported | supported | supported | inconclusive |
| 5 cutoff 24 cm | 407 (167) | 2 / 6 | supported | supported | supported | inconclusive |
| 6 ambiguous pair removed | 406 (108) | 3 / 8 | supported | supported | supported | inconclusive |
| 8 calibration share 35% | 407 (109) | 2 / 6 | supported | supported | supported | supported |

- **Primary 2 is the most robust conclusion:** supported in every setting except CED leave-one-subdistrict-out
  (inconclusive; A4 coverage 0.881).
- **CED Primary 1 and Secondary 1 hold in every sensitivity analysis.** CED Secondary 2 is the fragile one: supported
  only in the main analysis, leave-one-subdistrict-out and calibration share 35%; the other settings have routine
  shares of 0.198-0.237, close to the 0.20 line, so the interval straddles it.
- **Post-hoc MUAC check (anemia, `scripts/run_muac_sensitivity.py`):** MUAC predictor values outside 15-45 cm (8 women) set to missing. A3 AUPRC 0.274 (main 0.278), A4 coverage 0.936 (97.5% CP lower bound 0.850), routine 0.073; all four verdicts unchanged. Not part of the preset analyses.
- **Post-hoc CED baselines (`scripts/run_ced_baseline.py`, same folds and D_fit / D_cal splits):** logistic regression on pre-pregnancy weight + age (B2) gives AUPRC 0.556, coverage 0.930, routine share 0.258; adding height (B3) 0.576 / 0.927 / 0.248. Paired AUPRC difference against A4: B2 +0.043 [-0.018, 0.099], B3 +0.063 [0.000, 0.123] (point estimates; the script's printed bootstrap means, 0.040 and 0.060, are slightly lower). The per-fold short form (about 9.6 groups) is not better than 2-3 variables.
- **Anemia conclusions barely change:** Primary 1 is inconclusive in 6 of 7 sensitivity analyses (supported only with
  WHO trimester cutoffs, difference 0.001), Secondary 2 is not supported in all 7.
- **Cost scheme.** Reported short-form size and A3 AUPRC: anemia uniform 0 groups / 0.268, main 1 / 0.278,
  steep 2 / 0.294; CED uniform 2 / 0.521 (cost 7), main 3 / 0.513 (cost 8), steep 2 / 0.507 (cost 6). Jaccard overlap
  with the main short form: anemia 0.00 (uniform) and 0.50 (steep); CED 0.67 for both. For CED the schemes largely
  select the same questions; for anemia the reported form is not stable. Under the uniform scheme no anemia group
  reaches the 60% frequency, so the reported form is empty.
- Leave-one-subdistrict-out for anemia selects 5 groups (Jaccard 0.20 with main), so the anemia short form is not stable.

### 9.6 Additional analyses for the paper claims (post-hoc, `scripts/post_hoc_cost_analysis.py`)

All from the stored per-fold selections of the main run (25 folds), field cost under the main 1/2/5 scheme.

| | Anemia | CED risk |
|---|---|---|
| Mean groups per fold (k*) | 8.8 (range 4-20) | 9.6 (range 4-14) |
| **Mean field cost, cost-aware short form (per fold)** | **14.3** (SD 6.2; of 214) | **18.8** (SD 4.9; of 209) |
| Mean field cost, lasso form of the same size | 15.8 (SD 6.8) | 20.6 (SD 4.8) |
| Lasso minus cost-aware (mean, bootstrap 95% over folds) | +1.5 [0.6, 2.4] | +1.8 [1.2, 2.4] |
| Folds where lasso is costlier / equal / cheaper | 19 / 1 / 5 | 20 / 4 / 1 |
| Folds that hit K_max = 20 | **1 of 25** | 0 of 25 |
| Tier-2 (instrument) groups per fold, cost-aware / lasso | 0.2 / 0.3 | 1.0 / 1.0 |

- **Report the per-fold form, not the 60%-frequency form.** The "short form" evaluated in A3/A4 has about 9 groups and
  field cost 14 (anemia) and 19 (CED), not the 1-group (cost 1) and 3-group (cost 8) forms from the 60% rule.
- **Cost-aware vs lasso.** AUPRC is equal or slightly lower for cost-aware (anemia 0.278 vs 0.297, CED 0.513 vs 0.521),
  and the lasso form costs about 8-10% more (about 1.5 and 1.8 cost units per fold). The advantage is real in sign
  but small, and it comes from cost, not accuracy. The interval treats folds as independent although the 5 repetitions
  reuse the same women, so it is optimistic. The cost term is a weak contribution, not a headline one.
- **K_max.** One anemia fold (4%) selected k* = 20; the AUPRC curve may still have been rising there, but this affects
  one fold, so the 20-group cap does not materially limit the anemia result.
- Subgroup (subdistrict, income) and alpha-grid tables with CP intervals: `docs/paper_tables.md`.

## 10. Conclusions

1. **The conformal layer does what the theory says.** Class-conditional conformal sets keep positive-class coverage at or
   near the nominal 0.90 for both targets (anemia 0.934, CED 0.899), with 97.5% lower bounds above the 0.80
   criterion (Primary 2 supported for both). A single marginal threshold fails this (0.595 and 0.712), and the tuned
   threshold has no guarantee and falls below nominal for CED (0.851).
2. **The price of the guarantee is abstention.** With 17-22 positive calibration women per fold the layer is conservative:
   83% (anemia) and 57% (CED) of women are sent to a confirmatory test. Only 7.7% of anemia women and 25.5% of CED
   women can go to routine follow-up.
3. **CED risk is where the framework pays off.** Per-fold cost-aware short forms (9.6 groups on average, field cost 18.8 of 209)
   are non-inferior to the full questionnaire under logistic regression (AUPRC 0.513 vs 0.457), beats random sets of the same
   size (+0.172), and routes 25.5% of women to routine follow-up. Because the confirmatory step for CED is a cheap tape
   measurement, the operational gain is modest.
4. **Anemia is not predictable enough from this questionnaire.** AUROC is 0.56-0.60 for every model, the short form is
   neither shown non-inferior nor shown better than random, and routine routing fails the 20% criterion in every
   analysis. Anemia still needs the blood test; the short form should not be presented as a screening tool for it.
5. **Caveats that limit these claims.** (a) The reported short forms (1 and 3 groups) summarise selection frequency;
   the evaluated arms use per-fold k* (mean 8.8 and 9.6). (b) Single-fold coverage is volatile (28% and 40% of folds
   below 0.90). (c) CED coverage appears lower for the 2-3 juta and > 3 juta income bands (0.72 and 0.71, small cells).
   (d) The bootstrap treats fold models as fixed and understates refit variability. (e) Anemia selection is not stable
   across cost schemes or held-out subdistricts.

## 11. Limitations (carried from the proposal)

Single site and single wave; whole-number Hb and MUAC, measured on different days from the interview and with
possibly different devices; self-reported answers collected by trained enumerators rather than cadres; leave-one-
subdistrict-out tests only within-regency shift; a short form and thresholds fitted here need a new local
calibration set (at least ~20 confirmed positives for alpha = 0.10) and probably re-selection elsewhere. The study
is retrospective: it does not show that cadres can use the short form or that routine follow-up is safe in practice.
Costs are ordinal assumptions, not measured times. MUAC values outside a plausible range (0, 10, 12, 66, 77) are used
as recorded as an anemia predictor; the main analysis was kept as specified after the results were known. The sample is 393 / 407 rather than the proposal's 392 / 406
because the named dedup rule found 4 duplicate pairs (Section 2).

## 12. Findings to discuss in the paper

1. **WHO trimester sensitivity (analysis 4) does not test alternative cutoffs.** Hb is recorded in whole numbers, so
   Hb < 10.5 and Hb < 11.0 give identical labels: positives stay at 85, and n differs (391 vs 393) only because 2 women
   have no gestational age. Do not claim it tests other cutoffs. Anemia Primary 1 turns "supported" (difference +0.001)
   only in this run, so the anemia Primary 1 verdict is fragile, driven by 2 women.
2. **XGBoost on the full questionnaire beats the short form for CED** (AUPRC 0.576 vs 0.513; routine share 30.3% vs
   25.5%). It is exploratory, with no verdict, but the claim is "the short form matches full-questionnaire *logistic
   regression*", not "matches the best model".
3. **Anemia selection is unstable.** Jaccard against the main short form is 0.00 for the uniform scheme and the Hb > 90
   days analysis, 0.20 for leave-one-subdistrict-out and 0.50 for steep costs. This fits a weak signal and is a finding.
4. **RQ3 is a trade-off, shown side by side (anemia):**

| Anemia | Routine share | Miss among positives |
|---|---|---|
| A3 tuned threshold | 12.8% | 10.1% |
| A4 conformal | 7.7% | 6.6% |

   Conformal is safer but saves fewer tests. For CED: A3 36.3% routine / 14.9% miss vs A4 25.5% / 10.1%.
5. **CED leave-one-subdistrict-out** has positive coverage 0.881 and lower 97.5% bound 0.793 < 0.80, so Primary 2 is
   inconclusive (not supported). This is empirical support that a new area needs local calibration.
6. **Fold-level coverage is expected noise, not a method failure.** Coverage on a test fold is a Beta draw (calibration
   set of n = 17 positives gives Beta(17, 1), mean 0.944; CED n = 22 gives Beta(21, 2), mean 0.913) followed by binomial
   noise from about 17-22 test positives. A simulation of this two-stage model (my own approximation, equal
   calibration and test positive counts) gives a fold-level SD of 0.076 (anemia) and 0.082 (CED), 24% / 30% of folds
   below 0.90 (observed 28% / 40%), and a 21% / 43% chance that the minimum over 25 folds is at or below the observed
   0.647 / 0.727. Folds are not independent across repetitions, so this is a rough guide. A fold at 0.647 is 11 of 17
   positives.

## 13. Remaining work

1. Check that the proposal's `.tex` source carries the corrections (abstract, Sec 1.4, Figure 3.1, Sec 3.2, Sec 3.7,
   Sec 3.9). `docs/proposal_updates.md` is superseded by that source. The `smoke_*` outputs were deleted after the
   code commit.
   When quoting A4 positive-class coverage, always state the interval level: 97.5% CP is the Primary 2 verdict
   interval (anemia [0.847, 0.980], CED [0.816, 0.954]); 95% CP is used in `docs/paper_tables.md`.
2. Paper writing, supervisor review and submission.
