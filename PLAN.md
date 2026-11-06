# PLAN: Conformal Triage pipeline

Specification: `docs/proposal.pdf`. Section numbers below refer to it. Nothing
here adds a method beyond the proposal. Open questions are listed at the end
and must be resolved before the stage that needs them.

## Repository layout

```
data/raw/kuesioner_raw.csv        raw export (git-ignored; latin-1, ';' separated, 420 x 166)
data/processed/                   de-identified analysis tables (no names, codes, GPS, villages)
docs/proposal.pdf                 specification
config/question_catalog.csv       one row per raw column: group, domain, tier, cost, encoding, exclusion (frozen)
config/category_maps.yaml         free-text merges (frozen)
config/ordinal_levels.yaml        ordinal level orders
config/multi_answer_options.yaml  options of multi-answer questions
config/dedup_decisions.yaml       rows to drop + ambiguous pairs (written once from the named export)
scripts/draft_catalog.py          Phase 0 record of how the catalog draft was built
scripts/build_dedup_decisions.py  only code that opens the named export; prints/writes no names
src/
  catalog.py       groups, tiers, cost schemes (uniform 1/1/1, main 1/2/5, steep 1/3/10)
  data.py          Sec 3.2 + Table 3.1 with count assertions; labels incl. sensitivity variants;
                   deterministic parsing; writes data/processed/analysis.csv (synthetic row_id)
  multianswer.py   split multi-answer responses against the option list
  preprocess.py    Sec 3.4 transformer: leakage assertion, plausibility, imputation, one-hot, scaling;
                   returns only the columns of requested groups
  selection.py     Algorithm 3.1 + one-SE rule, resumable cache; lasso group ranking
  models.py        tuned LR (main), XGBoost (exploratory)
  conformal.py     Eq 2.1-2.2, triage (Table 3.5), marginal variant, tuned threshold, Table 3.4
  experiment.py    outer CV, D_fit/D_cal, arms A1-A6 + no-verdict comparisons, stored predictions
  evaluate.py      Table 3.7 metrics, Eq 3.2, paired bootstrap, Clopper-Pearson, Table 1.1 verdicts
  sensitivity.py   the eight analyses of Sec 3.9 plus subgroups by subdistrict and income
  figures.py, summary.py, run.py (entry point: python -m src.run --stage ...)
tests/             conformal brute force + Table 3.4, leakage, preprocessing slices, weighted metrics,
                   selection rule, multi-answer consumption
results/
  cache/           per-fold results and selection paths for resuming (git-ignored)
  predictions/, selection/, tables/, figures/, verdicts.json, SUMMARY.md, environment.txt, run_log.txt
```

## Run order

0. **Catalog (this phase).** Draft `config/question_catalog.csv`, review it, freeze costs. No model is fitted before this.
1. **Clean and label.** `clean.py` reproduces Table 3.1 exactly (420 -> 415 -> 399 -> 392 anemia; 415 -> 406 CED).
   The run stops if the counts or label prevalences (85/392, 108/406) differ. Writes the de-identified table to `data/processed/`.
2. **Splits.** Fixed outer folds (5 reps x 5 folds) and D_fit/D_cal splits per target, saved once and reused by every arm.
3. **Selection.** Greedy selection per outer fold and per cost scheme, cached. Also the lasso comparator.
4. **Fit and calibrate.** Tune and fit LR on D_fit; score D_cal; class-conditional and marginal thresholds; tuned threshold.
5. **Arms A1-A6.** Out-of-fold predictions, sets, and triage actions for every woman in every repetition.
6. **Inference and verdicts.** Point estimates, bootstrap and Clopper-Pearson intervals, Table 1.1 verdicts.
7. **Sensitivity and subgroups.** Cost schemes, alpha 0.05-0.30, leave-one-subdistrict-out, trimester Hb cutoffs,
   CED cutoffs 22/24 cm, ambiguous duplicate pairs removed, Hb > 90 days from interview removed, calibration share 35%.
8. **XGBoost (exploratory).** Short form and full questionnaire, reported without verdict.
9. **Report.** Tables and figures for the paper; `results/environment.txt` with library versions.

Every stage reads `config/` and writes to `results/`; stages 3-8 reuse cached folds so all arms share the same women.

## Privacy

Names, phone values, respondent codes, enumerator names, villages, health center names and GPS are never printed or
written outside `data/raw/`. Only de-identified tables are stored in the repository (Sec 3.11).

## Data decisions

Row indices are synthetic: the 0-based position of the record in `data/raw/kuesioner_raw.csv`.

- **Catalog frozen (2026-11-01).** `config/question_catalog.csv`: anemia 112 groups / cost 214, CED 111 groups /
  cost 209 (MUAC anemia only). SHA-256 at freeze:
  `afcf10e0f6c394db08678c4ca2535909235b0a90672c3a54490e513b7ca40002`. Costs do not change after this point.
- **Age.** Col 14 is the authoritative age; col 3 (admin-block copy) is excluded. They differ in 12 rows:
  1, 59, 184, 191, 220, 240, 285, 350, 351, 373, 379, 418. Row 240 has col 14 = 3, which the plausibility rule
  (Sec 3.4, age outside 15-50 -> missing) turns into a missing value, with no fallback to col 3 (approved; to be
  stated in `results/SUMMARY.md`). The others are used as recorded.
- **Multi-answer items** (cols 24, 65, 68, 93) are split against the option lists in
  `config/multi_answer_options.yaml`, not on commas (some options contain commas). `tests/test_multianswer.py`
  checks that every raw answer is fully consumed. Only col 68 leaves free text ("other" answers, 5 records):
  gampang lelah, banyak pikiran, mual, kurang tidur, susah tidur. Decision (2026-11-02): they get no column.
- **Phone ownership (col 13).** Kept in tier 0, merged into self / family / health_worker / none. Some raw answers name
  a specific midwife; the mapping uses patterns only, and no raw value of this column is ever written outside
  `data/raw/`.
- **Counselling and food aid.** Col 93 (IFA programme obstacles) is the 8th group. Col 96 stays excluded; col 92 is
  excluded as the open-text version of col 93.
- **Free-text merges.** Occupations (cols 20, 22), chronic disease as none vs any (col 55), counselling organiser
  (col 86), food aid form (col 89) and phone ownership (col 13) follow `config/category_maps.yaml`
  (SHA-256 `84cc9918fb7026e269f1a402f6975f07cb19b4c98c8d2de7dad296ceb0ce29dc`). Every raw answer matches a rule.
- **Encoding.** Ordinal for frequency items, education and income; food portions ordinal with "Tidak tahu" as
  missing; other categorical items one-hot; multi-answer items one binary column per answer.
- **Fish pairing.** FFQ frequency "Ikan laut/pindang/teri" is paired with portion "Ikan (kembung/tongkol)" by position.
- **Duplicates: BLOCKED.** The export has no respondent name column. The name-free rule (same col 14 age, enumerator
  and subdistrict, at most 20 of 144 differing response items; items = cols 14-157) finds **7 pairs, not 5**. Three
  of them form one triple (rows 190, 194, 215). Removing duplicates would give 414 records, not 415. The 48- and
  59-item pairs cannot be told apart from 12+ other same-age, same-enumerator pairs at those distances. Nothing has
  been removed. Waiting for a data version with names for the dedup step (Table 3.1 step 2).
  Planned flow once it arrives: align it 1:1 to the raw file (timestamp, col 14 age, enumerator, subdistrict), apply
  the named rule, and write only `config/dedup_decisions.yaml` (rows to drop, ambiguous pairs, rule). The pipeline
  then reads `kuesioner_raw.csv` plus that YAML and never the named file.

## Open questions

- Dedup step: `data/raw/kuesioner_with_names.csv` was not found on 2026-11-01 or 2026-11-02.
  Phases 2-9 are implemented and smoke-tested on a provisional table built without
  deduplication (`results/cache/`, never reported). Without dedup the other Table 3.1 steps
  already match (Hb = 0: 16, Hb > 20: 7, MUAC outside 15-45: 9; anemia positives 85); the
  remaining differences (420/404/397/411, CED positives 111) are exactly the 5 missing removals.
