# Conformal Triage

Code for "Fewer Questions, Guaranteed Coverage: Conformal Triage for Maternal Anemia and
Chronic Energy Deficiency Screening in North Lombok, Indonesia". The specification is
`docs/proposal.pdf`; repository structure and data decisions are in `PLAN.md`.

## Setup

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows; use .venv/bin/python elsewhere
```

Place the raw export at `data/raw/kuesioner_raw.csv` (git-ignored).

## Deduplication (once)

The raw export has no respondent names. Duplicates (Table 3.1 step 2) are resolved once from the
named export, which never leaves `data/raw/`:

```
python scripts/build_dedup_decisions.py      # reads data/raw/kuesioner_with_names.csv
```

It writes only `config/dedup_decisions.yaml` (row indices, no names). Everything else reads the
raw export plus that file.

## Run

```
python -m src.run --stage all            # data, main experiment, sensitivity, figures, summary
python -m src.run --stage main           # data must be built first (--stage data)
python -m src.run --stage sensitivity    # after main
python -m src.run --stage all --smoke    # 1 repetition, K_max=5, 5 draws, 100 bootstrap resamples
python -m pytest tests
```

Outputs: `results/predictions/` (every out-of-fold prediction), `results/selection/` (selected
groups and k* per fold), `results/tables/`, `results/figures/`, `results/verdicts.json`,
`results/SUMMARY.md`, `results/environment.txt`, `results/run_log.txt`. Interrupted runs resume
from `results/cache/`.
