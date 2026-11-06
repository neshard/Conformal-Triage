"""Section 3.2 and Table 3.1: deduplication, cleaning, labels, de-identified table.

The pipeline reads only data/raw/kuesioner_raw.csv plus config/dedup_decisions.yaml.
The named export is used once, by scripts/build_dedup_decisions.py, to write that YAML.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.catalog import feature_name, load_catalog
from src.multianswer import normalize, split_answer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "kuesioner_raw.csv"
DEDUP = ROOT / "config" / "dedup_decisions.yaml"
PROCESSED = ROOT / "data" / "processed" / "analysis.csv"
REPORT = ROOT / "data" / "processed" / "cleaning_report.json"
PROVISIONAL = ROOT / "results" / "cache" / "PROVISIONAL_analysis_no_dedup.csv"

COL = dict(timestamp=0, enumerator=1, interview_date=4, subdistrict=6, age=14,
           gest=40, muac=41, hb=47, hb_date=48)
RESPONSE_ITEMS = list(range(14, 158))      # the 144 items compared between duplicates

# Final counts after the named dedup step (2026-11-02). The proposal specified 5 duplicates and
# 415 / 399 / 392 / 406 (108 CED positives); the named rule on the aligned export finds 4.
EXPECTED = dict(raw=420, duplicates_removed=4, after_dedup=416, hb_zero=16,
                hb_measured=400, hb_above_20=7, anemia_n=393, muac_out=9,
                ced_n=407, anemia_pos=85, ced_pos=109)

WHO_CUTOFF = {1: 11.0, 2: 10.5, 3: 11.0}  # g/dL by trimester (Sec 2.3)


class CountMismatch(RuntimeError):
    pass


# ---------------------------------------------------------------- parsing
def read_raw():
    df = pd.read_csv(RAW, sep=";", encoding="latin-1", dtype=str, keep_default_na=False)
    df.columns = range(df.shape[1])
    df = df.apply(lambda s: s.str.strip().str.replace("\x96", "-", regex=False))
    df.index.name = "row_id"                  # synthetic: 0-based position in the raw file
    return df


def to_number(s):
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s.where(s.str.fullmatch(r"-?\d+(\.\d+)?")), errors="coerce")


def parse_gest_weeks(s):
    """Weeks as a number. "24 minggu" -> 24; "8bln (32minggu)" -> 32; months only -> NaN."""
    num = to_number(s)
    weeks = pd.to_numeric(s.str.extract(r"(\d+)\s*minggu", flags=re.I)[0], errors="coerce")
    return num.fillna(weeks)


def parse_date(text):
    """Dates are DD/MM/YYYY; entries with a 2-digit year (25, 0025) are MM/DD/YY.
    Other years (typos such as 2015, 0225, 1015) are left unparsed."""
    try:
        a, b, y = (int(p) for p in text.split("/"))
    except ValueError:
        return pd.NaT
    if y in (2024, 2025):
        day, month = a, b
    elif y in (24, 25):
        y, month, day = 2000 + y, a, b
    else:
        return pd.NaT
    if month > 12 and day <= 12:
        day, month = month, day
    try:
        return pd.Timestamp(year=y, month=month, day=day)
    except ValueError:
        return pd.NaT


def timestamps(df):
    return pd.to_datetime(df[COL["timestamp"]], format="%m/%d/%Y %H:%M:%S")


# ---------------------------------------------------------------- dedup
def load_dedup():
    if not DEDUP.exists():
        raise FileNotFoundError(
            f"{DEDUP.relative_to(ROOT)} is missing. Run scripts/build_dedup_decisions.py "
            "with the named export first (Table 3.1 step 2).")
    d = yaml.safe_load(DEDUP.read_text(encoding="utf-8"))
    return [int(r) for r in d["drop"]], [tuple(map(int, p)) for p in d["ambiguous_pairs"]]


def n_differing(df, i, j):
    """Items that differ; blanks equal blanks, case and surrounding/repeated whitespace ignored."""
    def clean(v):
        return re.sub(r"\s+", " ", str(v).strip().lower())
    a, b = df.loc[i, RESPONSE_ITEMS].values, df.loc[j, RESPONSE_ITEMS].values
    return sum(clean(x) != clean(y) for x, y in zip(a, b))


# ---------------------------------------------------------------- features
def _load_yaml(name):
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))


def _map_category(values, rules):
    out = []
    for v in values:
        v = normalize(v)
        cat = next((r["category"] for r in rules if re.search(r["pattern"], v)), None)
        if cat is None:
            raise ValueError(f"no category rule matches a value: {v!r}")
        out.append(np.nan if cat == "__missing__" else cat)
    return out


def encode_features(df):
    """Deterministic parsing of every catalog column (no fitted state).
    Fold-fitted steps (plausibility, imputation, one-hot, scaling) live in preprocess.py."""
    cat = load_catalog()
    cat = cat[cat.group_id != ""]
    cmaps = {int(k.split("_")[1]): v for k, v in _load_yaml("category_maps.yaml").items()}
    multi = {int(k.split("_")[1]): v for k, v in _load_yaml("multi_answer_options.yaml").items()}
    ordl = _load_yaml("ordinal_levels.yaml")
    missing = {normalize(m) for m in ordl["missing"]}
    out = {}
    for _, r in cat.iterrows():
        c, name, enc, s = int(r.col_index), feature_name(r.col_index), r.encoding, df[int(r.col_index)]
        if enc == "numeric":
            out[name] = parse_gest_weeks(s) if c == COL["gest"] else to_number(s)
        elif enc == "ordinal":
            levels = {normalize(v): k for k, v in enumerate(ordl["scales"][ordl["columns"][c]])}
            vals = s.map(normalize)
            bad = set(vals) - set(levels) - missing
            if bad:
                raise ValueError(f"col {c}: unmatched ordinal values {sorted(bad)}")
            out[name] = vals.map(levels).astype(float)
        elif enc == "onehot":
            if c in cmaps:
                out[name] = _map_category(s, cmaps[c])
            else:
                out[name] = s.map(normalize).replace("", np.nan)
        elif enc == "multibinary":
            opts = multi[c]["options"]
            chosen = [set(split_answer(v, opts)[0]) for v in s]  # free-text leftovers ignored
            for k, o in enumerate(opts):
                out[f"{name}__{k}"] = [float(o in ch) for ch in chosen]
        else:
            raise ValueError(f"unknown encoding {enc!r} for col {c}")
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------- build
def build(provisional=False):
    """Return (analysis frame, report). Raises CountMismatch unless provisional."""
    raw = read_raw()
    rep = {"raw": len(raw)}
    if provisional:
        drop, amb = [], []
    else:
        drop, amb = load_dedup()
    rep["duplicates_removed"] = len(drop)
    d = raw.drop(index=drop)
    rep["after_dedup"] = len(d)

    hb = to_number(d[COL["hb"]])
    hb_zero = hb == 0
    hb = hb.mask(hb_zero)
    hb_high = hb > 20
    muac = to_number(d[COL["muac"]])
    muac_out = ~muac.between(15, 45)
    in_anemia = hb.notna() & ~hb_high
    in_ced = ~muac_out
    rep.update(hb_zero=int(hb_zero.sum()), hb_measured=int(hb.notna().sum()),
               hb_above_20=int(hb_high.sum()), anemia_n=int(in_anemia.sum()),
               muac_out=int(muac_out.sum()), ced_n=int(in_ced.sum()))
    rows = {"hb_zero": hb_zero, "hb_above_20": hb_high, "muac_out": muac_out}
    rep["rows"] = {k: [int(i) for i in v[v].index] for k, v in rows.items()}
    rep["rows"]["hb_not_parsed"] = [int(i) for i in hb[hb.isna() & ~hb_zero].index]

    gest = parse_gest_weeks(d[COL["gest"]])
    gest = gest.where(gest.between(1, 42))
    tri = pd.cut(gest, [0, 13.999, 27.999, 42], labels=[1, 2, 3]).astype(float)
    hb_date = d[COL["hb_date"]].map(parse_date)
    interview = d[COL["interview_date"]].map(parse_date)
    days = (hb_date - interview).dt.days.abs()

    ts = timestamps(d)
    amb_later = pd.Series(False, index=d.index)
    for i, j in amb:
        amb_later[i if ts[i] > ts[j] else j] = True

    a = pd.DataFrame(index=d.index)
    a["in_anemia"] = in_anemia.astype(int)
    a["in_ced"] = in_ced.astype(int)
    a["anemia"] = (hb < 11.0).astype(float).where(in_anemia)
    a["anemia_who"] = (hb < tri.map(WHO_CUTOFF)).astype(float).where(in_anemia & tri.notna())
    a["ced"] = (muac <= 23).astype(float).where(in_ced)
    a["ced22"] = (muac <= 22).astype(float).where(in_ced)
    a["ced24"] = (muac <= 24).astype(float).where(in_ced)
    a["trimester"] = tri
    a["hb_interview_days"] = days
    a["ambiguous_dup_later"] = amb_later.astype(int)
    a = a.join(encode_features(d))

    rep.update(anemia_pos=int(a.anemia.sum()), ced_pos=int(a.ced.sum()),
               anemia_who_missing=int((in_anemia & tri.isna()).sum()),
               anemia_who_pos=int(a.anemia_who.sum()),
               ced22_pos=int(a.ced22.sum()), ced24_pos=int(a.ced24.sum()),
               hb_days_unparsed=int((in_anemia & days.isna()).sum()),
               hb_days_over_90=int((in_anemia & (days > 90)).sum()),
               ambiguous_pairs=[list(p) for p in amb], provisional=provisional)
    rep["mismatches"] = {k: {"expected": v, "observed": rep[k]}
                         for k, v in EXPECTED.items() if rep[k] != v}
    if rep["mismatches"] and not provisional:
        raise CountMismatch(json.dumps(rep["mismatches"], indent=1) +
                            "\nrow-level reasons: " + json.dumps(rep["rows"]))
    return a, rep


def write(provisional=False):
    a, rep = build(provisional)
    path = PROVISIONAL if provisional else PROCESSED
    path.parent.mkdir(parents=True, exist_ok=True)
    a.to_csv(path)
    (path.with_name(path.stem + "_report.json") if provisional else REPORT).write_text(
        json.dumps(rep, indent=1))
    return a, rep


def load(provisional=False):
    path = PROVISIONAL if provisional else PROCESSED
    return pd.read_csv(path, index_col="row_id")


if __name__ == "__main__":
    import sys
    a, rep = write(provisional="--provisional" in sys.argv)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))
