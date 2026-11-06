"""Write config/dedup_decisions.yaml from the named export (Table 3.1 step 2).

This is the only code that opens data/raw/kuesioner_with_names.csv. It never
prints or writes names: output is synthetic row indices, timestamps, and
differing-item counts only. The pipeline afterwards reads only the raw export
plus the YAML written here.

Rule (Sec 3.2): two records are one respondent when they share normalized name
(lowercase, trimmed, collapsed whitespace), age (col 14), enumerator and
subdistrict, and differ in at most 20 of the 144 response items (cols 14-157).
The earliest record by timestamp is kept.
"""
import re
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import COL, RESPONSE_ITEMS, n_differing, read_raw, timestamps  # noqa: E402

NAMED = ROOT / "data" / "raw" / "kuesioner_with_names.csv"
OUT = ROOT / "config" / "dedup_decisions.yaml"
MAX_DIFF = 20
# The proposal expected 5 pairs and ambiguous pairs of 48 and 59 items; the named rule on the aligned
# export gives 4 pairs and one ambiguous pair of 59 items (decision 2026-11-02: the data count is final).
EXPECTED_PAIRS, EXPECTED_AMBIGUOUS = 4, [59]


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def read_named():
    raw = NAMED.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    first = text.splitlines()[0]
    sep = ";" if first.count(";") > first.count(",") else ","
    df = pd.read_csv(NAMED, sep=sep, encoding=enc, dtype=str, keep_default_na=False)
    return df.apply(lambda s: s.str.strip().str.replace("\x96", "-", regex=False))


def locate(named, raw_header, raw_col, occurrence=0):
    """Position in the named file of the raw column with the same header text."""
    target = raw_header[raw_col]
    hits = [i for i, h in enumerate(named.columns) if h.strip() == target]
    nth = sum(1 for c in range(raw_col) if raw_header[c] == target)
    if len(hits) <= nth:
        sys.exit(f"named file has no column matching raw col {raw_col} ({target!r})")
    return hits[nth]


def main():
    if not NAMED.exists():
        sys.exit(f"{NAMED.relative_to(ROOT)} not found")
    raw = read_raw()
    raw_header = [h.strip() for h in pd.read_csv(ROOT / "data/raw/kuesioner_raw.csv", sep=";",
                                                 encoding="latin-1", nrows=0).columns]
    named = read_named()
    name_cols = [i for i, h in enumerate(named.columns)
                 if re.search(r"\bnama\b", h, re.I) and not re.search(r"enum", h, re.I)]
    if len(name_cols) != 1:
        sys.exit(f"expected one respondent-name column, found {len(name_cols)} candidates "
                 f"(headers: {[named.columns[i] for i in name_cols]})")
    print(f"named file: {named.shape[0]} rows x {named.shape[1]} cols; name column located")

    # ---- 1. align 1:1 on timestamp, age, enumerator, subdistrict
    keys = ["timestamp", "age", "enumerator", "subdistrict"]
    npos = {k: locate(named, raw_header, COL[k]) for k in keys}
    rkey = raw[[COL[k] for k in keys]].apply(lambda r: tuple(norm(v) for v in r), axis=1)
    nkey = named.iloc[:, [npos[k] for k in keys]].apply(lambda r: tuple(norm(v) for v in r), axis=1)
    nidx = {}
    for i, k in nkey.items():
        nidx.setdefault(k, []).append(i)
    bad = [int(r) for r, k in rkey.items() if len(nidx.get(k, [])) != 1]
    if len(named) != len(raw) or bad or rkey.duplicated().any():
        print(f"ALIGNMENT FAILED: raw rows without a unique match: {bad}; "
              f"raw={len(raw)} named={len(named)}; duplicated raw keys: "
              f"{[int(i) for i in rkey[rkey.duplicated(keep=False)].index]}")
        sys.exit(1)
    name = pd.Series({r: norm(named.iat[nidx[k][0], name_cols[0]]) for r, k in rkey.items()})
    print("alignment: 1:1 for all", len(raw), "rows")

    # ---- 2. duplicate rule
    ts = timestamps(raw)
    enum = raw[COL["enumerator"]].map(norm)
    sub = raw[COL["subdistrict"]].map(norm)
    age = raw[COL["age"]]
    same_nae, dup_pairs = [], []
    for i, j in combinations(raw.index, 2):
        if name[i] == name[j] and age[i] == age[j] and enum[i] == enum[j]:
            k = n_differing(raw, i, j)
            same_nae.append((i, j, k, sub[i] == sub[j]))
            if sub[i] == sub[j] and k <= MAX_DIFF:
                dup_pairs.append((i, j, k))
    # clusters of linked duplicates: keep the earliest record of each
    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    for i, j, _ in dup_pairs:
        parent[find(i)] = find(j)
    clusters = {}
    for i, j, _ in dup_pairs:
        for r in (i, j):
            clusters.setdefault(find(r), set()).add(r)
    drop = sorted(r for c in clusters.values() for r in sorted(c, key=lambda r: ts[r])[1:])
    ambiguous = [(i, j, k) for i, j, k, _ in same_nae if k > MAX_DIFF]

    def show(i, j, k):
        return f"rows {i:>3}-{j:<3} | {ts[i]} | {ts[j]} | differing items {k}"

    print(f"\npairs within {MAX_DIFF} items: {len(dup_pairs)} (proposal: {EXPECTED_PAIRS})")
    for p in dup_pairs:
        print("  ", show(*p))
    print(f"records dropped (earliest kept): {drop} -> {len(raw) - len(drop)} records")
    print(f"\nsame name, age, enumerator but > {MAX_DIFF} items: {len(ambiguous)}")
    for i, j, k in ambiguous:
        print("  ", show(i, j, k), "| same subdistrict:", sub[i] == sub[j])
    namefree = [(21, 344), (30, 37), (66, 147), (109, 288), (190, 194), (190, 215), (194, 215)]
    print("\nname-free candidate pairs under the named rule:")
    for i, j in namefree:
        print(f"   rows {i}-{j}: same name {name[i] == name[j]} -> "
              f"{'duplicate' if (i, j, n_differing(raw, i, j)) in dup_pairs else 'not a duplicate'}")

    ok = (len(dup_pairs) == EXPECTED_PAIRS and len(drop) == EXPECTED_PAIRS
          and sorted(k for *_, k in ambiguous) == EXPECTED_AMBIGUOUS)
    OUT.write_text(yaml.safe_dump(dict(
        rule=dict(match=["normalized name (lowercase, trimmed, collapsed whitespace)",
                         "age (col 14)", "enumerator", "subdistrict"],
                  max_differing_items=MAX_DIFF, items="cols 14-157 (144)",
                  keep="earliest by form timestamp"),
        drop=[int(r) for r in drop],
        duplicate_pairs=[dict(rows=[int(i), int(j)], differing_items=int(k)) for i, j, k in dup_pairs],
        ambiguous_pairs=[[int(i), int(j)] for i, j, _ in ambiguous],
        ambiguous_differing_items=[int(k) for *_, k in ambiguous],
        ambiguous_use="kept in the main analysis; later record of each pair dropped in sensitivity analysis 6",
        matches_proposal=bool(ok),
        source="aligned named export; names are not stored"), sort_keys=False))
    print(f"\nwrote {OUT.relative_to(ROOT)}; matches proposal: {ok}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
