"""Draft config/question_catalog.csv from the raw questionnaire header.

Phase 0 helper. Columns are addressed by position because several headers are
duplicated ("Usia", empty trailing headers). After review, the CSV is the
source of truth and this script is kept only as a record of the first draft.
Prints no respondent values.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "kuesioner_raw.csv"
OUT = ROOT / "config" / "question_catalog.csv"

COST = {0: 1, 1: 2, 2: 5}

# --- excluded columns: col index -> reason ---------------------------------
EXCLUDE = {
    0: "identifier/admin: form timestamp (used only for duplicate handling)",
    1: "identifier: enumerator name",
    2: "admin: consent statement",
    3: "duplicate age (admin block); col 14 is authoritative, 12 rows differ",
    4: "measurement date: interview date",
    5: "identifier: respondent code",
    7: "village name", 8: "village name", 9: "village name",
    10: "village name", 11: "village name",
    12: "health center name",
    39: "gravida-para-abortus field (inconsistent format)",
    42: "measurement date: MUAC", 44: "measurement date: weight",
    46: "measurement date: height", 48: "measurement date: hemoglobin",
    47: "label source: hemoglobin",
    63: "empty column",
    92: "open-text field (free-text version of col 93)",
    94: "hemoglobin test ever done (describes label production)",
    95: "open-text follow-up of col 94",
    96: "hypothetical action if Hb shows anemia (Hb-test block; not in catalog)",
    103: "empty column",
    109: "open-text follow-up of col 108",
    158: "GPS coordinate", 159: "GPS coordinate",
    160: "GPS coordinate", 161: "GPS coordinate",
    162: "empty column", 163: "empty column",
    164: "empty column", 165: "empty column",
}

# --- question groups: (domain, tier, [(col, encoding), ...], short name) ----
O, N, H, M = "ordinal", "numeric", "onehot", "multibinary"
G = []
def g(domain, tier, name, *cols):
    G.append((domain, tier, name, list(cols)))

# Tier 0: context and sociodemographics incl. gestational age (19)
g("context", 0, "subdistrict", (6, H))
g("context", 0, "phone number ownership", (13, H))
g("context", 0, "maternal age", (14, N))
g("context", 0, "household size", (15, N))
g("context", 0, "number of children", (16, N))
g("context", 0, "children under five", (17, N))
g("context", 0, "marital status", (18, H))
g("context", 0, "maternal education", (19, O))
g("context", 0, "maternal occupation", (20, H))
g("context", 0, "husband education", (21, O))
g("context", 0, "husband occupation", (22, H))
g("context", 0, "monthly household income", (23, O))
g("context", 0, "household assets", (24, M))
g("context", 0, "usual place of health check", (25, H))
g("context", 0, "distance to health service", (26, H))
g("context", 0, "transport to health service", (27, H))
g("context", 0, "social assistance received", (28, H))
g("context", 0, "health insurance", (29, H))
g("context", 0, "gestational age (weeks)", (40, N))

# Tier 1: HFIAS, each item its own group (9)
for i, c in enumerate(range(30, 39), 1):
    g("hfias", 1, f"HFIAS item {i}", (c, O))

# Tier 1: ANC and health history (7)
g("anc_health", 1, "timing of first ANC", (49, H))
g("anc_health", 1, "place of first ANC", (50, H))
g("anc_health", 1, "reason for ANC place", (51, H))
g("anc_health", 1, "number of ANC visits", (52, N))
g("anc_health", 1, "who reminds about ANC", (53, H))
g("anc_health", 1, "delayed/missed ANC and why", (54, H))
g("anc_health", 1, "chronic disease history", (55, H))

# Tier 1: IFA receipt and adherence (10)
g("ifa", 1, "received IFA this pregnancy", (56, H))
g("ifa", 1, "timing of first IFA receipt", (57, H))
g("ifa", 1, "who gave first IFA", (58, H))
g("ifa", 1, "tablets in first IFA supply", (59, H))
g("ifa", 1, "received additional IFA", (60, H))
g("ifa", 1, "total IFA tablets received", (61, N))
g("ifa", 1, "share of IFA tablets taken", (62, H))
g("ifa", 1, "when IFA is usually taken", (64, H))
g("ifa", 1, "reasons for not taking IFA", (65, M))
g("ifa", 1, "who reminds to take IFA", (66, H))

# Tier 1: knowledge of anemia, IFA, stunting, CED (18)
KNOW = {67: "knows what anemia is", 68: "anemia symptoms known",
        69: "consequence of anemia in pregnancy", 70: "purpose of IFA",
        71: "recommended IFA count", 72: "first source of IFA info",
        73: "best time to start IFA", 74: "best daily time for IFA",
        75: "IFA with tea/coffee/milk", 76: "heard of stunting",
        77: "what stunting is", 78: "when stunting starts",
        79: "main cause of stunting", 80: "how to prevent stunting",
        81: "heard of CED (KEK)", 82: "what CED is", 83: "causes of CED",
        84: "CED vs stunting"}
for c, name in KNOW.items():
    g("knowledge", 1, name, (c, M if c == 68 else H))

# Tier 1: nutrition counselling and food aid (8)
g("counselling_aid", 1, "counselling held in village (6 mo)", (85, H))
g("counselling_aid", 1, "counselling organiser", (86, H))
g("counselling_aid", 1, "counselling attendance frequency", (87, O))
g("counselling_aid", 1, "received food aid", (88, H))
g("counselling_aid", 1, "form of food aid", (89, H))
g("counselling_aid", 1, "food aid frequency", (90, O))
g("counselling_aid", 1, "food aid changed food choice", (91, H))
g("counselling_aid", 1, "IFA programme obstacles", (93, M))

# Tier 1: WASH and indoor smoking (7)
g("wash_smoking", 1, "handwashing with soap", (97, O))
g("wash_smoking", 1, "washing food ingredients", (98, O))
g("wash_smoking", 1, "drinking water source", (99, H))
g("wash_smoking", 1, "clean water source", (100, H))
g("wash_smoking", 1, "water source distance to waste", (101, H))
g("wash_smoking", 1, "toilet location", (102, H))
g("wash_smoking", 1, "indoor smoking in household", (104, H))

# Tier 1: diet pattern and physical activity (10)
g("diet_activity", 1, "eating pattern vs before pregnancy", (105, H))
g("diet_activity", 1, "main meals per day", (106, O))
g("diet_activity", 1, "snacks per day", (107, O))
g("diet_activity", 1, "food taboos in pregnancy", (108, H))
g("diet_activity", 1, "walking days (7 d)", (110, O))
g("diet_activity", 1, "walking duration", (111, H))
g("diet_activity", 1, "household chores frequency", (112, O))
g("diet_activity", 1, "pregnancy exercise", (113, O))
g("diet_activity", 1, "overall physical activity level", (114, H))
g("diet_activity", 1, "yesterday's diet condition", (115, H))

# Tier 1: food frequency, frequency + portion paired by position (21)
FFQ_BLOCKS = [  # (frequency cols, portion cols)
    (range(116, 121), range(121, 126)),   # animal iron
    (range(126, 130), range(130, 134)),   # plant iron
    (range(134, 140), range(140, 146)),   # plant iron + vitamin C
    (range(146, 150), range(150, 154)),   # iron inhibitors
    (range(154, 156), range(156, 158)),   # other
]

# Tier 2: measurements (3)
g("measurement", 2, "pre-pregnancy weight (kg)", (43, N))
g("measurement", 2, "height (cm)", (45, N))
g("measurement", 2, "MUAC (cm)", (41, N))


def main():
    if OUT.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUT} is frozen; pass --force to regenerate the draft")
    hdr =pd.read_csv(RAW, sep=";", encoding="latin-1", dtype=str, nrows=0).columns
    hdr = [h.strip() for h in hdr]
    for fcols, pcols in FFQ_BLOCKS:
        for fc, pc in zip(fcols, pcols):
            item = hdr[fc].rsplit("[", 1)[-1].rstrip("]").strip()
            g("food_frequency", 1, f"FFQ {item}", (fc, O), (pc, O))

    rows, seen = [], set()
    for k, (domain, tier, name, cols) in enumerate(G, 1):
        gid = f"G{k:03d}"
        for c, enc in cols:
            assert c not in seen and c not in EXCLUDE, c
            seen.add(c)
            rows.append(dict(col_index=c, column=hdr[c], group_id=gid,
                             group_name=name, domain=domain, tier=tier,
                             cost=COST[tier], include_anemia=1,
                             include_ced=0 if c == 41 else 1,
                             encoding=enc, exclude_reason=""))
    for c, why in EXCLUDE.items():
        rows.append(dict(col_index=c, column=hdr[c], group_id="", group_name="",
                         domain="excluded", tier="", cost="", include_anemia=0,
                         include_ced=0, encoding="", exclude_reason=why))
    cat = pd.DataFrame(rows).sort_values("col_index")
    assert len(cat) == len(hdr) and cat.col_index.is_unique
    OUT.parent.mkdir(exist_ok=True)
    cat.to_csv(OUT, index=False, encoding="utf-8")

    # --- summary ---
    inc = cat[cat.group_id != ""]
    grp = inc.groupby("group_id").agg(domain=("domain", "first"), tier=("tier", "first"),
                                      cost=("cost", "first"), a=("include_anemia", "max"),
                                      c=("include_ced", "max"), ncols=("column", "size"))
    order = list(dict.fromkeys(grp.domain))
    summ = (grp.groupby("domain", sort=False)
               .agg(tier=("tier", "first"), cost_each=("cost", "first"),
                    groups_anemia=("a", "sum"), groups_ced=("c", "sum"),
                    raw_cols=("ncols", "sum"))
               .reindex(order))
    summ["cost_anemia"] = summ.groups_anemia * summ.cost_each
    summ["cost_ced"] = summ.groups_ced * summ.cost_each
    summ.loc["TOTAL"] = ["", "", summ.groups_anemia.sum(), summ.groups_ced.sum(),
                         summ.raw_cols.sum(), summ.cost_anemia.sum(), summ.cost_ced.sum()]
    print(summ.to_string())
    print(f"\nexcluded columns: {len(EXCLUDE)} | catalog rows: {len(cat)}")


if __name__ == "__main__":
    main()
