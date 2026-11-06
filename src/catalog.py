"""Question catalog (Sec 3.3): groups, tiers, and cost schemes (Table 3.2)."""
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "config" / "question_catalog.csv"

SCHEMES = {
    "uniform": {0: 1, 1: 1, 2: 1},
    "main": {0: 1, 1: 2, 2: 5},
    "steep": {0: 1, 1: 3, 2: 10},
}
TARGETS = ("anemia", "ced")
# CED-type labels share the CED catalog (no MUAC); anemia-type labels the anemia one
CATALOG_OF = {"anemia": "anemia", "anemia_who": "anemia", "ced": "ced", "ced22": "ced", "ced24": "ced"}


@lru_cache(maxsize=None)
def load_catalog():
    return pd.read_csv(CATALOG, keep_default_na=False)


def included(target):
    """Catalog rows of the predictor columns for one target, in col order."""
    cat = load_catalog()
    rows = cat[cat[f"include_{CATALOG_OF[target]}"] == 1]
    return rows.sort_values("col_index").reset_index(drop=True)


def group_ids(target):
    return sorted(included(target).group_id.unique())


def group_info(target):
    """DataFrame indexed by group_id with name, domain, tier."""
    rows = included(target)
    return rows.groupby("group_id").agg(group_name=("group_name", "first"),
                                        domain=("domain", "first"),
                                        tier=("tier", "first"))


def group_costs(target, scheme="main"):
    tiers = group_info(target).tier.astype(int)
    return {g: SCHEMES[scheme][t] for g, t in tiers.items()}


def feature_name(col_index):
    return f"c{int(col_index):03d}"
