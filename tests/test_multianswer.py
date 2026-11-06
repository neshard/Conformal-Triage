from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.multianswer import normalize, split_answer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "kuesioner_raw.csv"
SPEC = yaml.safe_load((ROOT / "config" / "multi_answer_options.yaml").read_text(encoding="utf-8"))


def test_comma_inside_option_is_kept_whole():
    opts = SPEC["col_65_ifa_nonadherence_reasons"]["options"]
    chosen, left = split_answer("Lupa, Mual, sembelit, atau efek samping lain", opts)
    assert chosen == ["Lupa", "Mual, sembelit, atau efek samping lain"] and left == []


def test_spacing_variants_and_trailing_comma():
    opts = SPEC["col_68_anemia_symptoms_known"]["options"]
    assert split_answer("Letih /lemas", opts) == (["Letih/lemas"], [])
    assert split_answer("Letih/lemas,", opts) == (["Letih/lemas"], [])


def test_unknown_text_is_reported():
    assert split_answer("Pusing, Banyak pikiran", ["Pusing"]) == (["Pusing"], ["banyak pikiran"])


@pytest.mark.skipif(not RAW.exists(), reason="raw data not available")
@pytest.mark.parametrize("key", sorted(SPEC))
def test_every_raw_answer_is_consumed(key):
    col = int(key.split("_")[1])
    df = pd.read_csv(RAW, sep=";", encoding="latin-1", dtype=str, keep_default_na=False)
    allowed = {normalize(t) for t in SPEC[key]["other_text"]}
    leftovers = set()
    for raw in df.iloc[:, col]:
        _, left = split_answer(raw, SPEC[key]["options"])
        leftovers.update(left)
    assert leftovers <= allowed, f"unlisted text in col {col}: {sorted(leftovers - allowed)}"
    assert leftovers == allowed, f"stale other_text in col {col}: {sorted(allowed - leftovers)}"
