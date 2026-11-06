import numpy as np
import pytest

from src import selection
from src.selection import one_se_rule


def test_one_se_rule():
    A = [0.30, 0.40, 0.45, 0.47, 0.46]
    SE = [0.02, 0.02, 0.02, 0.03, 0.02]
    assert one_se_rule(A, SE) == 3          # best k=4 (0.47); 0.45 >= 0.47 - 0.03
    assert one_se_rule([0.5, 0.4], [0.0, 0.0]) == 1


def test_greedy_follows_eq_3_1_with_lowest_id_ties(monkeypatch):
    """Fake A(S) so the expected path is known: value per group, cost from the scheme."""
    value = {"G1": 0.10, "G2": 0.10, "G3": 0.12, "G4": 0.05}
    costs = {"G1": 1, "G2": 1, "G3": 2, "G4": 1}
    monkeypatch.setattr(selection, "group_ids", lambda t: sorted(value))
    monkeypatch.setattr(selection, "inner_folds", lambda *a, **k: ["fold"])
    monkeypatch.setattr(selection, "fold_scores",
                        lambda folds, S: [0.2 + sum(value[g] for g in S)] * 5)
    import pandas as pd
    X = pd.DataFrame({"c": range(10)})
    y = np.r_[np.ones(2), np.zeros(8)]
    out = selection.greedy_select(X, y, "anemia", costs, k_max=4, seed=0)
    # gains per cost: G1 .10, G2 .10 (tie -> G1), G3 .06, G4 .05 -> G1, G2, G3, G4
    assert out["path"] == ["G1", "G2", "G3", "G4"]
    assert out["A0"] == pytest.approx(0.2)
    assert out["SE"] == [0.0] * 4
    assert out["k_star"] == 4                  # zero SE: the best k is chosen
