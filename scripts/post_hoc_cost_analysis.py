"""Post-hoc: per-fold field cost of cost-aware vs lasso short forms, folds hitting K_max.
Reads results/selection/main_*.json only; writes results/tables/posthoc_cost.csv."""
import json
import numpy as np
import pandas as pd
from src.catalog import group_costs, group_info

rows = []
for tgt in ["anemia", "ced"]:
    d = json.load(open(f"results/selection/main_{tgt}.json"))
    costs = group_costs(tgt, "main")
    folds = d["folds"]
    k = np.array([f["k_star"] for f in folds])
    ca = np.array([sum(costs[g] for g in f["selected"]) for f in folds])
    la = np.array([sum(costs[g] for g in f["lasso_order"][:f["k_star"]]) for f in folds])
    diff = la - ca
    rng = np.random.default_rng(0)
    boot = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
    rows.append(dict(target=tgt, n_folds=len(folds), total_cost=sum(costs.values()),
                     k_mean=k.mean(), k_min=k.min(), k_max=k.max(), n_at_kmax=int((k == 20).sum()),
                     cost_aware_mean=ca.mean(), cost_aware_sd=ca.std(ddof=1),
                     lasso_mean=la.mean(), lasso_sd=la.std(ddof=1),
                     diff_mean=diff.mean(), diff_lo=np.percentile(boot, 2.5), diff_hi=np.percentile(boot, 97.5),
                     folds_lasso_costlier=int((diff > 0).sum()), folds_equal=int((diff == 0).sum()),
                     folds_lasso_cheaper=int((diff < 0).sum()),
                     tier2_in_cost_aware=np.mean([sum(group_info(tgt).tier.astype(int)[g] == 2 for g in f["selected"]) for f in folds]),
                     tier2_in_lasso=np.mean([sum(group_info(tgt).tier.astype(int)[g] == 2 for g in f["lasso_order"][:f["k_star"]]) for f in folds])))
df = pd.DataFrame(rows)
df.to_csv("results/tables/posthoc_cost.csv", index=False)
print(df.T.to_string())
