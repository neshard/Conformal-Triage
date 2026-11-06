"""Post-hoc sensitivity: anemia model with implausible MUAC predictor values (outside 15-45 cm)
set to missing. Added after the main results were known; the main analysis is unchanged.
Writes results/tables/muac45_anemia_*.  Run: python -m scripts.run_muac_sensitivity"""
import json
from dataclasses import replace

from src import evaluate as ev_mod
from src.experiment import main_settings, n_jobs_default, run
from src.run import make_log

log = make_log()
main = {s.target: s for s in main_settings()}
s = replace(main["anemia"], name="muac45_anemia", extras=False, muac_clean=True)
arms, meta = run(s, n_jobs=n_jobs_default(), log=log)
e = ev_mod.evaluate(arms, meta, B=2000)
e["verdicts"] = ev_mod.verdicts(e)
ev_mod.save(e, s.name)
print(json.dumps(e["verdicts"], indent=1, default=float))
