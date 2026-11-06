"""Entry point: python -m src.run --stage all|data|main|sensitivity|figures|summary

--smoke        1 repetition, K_max=5, 5 random draws, 100 bootstrap resamples,
               outputs prefixed "smoke_" (never used for reporting)
--provisional  use the analysis table built without deduplication (smoke testing only)
"""
import argparse
import json
import platform
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

from src import data, evaluate, figures, sensitivity, summary
from src.experiment import main_settings, n_jobs_default, run

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LIBS = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "pyarrow", "matplotlib",
        "joblib", "pyyaml"]


def log_environment():
    RESULTS.mkdir(exist_ok=True)
    lines = [f"python {platform.python_version()} ({platform.platform()})"]
    lines += [f"{lib} {version(lib)}" for lib in LIBS]
    (RESULTS / "environment.txt").write_text("\n".join(lines) + "\n")


def make_log():
    path = RESULTS / "run_log.txt"
    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "data", "main", "sensitivity", "figures", "summary"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--provisional", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=n_jobs_default())
    args = ap.parse_args()
    if args.provisional and not args.smoke:
        ap.error("--provisional is only allowed with --smoke")
    log_environment()
    log = make_log()
    prefix = "smoke_" if args.smoke else ""
    B = 100 if args.smoke else 2000
    mains = main_settings(provisional=args.provisional)
    if args.smoke:
        mains = [replace(s, name=prefix + s.name, n_reps=1, k_max=5, n_draws=5) for s in mains]
    main = {s.target: s for s in mains}
    t0 = time.time()
    log(f"stage={args.stage} smoke={args.smoke} provisional={args.provisional} n_jobs={args.n_jobs}")

    if args.stage in ("all", "data"):
        _, rep = data.write(provisional=args.provisional)
        log("data: " + json.dumps({k: v for k, v in rep.items() if k not in ("rows",)}))
    if args.stage in ("all", "main"):
        verdicts = {}
        for s in mains:
            arms, meta = run(s, n_jobs=args.n_jobs, log=log)
            t = time.time()
            ev = evaluate.evaluate(arms, meta, B=B)
            ev["verdicts"] = evaluate.verdicts(ev)
            evaluate.save(ev, s.name)
            verdicts[s.target] = ev["verdicts"]
            log(f"evaluate {s.name}: B={B} {time.time() - t:.0f}s")
        (RESULTS / f"{prefix}verdicts.json").write_text(json.dumps(verdicts, indent=1, default=float))
    if args.stage in ("all", "sensitivity"):
        sensitivity.run_all(main, args.n_jobs, B, prefix=prefix, log=log)
    if args.stage in ("all", "figures"):
        figures.all_figures(main, sens_prefix=prefix)
    if args.stage in ("all", "summary"):
        summary.write(main, prefix=prefix)
    log(f"done stage={args.stage} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
