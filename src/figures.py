"""Figures for the paper (matplotlib, 300 dpi PNG plus PDF). Every figure has a
CSV twin in results/tables, so values are also available as a table."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.catalog import group_costs  # noqa: E402
from src.evaluate import TABLES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"
SEL = ROOT / "results" / "selection"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # validated categorical slots 1-3 (light)
TARGET_NAME = {"anemia": "Anemia", "ced": "CED risk"}
PREFIX = [""]          # "smoke_" for smoke-test output

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "text.color": INK,
    "legend.frameon": False})


def _save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGS / f"{PREFIX[0]}{name}.png", dpi=300)
    fig.savefig(FIGS / f"{PREFIX[0]}{name}.pdf")
    plt.close(fig)


def _ev(name):
    return json.loads((TABLES / f"{name}_evaluation.json").read_text())


def selection_curves(names):
    """A(S_k) against k and against cumulative field cost, per outer fold and mean."""
    fig, axes = plt.subplots(len(names), 2, figsize=(7, 2.6 * len(names)), squeeze=False)
    for row, (t, name) in enumerate(names.items()):
        meta = json.loads((SEL / f"{name}.json").read_text())
        costs = group_costs(t, meta["setting"]["scheme"])
        A = np.array([f["A"] for f in meta["folds"]])
        cum = np.array([np.cumsum([costs[g] for g in f["path"]]) for f in meta["folds"]])
        k = np.arange(1, A.shape[1] + 1)
        for ax, x, xl in ((axes[row, 0], np.tile(k, (len(A), 1)), "Number of question groups k"),
                          (axes[row, 1], cum, "Cumulative field cost")):
            for xi, ai in zip(x, A):
                ax.plot(xi, ai, color=SERIES[0], alpha=0.15, lw=1)
            ax.plot(x.mean(0), A.mean(0), color=SERIES[0], lw=2, label="Mean of outer folds")
            kstar = np.mean([f["k_star"] for f in meta["folds"]])
            if ax is axes[row, 0]:
                ax.axvline(kstar, color=MUTED, lw=1, ls="--")
                ax.annotate(f"mean k* = {kstar:.1f}", (kstar, ax.get_ylim()[0]), xytext=(4, 6),
                            textcoords="offset points", color=INK2, fontsize=8)
            ax.set_xlabel(xl)
            ax.set_ylabel("Inner AUPRC A(S_k)")
        axes[row, 0].set_title(TARGET_NAME[t], loc="left", fontsize=10, color=INK)
        pd.DataFrame({"k": k, "mean_A": A.mean(0), "mean_cum_cost": cum.mean(0)}).to_csv(
            TABLES / f"{name}_selection_curve.csv", index=False)
    _save(fig, "selection_curve")


def selection_frequency(names, top=25):
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 5.2), squeeze=False)
    for ax, (t, name) in zip(axes[0], names.items()):
        sf = pd.DataFrame(_ev(name)["instrument"]["selection_frequency"])
        sf = sf.sort_values(["frequency", "group_id"], ascending=[False, True]).head(top)[::-1]
        ax.barh([f"{g} {n[:28]}" for g, n in zip(sf.group_id, sf.group_name)], sf.frequency,
                color=SERIES[0], height=0.7)
        ax.axvline(0.60, color=MUTED, lw=1, ls="--")
        ax.text(0.61, -0.9, "60% rule", color=INK2, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of 25 outer folds selecting the group")
        ax.set_title(TARGET_NAME[t], loc="left", fontsize=10)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="y", visible=False)
    _save(fig, "selection_frequency")


def fold_coverage(names):
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    for i, (t, name) in enumerate(names.items()):
        v = np.array(_ev(name)["per_fold"]["values"])
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(v))
        ax.scatter(np.full(len(v), i) + jitter, v, s=22, color=SERIES[0],
                   edgecolor="white", linewidth=1, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=INK, lw=2)
    ax.axhline(0.90, color=MUTED, lw=1, ls="--")
    ax.text(len(names) - 0.5, 0.905, "nominal 0.90", color=INK2, fontsize=7, ha="right")
    ax.set_xticks(range(len(names)), [TARGET_NAME[t] for t in names])
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylabel("Positive-class coverage per outer fold (A4)")
    ax.grid(axis="x", visible=False)
    _save(fig, "fold_coverage")


def triage_shares(names, arms=("A1", "A2", "A3", "A4", "A5", "A6")):
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 2.9), squeeze=False)
    rows = []
    for ax, (t, name) in zip(axes[0], names.items()):
        m = {(r["arm"], r["metric"]): r["estimate"] for r in _ev(name)["metrics"]}
        present = [a for a in arms if (a, "refer") in m]
        left = np.zeros(len(present))
        for c, (key, lab) in enumerate((("refer", "Refer"), ("routine", "Routine follow-up"),
                                        ("confirm", "Confirmatory test"))):
            v = np.array([m[(a, key)] for a in present])
            ax.barh(present, v, left=left, color=SERIES[c], height=0.62, label=lab,
                    edgecolor="white", linewidth=2)
            for y, (l, w) in enumerate(zip(left, v)):
                if w >= 0.08:
                    ax.text(l + w / 2, y, f"{w:.0%}", ha="center", va="center", fontsize=7,
                            color="#0b0b0b" if c == 2 else "white")
            left += v
            rows += [dict(target=t, arm=a, action=key, share=x) for a, x in zip(present, v)]
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of women")
        ax.set_title(TARGET_NAME[t], loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0, -1].legend(loc="upper center", bbox_to_anchor=(0, -0.28), ncol=3, fontsize=7.5)
    pd.DataFrame(rows).to_csv(TABLES / f"{PREFIX[0]}triage_shares.csv", index=False)
    _save(fig, "triage_shares")


def cost_tradeoff(scheme_names):
    """scheme_names: target -> {scheme: setting name}. Reported short-form cost (main
    scheme units) against AUPRC of A3, with the full questionnaire for reference."""
    fig, axes = plt.subplots(1, len(scheme_names), figsize=(3.6 * len(scheme_names), 2.9),
                             squeeze=False)
    rows = []
    for ax, (t, schemes) in zip(axes[0], scheme_names.items()):
        for scheme, name in schemes.items():
            e = _ev(name)
            m = {(r["arm"], r["metric"]): r for r in e["metrics"]}
            x, y = e["instrument"]["reported_cost_main"], m[("A3", "auprc")]["estimate"]
            ax.scatter(x, y, s=40, color=SERIES[0], edgecolor="white", linewidth=1.5, zorder=3)
            ax.annotate(scheme, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7.5,
                        color=INK2)
            rows.append(dict(target=t, scheme=scheme, reported_cost_main=x, auprc_A3=y,
                             reported_size=e["instrument"]["reported_size"]))
            if scheme == "main":
                full = e["instrument"]["full_cost_main"], m[("A1", "auprc")]["estimate"]
        ax.scatter(*full, s=40, marker="s", color=MUTED, zorder=3)
        ax.annotate("full", full, xytext=(-18, 4), textcoords="offset points", fontsize=7.5,
                    color=INK2)
        ax.set_xlabel("Field cost of reported short form (main-scheme units)")
        ax.set_ylabel("AUPRC")
        ax.set_title(TARGET_NAME[t], loc="left", fontsize=10)
    pd.DataFrame(rows).to_csv(TABLES / f"{PREFIX[0]}cost_tradeoff.csv", index=False)
    _save(fig, "cost_tradeoff")


def all_figures(main, sens_prefix=""):
    PREFIX[0] = sens_prefix
    names = {t: s.name for t, s in main.items()}
    selection_curves(names)
    selection_frequency(names)
    fold_coverage(names)
    triage_shares(names)
    schemes = {t: {"uniform": f"{sens_prefix}uniform_{t}", "main": names[t],
                   "steep": f"{sens_prefix}steep_{t}"} for t in names}
    if all((TABLES / f"{n}_evaluation.json").exists() for s in schemes.values() for n in s.values()):
        cost_tradeoff(schemes)
