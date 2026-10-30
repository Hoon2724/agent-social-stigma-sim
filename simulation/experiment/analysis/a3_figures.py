"""Figure 2 (overall), Figure 3 (by severity) and Figure 8 (persona x severity grid) from the released runs
(Peer solid red-orange, General dashed blue, +-1 SEM bars, ISMI severity bands shaded). Session 0 = pooled
pre, as in the tables.

  python3 a3_figures.py [--data final]  ->  figures_<data>/figure_ismi_{overall,severity}.pdf,
                                                  appendix_ismi_persona_5x3.pdf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = {"final": HERE.parent / "results/ismi_long.csv"}
SEV = ["mild", "moderate", "severe"]
PERSONAS = ["Avery", "Jordan", "Riley", "Ravi", "Noor"]
PEER, GEN = "#e8664a", "#2f6fd6"
BANDS = [(1.0, 2.0, "#d9ead3", "Minimal"), (2.0, 2.5, "#fdf2cc", "Mild"),
         (2.5, 3.0, "#fbe0bd", "Moderate"), (3.0, 4.0, "#f5cccc", "Severe")]
XL = ["pre"] + [f"s{i}" for i in range(1, 9)]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
# The appendix grid (Figure 8) is drawn at the width it is printed at (\textwidth = 5.5 in times the
# \includegraphics width), so these font sizes are the printed sizes. Figures 2-3 keep their original size.
TEXTWIDTH = 5.5
PRINTED = {"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
           "legend.fontsize": 8, "axes.linewidth": 0.8}


def load(path):
    d = pd.read_csv(path)
    pre = d[d.session == 0].groupby(["persona", "severity"])["ismi"].mean().rename("pre_pooled")
    d = d.join(pre, on=["persona", "severity"])
    d.loc[d.session == 0, "ismi"] = d.loc[d.session == 0, "pre_pooled"]
    return d


def bands(ax, ylo, yhi, labels=False):
    for lo, hi, c, name in BANDS:
        if hi <= ylo or lo >= yhi:
            continue
        ax.axhspan(max(lo, ylo), min(hi, yhi), color=c, zorder=0, lw=0)
        if labels:
            ax.text(-0.2, min(hi, yhi) - 0.03, name, ha="left", va="top", fontsize=7.5, fontweight="bold", color="#666")
    ax.set_ylim(ylo, yhi)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))


def curve(ax, d, cond, err=True, ms=4, lw=1.8):
    traj = d[d.condition == cond].groupby(["trajectory_id", "session"])["ismi"].mean().unstack()
    mean, sem = traj.mean(), traj.std(ddof=1) / np.sqrt(len(traj))
    kw = dict(color=PEER, ls="-", marker="o") if cond == "peer" else dict(color=GEN, ls=(0, (4, 2)), marker="s")
    ax.errorbar(range(9), mean.values, yerr=sem.values if err else None, capsize=2.5 if err else 0, lw=lw, ms=ms,
                elinewidth=0.9, label="Peer" if cond == "peer" else "Neutral", zorder=3, **kw)


def overall(d, out):
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    bands(ax, 1.8, 3.5, labels=True)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))  # auto ticks here are 0.25 apart, which "%.1f" mislabels
    curve(ax, d, "peer"); curve(ax, d, "general")
    ax.set_xticks(range(9)); ax.set_xticklabels(XL); ax.set_xlim(-0.3, 8.3)
    ax.set_xlabel("Session"); ax.set_ylabel("ISMI mean")
    ax.legend(loc="lower right", ncol=2, frameon=True, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figure_ismi_overall.pdf"); plt.close(fig)


def by_severity(d, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, s in zip(axes, SEV):
        bands(ax, 1.8, 3.85)
        sub = d[d.severity == s]
        curve(ax, sub, "peer"); curve(ax, sub, "general")
        ax.set_title(s.capitalize(), fontweight="bold", fontsize=10)
        ax.set_xticks([0, 2, 4, 6, 8]); ax.set_xticklabels(["pre", "s2", "s4", "s6", "s8"]); ax.set_xlim(-0.3, 8.3)
    axes[0].set_ylabel("ISMI mean"); axes[1].set_xlabel("Session")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out / "figure_ismi_severity.pdf"); plt.close(fig)


def grid(d, out):
    fig, axes = plt.subplots(5, 3, figsize=(0.84 * TEXTWIDTH, 0.84 * TEXTWIDTH * 8.4 / 6.2), sharex=True, sharey=True)
    for i, p in enumerate(PERSONAS):
        for j, s in enumerate(SEV):
            ax = axes[i, j]
            bands(ax, 1.8, 3.85)
            sub = d[(d.persona == p) & (d.severity == s)]
            curve(ax, sub, "peer", err=False, ms=2.8, lw=1.5); curve(ax, sub, "general", err=False, ms=2.8, lw=1.5)
            if i == 0:
                ax.set_title(s.capitalize(), fontweight="bold", fontsize=9)
            if j == 0:
                ax.set_ylabel(p, fontweight="bold")
            ax.set_xticks([0, 2, 4, 6, 8]); ax.set_xticklabels(["pre", "s2", "s4", "s6", "s8"])
    fig.supxlabel("Session", fontsize=8.5, y=0.035); fig.supylabel("ISMI mean", fontsize=8.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0.02, 0.045, 1, 1)); fig.savefig(out / "appendix_ismi_persona_5x3.pdf"); plt.close(fig)


ap = argparse.ArgumentParser()
ap.add_argument("--data", choices=SRC, default="final")
a = ap.parse_args()
d = load(SRC[a.data])
out = HERE / f"figures_{a.data}"
out.mkdir(exist_ok=True)
overall(d, out); by_severity(d, out)
with plt.rc_context(PRINTED):
    grid(d, out)
print(sorted(p.name for p in out.iterdir()))
