"""make_figure.py — render the Avery 10-level ISMI-29 sanity-check figure
(PDF + PNG) for the paper, in the paper's NeurIPS 2026 figure styling
(single-column width, Times New Roman, no in-figure caption).

Data: 10 gpt-4o-generated stigma levels (anchored on the main simulation's hand-authored
severe/moderate/mild Avery layers at level_9/5/2, with explicit plateau-
guard instructions so intermediate levels don't collapse toward the nearest
anchor), each measured with qwen3:32b ISMI-29, n=30 reps, judge ctx 4096
(prompt is backstory + 29 items only, well under 4096 -- see the main simulation's own
measure_ismi29.py for why that context size is sufficient here).
Spearman rho(level, measured) = 0.99, 1/9 adjacent-pair violation (levels 2-3).
"""
import json
import matplotlib.pyplot as plt

def load(dirpath):
    means, sds = [], []
    for i in range(1, 11):
        d = json.load(open(f"{dirpath}/level_{i}.json"))
        means.append(d["ismi29_total_mean"])
        sds.append(d["ismi29_total_sd"])
    return means, sds

mean, sd = load("outputs/level_ismi29_30reps")

# NeurIPS 2026 fit:
# single-column text width 5.5in, figures at ~0.95\linewidth, Times New
# Roman, 10pt body.
COL_MEASURED = "#2a78d6"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIG_WIDTH_IN = 5.5 * 0.95
fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.62))

levels = list(range(1, 11))
ax.errorbar(levels, mean, yerr=sd, marker="o", markersize=3.2, capsize=2.5,
            color=COL_MEASURED, linewidth=1.3, markerfacecolor=COL_MEASURED,
            markeredgecolor="white", markeredgewidth=0.4,
            label=f"Measured ISMI-29 (mean $\\pm$ SD, n=30)")

ax.set_xlabel("Generated stigma level (1 = lowest, 10 = highest)")
ax.set_ylabel("Measured ISMI-29")
ax.set_xticks(levels)
ax.grid(axis="y", color="#e2e0da", linewidth=0.7, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(loc="lower right", frameon=False)

fig.tight_layout()
fig.savefig("outputs/figures/avery_10level_ismi29.pdf", bbox_inches="tight")
fig.savefig("outputs/figures/avery_10level_ismi29.png", dpi=300, bbox_inches="tight")
print("wrote outputs/figures/avery_10level_ismi29.{pdf,png}")
