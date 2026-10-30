"""Figure 6: belief state vs accumulated dialogue on the evaluation set, drawn from
../results/summaries/a4_validity.json. Change from session 0 by trajectory type (target change, belief state
under the final update, accumulated dialogue); top row the development split, bottom row the sealed test split.

  python3 a4_figure.py  ->  figures_final/validation_sc2.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
D = json.loads((HERE.parent / "results/summaries/a4_validity.json").read_text())
FINAL, CUM, TGT = "#2f6fd6", "#e8664a", "#222222"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 8})
x = np.arange(9)
fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.9), sharey=True)
fig.subplots_adjust(hspace=0.75, wspace=0.12)
order = [("stigma_reinforcing", "Worsening (+0.5)"), ("stigma_reducing", "Reducing (−0.5)"),
         ("neutral_dry", "Flat, dry (0)"), ("general_like", "Flat, empathic (0)")]
splits = [("dev", "Development split (16 dialogues, 4 per type; 2 runs × 5 generations)"),
          ("test", "Test split (8 dialogues, held out; 2 per type; 3 runs × 5 generations)")]
for row, (split, label) in enumerate(splits):
    items = D["items"][split]
    for j, (kind, title) in enumerate(order):
        ax = axes[row, j]
        rs = [r for r in items if r["kind"] == kind]
        tgt = np.mean([[0.0] + [r["targets"][k] - r["targets"][1] for k in range(1, 9)] for r in rs], axis=0)
        final = np.mean([r["belief_final"]["change"] for r in rs], axis=0)
        cum = np.mean([r["cumulative"]["change"] for r in rs], axis=0)
        final[0] = cum[0] = 0.0  # changes are measured from the pooled session-0 mean, so session 0 is 0 by definition
        ax.axhline(0, color="#cccccc", lw=0.6)
        ax.plot(x, tgt, color=TGT, ls=(0, (3, 2)), lw=1.2)
        ax.plot(x, cum, color=CUM, ls=(0, (5, 2)), marker="^", ms=2.5, lw=1.2)
        ax.plot(x, final, color=FINAL, marker="o", ms=2.5, lw=1.6)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_ylim(-0.75, 0.8); ax.set_xticks([0, 2, 4, 6, 8]); ax.set_xlabel("Session")
        if j == 0:
            ax.set_ylabel("Change from session 0")
    pos = axes[row, 0].get_position()
    fig.text(0.5, pos.y1 + 0.075, label, ha="center", fontsize=8, style="italic")
handles = [Line2D([], [], color=TGT, ls=(0, (3, 2)), lw=1.2, label="Target change"),
           Line2D([], [], color=FINAL, marker="o", ms=3, lw=1.6, label="Belief state (final update)"),
           Line2D([], [], color=CUM, ls=(0, (5, 2)), marker="^", ms=3, lw=1.2, label="Accumulated dialogue")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
out = HERE / "figures_final" / "validation_sc2.pdf"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, bbox_inches="tight"); print(out)
