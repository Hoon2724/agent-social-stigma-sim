"""Figure 6: belief state vs accumulated dialogue, drawn from ../results/summaries/a4_validity.json.
Top row: the two GPT-5.5 dialogues of Section 4.3 on the ISMI scale (target, belief state under the final update
and, for reference, under the earlier update, accumulated dialogue). Bottom row: the sealed test split of the
evaluation set, change from session 0 by trajectory type (target change, belief state, accumulated dialogue).

  python3 a4_figure.py  ->  figures_final/validation_sc2.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
D = json.loads((HERE.parent / "results/summaries/a4_validity.json").read_text())
FINAL, PAPER, CUM, TGT = "#2f6fd6", "#9aa3ad", "#e8664a", "#222222"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 8})
x = np.arange(9)
fig = plt.figure(figsize=(7.4, 4.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.75, wspace=0.32)

names = {"F_reducing": "Section 4.3 dialogue: stigma-reducing", "F_neutral": "Section 4.3 dialogue: no change"}
def from_base(curve, base):
    # Session 0 is the same text for every input; draw it at the pooled mean of all its scorings (the point
    # changes are measured from) rather than at each input's own judge draw.
    return [base] + list(curve[1:])


for j, r in enumerate(D["items"]["fig6"]):
    ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
    ax.plot(x, r["targets"], color=TGT, ls=(0, (3, 2)), lw=1.2, label="Target")
    ax.plot(x, from_base(r["cum_curve"], r["base"]), color=CUM, ls=(0, (5, 2)), marker="^", ms=3, lw=1.3,
            label=f"Accumulated dialogue (error {r['cumulative']['traj']:.2f})")
    ax.plot(x, from_base(r["belief_earlier_curve"], r["base"]), color=PAPER, marker="s", ms=2.5, lw=1.2,
            label=f"Belief state, earlier update (error {r['belief_earlier']['traj']:.2f})")
    ax.plot(x, from_base(r["belief_final_curve"], r["base"]), color=FINAL, marker="o", ms=3, lw=1.8,
            label=f"Belief state, final update (error {r['belief_final']['traj']:.2f})")
    ax.set_title(names[r["id"]], fontsize=8.5, fontweight="bold")
    ax.set_ylim(1.7, 3.7); ax.set_xticks(x); ax.set_xlabel("Session")
    if j == 0:
        ax.set_ylabel("ISMI-29")
    ax.legend(fontsize=6.3, frameon=False, loc="lower left" if j == 0 else "lower left")

order = [("stigma_reinforcing", "Worsening (+0.5)"), ("stigma_reducing", "Reducing (−0.5)"),
         ("neutral_dry", "Flat, dry (0)"), ("general_like", "Flat, empathic (0)")]
test = D["items"]["test"]
for j, (kind, title) in enumerate(order):
    ax = fig.add_subplot(gs[1, j])
    rs = [r for r in test if r["kind"] == kind]
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
fig.text(0.5, 0.468, "Evaluation set, sealed test split (2 dialogues per type; final update vs accumulated dialogue)",
         ha="center", fontsize=8, style="italic")
out = HERE / "figures_final" / "validation_sc2.pdf"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, bbox_inches="tight"); print(out)
