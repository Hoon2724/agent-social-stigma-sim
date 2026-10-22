"""Figure 8 (Appendix F): development error E per iteration of the four search loops.

Values are read from the loops' own records: loopA/iterations.json, loopB/autoresearch.jsonl,
loopC/iterations.json, loopD/autoresearch_min.jsonl (SCORE = 1 - E in the jsonl logs).
Each point is the mean of the runs behind that iteration (faint dots = single runs); filled blue = kept, grey =
reverted, hollow = guard failed; the blue step is the running best (the kept file's E).

  python3 a6_search_figure.py   ->  search_loops.pdf
"""
import ast
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
KEEP, DISC, INK, MUTED = "#2f6fd6", "#9aa3ad", "#222222", "#5f6368"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 8})


def progress_data(path):
    """Loop A/C iteration records: iteration, E of each run, status."""
    return [(d["iteration"], d["E_runs"], d["status"], None) for d in json.loads(path.read_text())]


def jsonl_data(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    cfg = rows[0]
    # the starting point is the loop's own baseline rule (Loop B: median of three runs; Loop D: mean of two),
    # computed from the unrounded runs
    out = [(0, [1 - s for s in cfg["baseline_runs"]], "base", 1 - statistics.median(cfg["baseline_runs"]))]
    for r in rows[1:]:
        if r.get("type") != "result":
            continue
        runs = [r["score"]] + ([r["confirm_score"]] if r.get("confirm_score") is not None else [])
        out.append((r["iteration"], [1 - s for s in runs], r["status"], None))
    return out


loops = [
    ("Loop A\nstart: earlier update", progress_data(HERE / "loopA/iterations.json"), 3),
    ("Loop B\nstart: earlier update (plugin)", jsonl_data(HERE / "loopB/autoresearch.jsonl"), 3),
    ("Loop C = final update\nstart: A and B combined", progress_data(HERE / "loopC/iterations.json"), 5),
    ("Loop D (control)\nstart: a single LLM call", jsonl_data(HERE / "loopD/autoresearch_min.jsonl"), 5),
]

fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.0), sharey=True, gridspec_kw={"width_ratios": [11, 9, 6, 15]})
for ax, (title, data, gens) in zip(axes, loops):
    best, xs, ys = None, [], []
    for it, runs, st, rec in data:
        v = rec if rec is not None else sum(runs) / len(runs)
        # Loop B iteration 8 replaced the kept iteration-3 file (a prompt phrase copied six words of a development
        # dialogue was reworded), so the running best goes up there; Loop C's starting point is itself the kept file.
        if st in ("base", "keep") or best is None:
            best = v
        xs.append(it); ys.append(best)
        if len(runs) > 1:
            ax.scatter([it] * len(runs), runs, s=9, color=KEEP if st == "keep" else DISC, alpha=0.35,
                       linewidths=0, zorder=2)
        style = {"keep": dict(s=46, color=KEEP, edgecolors="white", linewidths=1.0),
                 "base": dict(s=40, color=INK, edgecolors="white", linewidths=1.0),
                 "discard": dict(s=26, color=DISC, edgecolors="white", linewidths=0.8),
                 "guard_fail": dict(s=26, facecolors="white", edgecolors=DISC, linewidths=1.2)}[st]
        ax.scatter(it, v, zorder=4, **style)
    ax.step(xs, ys, where="post", color=KEEP, lw=1.6, zorder=3)
    ax.annotate(f"{ys[0]:.3f}", (xs[0], ys[0]), xytext=(3, 5), textcoords="offset points", fontsize=7, color=MUTED)
    up = title.startswith("Loop B")   # Loop B ends on a kept point that sits on the step line
    ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(-6, 8) if up else (-2, -11), textcoords="offset points", fontsize=7.5,
                color=KEEP, fontweight="bold", ha="right")
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.set_xticks(range(0, xs[-1] + 1, 2 if xs[-1] > 6 else 1))
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    ax.grid(axis="y", color="#ececec", lw=0.6)
    ax.text(0.98, 0.97, f"{gens} generations\nper dialogue", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color=MUTED)
axes[1].annotate("phrase leak\nreworded", (8, 0.137), xytext=(5.4, 0.25), fontsize=6.5, color=MUTED,
                 arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.6))
axes[0].set_ylabel("Development error $E$ (lower is better)")
axes[0].set_ylim(0, 0.43)
fig.supxlabel("Iteration (0 = starting point)", fontsize=8, y=0.02)
h = [plt.scatter([], [], s=40, color=INK, edgecolors="white"), plt.scatter([], [], s=46, color=KEEP, edgecolors="white"),
     plt.scatter([], [], s=26, color=DISC, edgecolors="white"),
     plt.scatter([], [], s=26, facecolors="white", edgecolors=DISC, linewidths=1.2),
     plt.Line2D([], [], color=KEEP, lw=1.6)]
fig.legend(h, ["starting point", "kept", "reverted", "guard failed (reverted)", "kept file (running best)"],
           loc="upper center", ncol=5, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
out = HERE / "search_loops.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
for title, data, _ in loops:
    print(title.split("\n")[0], [(it, round(rec if rec is not None else sum(r) / len(r), 3), st) for it, r, st, rec in data])
print(out)
