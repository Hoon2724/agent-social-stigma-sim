"""Print the autoresearch SCORE for one finished evaluation run (results/v3_runs/<run>/_score.json).

SCORE = 1 - E, where E = mean over the 16 dev dialogues of |measured change - target change| (score_v3.py).
Higher is better. GUARD = score_v3's per-cell guards (reinforcing cells > 0, reducing cells < 0,
each of the four neutral cells |mean change| <= 0.15). Prints `SCORE: X.XXXX` on its own line.
"""
import json
import sys
from pathlib import Path

run = sys.argv[1]
p = Path(__file__).resolve().parent.parent / "results" / "v3_runs" / run / "_score.json"
if not p.exists():
    print(f"CRASH: no score file for {run} (see results/v3_runs/_{run}.log)")
    sys.exit(1)
s = json.loads(p.read_text())
if s.get("missing"):
    print(f"CRASH: dialogues missing {s['missing']}")
    sys.exit(1)
print(f"E: {s['E']:.4f}")
print("GUARD: " + ("PASS" if s["guards_pass"] else "FAIL"))
for c in s["cells"]:
    print(f"  cell {c['cell']} {c['kind']:18s} start {c['start']}  mean change {c['mean_delta']:+.3f}  {'ok' if c['ok'] else 'FAIL'}")
for k, v in s["kinds"].items():
    print(f"  kind {k:18s} mean change {v['mean_delta']:+.3f}  share of sessions changed {v['changed_share']:.2f}")
print(f"SCORE: {1 - s['E']:.4f}")
