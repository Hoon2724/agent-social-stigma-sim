"""Section 4.2 and Appendix A.2: belief state vs accumulated dialogue as the input the ISMI judge reads, on the
evaluation set (development 16, sealed test 8).

Both arms start from the same session-0 note, so both are anchored to one session-0 score (the pooled mean of
every judgment of that note) and compared on the change the dialogue was written to produce:
  trajectory error = mean over sessions 1..8 of |(S_s - S_0) - (T_s - T_1)|,  final error = same at session 8.
Targets are taken as change from the session-1 target: the evaluation set wrote its per-session targets from the
measured session-0 reading, so the nominal start (2.0/2.5/3.0) is not the target's own starting point.
Belief-state arm = runs of the final update (../results/validity/dev_final_r{1,2}: 2 x 5 generations).
Accumulated-dialogue arm = ../results/validity/cumulative_dev. The sealed test split is read only where it
exists (the authors' machines).

  python3 a4_validity.py  ->  a4_validity.json + printed tables
"""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent / "results/validity"
EV = HERE.parent / "evalset"
SEALED = Path.home() / "belief_v2_sealed"
KIND = {"stigma_reinforcing": "worsening", "stigma_reducing": "reducing", "neutral_dry": "flat (dry)",
        "general_like": "flat (empathic)"}
SPLITS = {
    "dev": dict(items=EV / "dev.jsonl", pooled=EV / "s0_pooled.json",
                belief={"final": [R / "dev_final_r1", R / "dev_final_r2"]},
                cum=R / "cumulative_dev"),
    "test": dict(items=SEALED / "test.jsonl", pooled=SEALED / "s0_pooled.json",
                 belief={"final": [SEALED / f"runs/rerun_owf_r{r}" for r in (1, 2, 3)]},
                 cum=SEALED / "runs/cr_cumulative_test"),
}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.fmean(xs) if xs else None


def belief_curve(dirs, tid):
    runs = []
    for d in dirs:
        p = d / f"{tid}.json"
        if p.exists():
            runs += json.loads(p.read_text())["runs"]
    if not runs:
        return None, None, []
    s0_sha = next(s.get("text_sha") for s in runs[0]["sessions"] if s["session"] == 0)
    curve = [mean([next((s.get("ismi") for s in r["sessions"] if s["session"] == k), None) for r in runs]) for k in range(9)]
    s0_reps = [v for r in runs for s in r["sessions"] if s["session"] == 0 for v in (s.get("per_rep") or [s.get("ismi")])]
    return curve, s0_sha, s0_reps


def errors(curve, base, targets):
    d = [curve[k] - base for k in range(9)]
    t = [0.0] + [targets[k] - targets[1] for k in range(1, 9)]  # change from the session-1 target 
    traj = mean([abs(d[k] - t[k]) for k in range(1, 9)])
    return {"traj": traj, "final": abs(d[8] - t[8]), "delta8": d[8], "target8": t[8],
            "abs": mean([abs(curve[k] - targets[k]) for k in range(9)]), "change": d}


out = {}
if not (SPLITS['test']['items']).exists():   # sealed split: only on the authors' machines
    SPLITS.pop('test')
for split, cfg in SPLITS.items():
    pooled = json.loads(cfg["pooled"].read_text()) if cfg["pooled"] else {}
    rows = []
    for line in cfg["items"].read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        tid = item["test_id"]
        targets = {t["session"]: t["target_ismi"] for t in item["hidden_ground_truth"]["session_targets"]}
        targets[0] = item["condition"]["initial_ismi"]
        targets = [targets[k] for k in range(9)]
        cum = json.loads((cfg["cum"] / f"{tid}.json").read_text())
        cum_curve = [next(s["ismi"] for s in cum["sessions"] if s["session"] == k) for k in range(9)]
        row = {"id": tid, "kind": item["condition"]["trajectory_type"], "targets": targets, "cum_curve": cum_curve}
        all_s0 = list(next(s["per_rep"] for s in cum["sessions"] if s["session"] == 0))
        curves = {}
        for name, dirs in cfg["belief"].items():
            curve, sha, s0_reps = belief_curve(dirs, tid)
            if curve is None:
                continue
            curves[name] = curve
            all_s0 += s0_reps
            row.setdefault("s0_sha", sha)
        base = pooled.get(row.get("s0_sha"), {}).get("mean") if pooled else None
        row["base"] = base if base is not None else st.fmean(all_s0)
        row["cumulative"] = errors(cum_curve, row["base"], targets)
        for name, curve in curves.items():
            row[f"belief_{name}"] = errors(curve, row["base"], targets)
            row[f"belief_{name}_curve"] = curve
        rows.append(row)
    out[split] = rows

summary = {}
for split, rows in out.items():
    arms = [k for k in rows[0] if k.startswith("belief_") and not k.endswith("_curve")] + ["cumulative"]
    print(f"== {split} ({len(rows)} dialogues)")
    for arm in arms:
        rs = [r for r in rows if arm in r]
        s = {"traj": mean([r[arm]["traj"] for r in rs]), "final": mean([r[arm]["final"] for r in rs]),
             "abs": mean([r[arm]["abs"] for r in rs])}
        by = {}
        for kind in sorted({r["kind"] for r in rs}):
            kr = [r for r in rs if r["kind"] == kind]
            by[KIND[kind]] = {"traj": mean([r[arm]["traj"] for r in kr]), "delta8": mean([r[arm]["delta8"] for r in kr]),
                              "target8": kr[0][arm]["target8"]}
        summary.setdefault(split, {})[arm] = {**s, "by_kind": by}
        print(f"  {arm:14s} traj {s['traj']:.3f}  final {s['final']:.3f}  abs {s['abs']:.3f}   " +
              "  ".join(f"{k}: d8 {v['delta8']:+.2f}/{v['target8']:+.1f}" for k, v in by.items()))
(HERE / "a4_validity.json").write_text(json.dumps({"summary": summary, "items": out}, indent=1))
if "test" not in out:
    print("sealed split not available: its numbers are in ../results/summaries/a4_validity.json")
