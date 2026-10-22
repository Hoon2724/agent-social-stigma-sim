"""Further checks on the released runs (Appendix C).

  1. label channel   how many changed Peer sessions came through the label rewrite (engaged entry with new=True)
                     vs the numbered (confirm/contest) channel, and the mean one-session ISMI change of each
  2. strategy mix    primary strategy of every Peer turn; strategy of the Peer turn each accepted quote came from
  3. severity unit   persona-level clustering: (a) mixed model with a persona random intercept and a trajectory
                     variance component; (b) cell-level test (5 cells per severity) of raw and normalised change
  4. case rank       the 15 severe Peer runs ranked by change (the case study of Section 5.4 is the median run)
  further sections: subscales, crisis-keyword screen, rewrites by severity, own-story quotes, engaged directions

  python3 a7_checks.py  ->  a7_checks.json + printout
"""
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "results/runs"
LONG = HERE.parent / "results/ismi_long.csv"
SEV = ["mild", "moderate", "severe"]

d = pd.read_csv(LONG)
d["peer"] = (d["condition"] == "peer").astype(int)
d["severity"] = pd.Categorical(d["severity"], SEV)
pre = d[d.session == 0].groupby(["persona", "severity"], observed=True)["ismi"].mean().rename("pre_pooled")
d = d.join(pre, on=["persona", "severity"])
dp = d.copy()
dp.loc[dp.session == 0, "ismi"] = dp.loc[dp.session == 0, "pre_pooled"]
score = {(r.trajectory_id, r.session): r.ismi for r in dp.itertuples()}
level2cell = d.drop_duplicates("level").set_index("level")[["persona", "severity"]].to_dict("index")
out = {}

# 1-2. label channel and strategies, from the run records
words = lambda s: re.findall(r"[a-z']+", s.lower())
chan = defaultdict(list)          # channel -> one-session ISMI changes
turn_strat, quote_strat = Counter(), Counter()
for cond in ("peer", "general"):
    for f in sorted(RUNS.glob(f"{cond}/level_*/run_[0-9][0-9].json")):
        r = json.loads(f.read_text())
        cell = level2cell[int(f.parent.name.split("_")[1])]
        tid = f"{cell['persona']}_{cell['severity']}_{cond}_{f.stem}"
        turns_by_s = defaultdict(list)
        for t in r["turns"]:
            if t["speaker"] != "patient":
                st = (t.get("metadata") or {}).get("strategy") or {}
                prim = st.get("primary") if isinstance(st, dict) else None
                turns_by_s[t["session"]].append((prim, t["content"]))
                if cond == "peer":
                    turn_strat[prim] += 1
        for b in r["belief_history"][1:]:
            s = b["session"]
            dd = score.get((tid, s), np.nan) - score.get((tid, s - 1), np.nan)
            if not b.get("changed"):
                chan[f"{cond}:unchanged"].append(dd)
                continue
            eng = b.get("engaged") or []
            kind = "label" if any(e.get("new") for e in eng) else "numbered"
            chan[f"{cond}:{kind}"].append(dd)
            if cond == "peer":
                for e in eng:
                    for q in e.get("grounded_in") or []:
                        qw = words(q)
                        grams = {" ".join(qw[i:i + 5]) for i in range(max(1, len(qw) - 4))}
                        hit = [p for p, c in turns_by_s[s] if any(g in " ".join(words(c)) for g in grams)]
                        quote_strat[hit[0] if hit else "unmatched"] += 1
out["channels"] = {k: {"n": len(v), "mean_dISMI": round(float(np.nanmean(v)), 3)} for k, v in sorted(chan.items())}
tot = sum(turn_strat.values())
out["peer_turn_strategy"] = {k: [n, round(n / tot, 3)] for k, n in turn_strat.most_common()}
qt = sum(quote_strat.values())
out["quote_strategy"] = {k: [n, round(n / qt, 3)] for k, n in quote_strat.most_common()}

# 3a. persona clustering
f_traj = smf.mixedlm("ismi ~ peer*session", dp, groups=dp["trajectory_id"]).fit(reml=True)
f_pers = smf.mixedlm("ismi ~ peer*session", dp, groups=dp["persona"],
                     vc_formula={"traj": "0 + C(trajectory_id)"}).fit(reml=True)
fs = smf.mixedlm("ismi ~ peer*session*C(severity)", dp, groups=dp["persona"],
                 vc_formula={"traj": "0 + C(trajectory_id)"}).fit(reml=True)
def b(fit, name):
    ci = fit.conf_int().loc[name]
    return {"beta": round(float(fit.params[name]), 4), "ci": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
            "p": float(fit.pvalues[name])}
out["overall_traj_ri"] = b(f_traj, "peer:session")
out["overall_persona_ri"] = b(f_pers, "peer:session")
out["severe_vs_mild_persona_ri"] = b(fs, "peer:session:C(severity)[T.severe]")
out["moderate_vs_mild_persona_ri"] = b(fs, "peer:session:C(severity)[T.moderate]")

# 3b. cell level (Peer runs averaged within persona x severity)
w = dp.pivot_table(index=["trajectory_id", "persona", "severity", "condition"], columns="session",
                   values="ismi", observed=True).reset_index()
w["delta"] = w[8] - w[0]
w["norm"] = w["delta"] / (w[0] - 1)
cells = w[w.condition == "peer"].groupby(["persona", "severity"], observed=True)[["delta", "norm"]].mean().reset_index()
gen = w[w.condition == "general"].groupby(["persona", "severity"], observed=True)["delta"].mean()
cells["diff"] = cells["delta"].values - gen.reindex(pd.MultiIndex.from_frame(cells[["persona", "severity"]])).values
for col in ("delta", "norm"):
    a, c = cells[cells.severity == "mild"][col], cells[cells.severity == "severe"][col]
    out[f"cell_{col}_severe_vs_mild"] = {
        "mild": [round(x, 3) for x in a], "severe": [round(x, 3) for x in c],
        "pairs_severe_more_negative": int(sum(y < x for x in a for y in c)), "of": len(a) * len(c),
        "mannwhitney_p_exact_one_sided": float(stats.mannwhitneyu(c, a, alternative="less", method="exact").pvalue),
        "mannwhitney_p_exact_two_sided": float(stats.mannwhitneyu(c, a, alternative="two-sided", method="exact").pvalue)}
out["cells_diff_gt_0.07"] = int((cells["diff"] < -0.07).sum())

# 4. case rank
sev = w[(w.condition == "peer") & (w.severity == "severe")].sort_values("delta")
out["severe_peer_runs_by_delta"] = [[t, round(x, 3)] for t, x in zip(sev.trajectory_id, sev.delta)]
out["severe_peer_mean_delta"] = round(float(sev.delta.mean()), 3)
out["severe_peer_median_delta"] = round(float(sev.delta.median()), 3)

(HERE / "a7_checks.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

# 5. ISMI subscales (stored per session by the judge) and the total without Stigma Resistance
SUB = ["alienation", "stereotype_endorsement", "discrimination_experience", "social_withdrawal", "stigma_resistance"]
NITEM = {"alienation": 6, "stereotype_endorsement": 7, "discrimination_experience": 5, "social_withdrawal": 6,
         "stigma_resistance": 5}
rows = []
for cond in ("peer", "general"):
    for f in sorted(RUNS.glob(f"{cond}/level_*/run_[0-9][0-9].beliefonly.qwen.ismi29.json")):
        m = json.loads(f.read_text())
        cell = level2cell[int(f.parent.name.split("_")[1])]
        pre_s = m["ismi29_pre"]["subscale_means"]
        post_s = [x for x in m["ismi29_post_by_session"] if x["session"] == 8][0]["subscale_means"]
        rows.append({"cond": cond, **cell, "run": f.name[:6],
                     **{f"pre_{k}": pre_s[k] for k in SUB}, **{f"post_{k}": post_s[k] for k in SUB}})
sd = pd.DataFrame(rows)
for k in SUB:   # pooled pre per cell, as for the total
    sd[f"pre_{k}"] = sd.groupby(["persona", "severity"])[f"pre_{k}"].transform("mean")
    sd[f"d_{k}"] = sd[f"post_{k}"] - sd[f"pre_{k}"]
no_sr = lambda p: sum(sd[f"{p}_{k}"] * NITEM[k] for k in SUB if k != "stigma_resistance") / 24
sd["d_total_noSR"] = no_sr("post") - no_sr("pre")
out["subscale_delta"] = {c: {k: round(float(g[f"d_{k}"].mean()), 3) for k in SUB + ["total_noSR"]}
                         for c, g in sd.groupby("cond")}

# 6. crisis content in Patient turns (keyword screen, counts only)
KW = re.compile(r"suicid|kill myself|end it all|end my life|self[- ]harm|hurt myself|cut myself|"
                r"don'?t want to be here|not want to be alive|better off dead|want to die", re.I)
hits = Counter()
for f in sorted(RUNS.glob("*/level_*/run_[0-9][0-9].json")):
    r = json.loads(f.read_text())
    n = sum(1 for t in r["turns"] if t["speaker"] == "patient" and KW.search(t["content"]))
    if n:
        hits[f"{f.parent.parent.name}/{f.parent.name}/{f.stem}"] = n
out["patient_crisis_keyword_turns"] = {"runs_with_hits": len(hits), "turns": sum(hits.values()), "by_run": dict(hits)}
(HERE / "a7_checks.json").write_text(json.dumps(out, indent=1))
print("subscale", out["subscale_delta"]); print("crisis", out["patient_crisis_keyword_turns"])

# 7. persona as the unit (cells averaged over 3 runs; Peer vs General paired within persona x severity)
def slope(g):
    return np.polyfit(g["session"], g["ismi"], 1)[0]
sl = dp.groupby(["trajectory_id", "persona", "severity", "condition"], observed=True).apply(slope).rename("slope").reset_index()
cs = sl.groupby(["persona", "severity", "condition"], observed=True)["slope"].mean().unstack("condition")
cs["diff"] = cs["peer"] - cs["general"]
out["persona_unit_slope_diff"] = {}
for s in SEV:
    x = cs.xs(s, level="severity")["diff"]
    out["persona_unit_slope_diff"][s] = {"mean": round(float(x.mean()), 4), "p": float(stats.ttest_1samp(x, 0).pvalue),
                                          "n_negative": int((x < 0).sum())}
allx = cs["diff"]
out["persona_unit_slope_diff"]["all_cells"] = {"mean": round(float(allx.mean()), 4), "n_negative": int((allx < 0).sum()),
                                               "wilcoxon_p": float(stats.wilcoxon(allx).pvalue)}
pm = cs["diff"].groupby(level="persona").mean()
out["persona_unit_slope_diff"]["personas"] = {"n_negative": int((pm < 0).sum()), "p": float(stats.ttest_1samp(pm, 0).pvalue)}
sev_m = cs.xs("severe", level="severity")["diff"]; mild_m = cs.xs("mild", level="severity")["diff"]
mod_m = cs.xs("moderate", level="severity")["diff"]
out["persona_unit_severe_vs_mild_p"] = float(stats.ttest_rel(sev_m, mild_m).pvalue)
out["persona_unit_moderate_vs_mild_p"] = float(stats.ttest_rel(mod_m, mild_m).pvalue)
cn = cells.set_index(["persona", "severity"])["norm"]
out["persona_unit_norm_severe_vs_mild_p"] = float(stats.ttest_rel(cn.xs("severe", level="severity"),
                                                                  cn.xs("mild", level="severity")).pvalue)
out["persona_unit_norm_moderate_vs_mild_p"] = float(stats.ttest_rel(cn.xs("moderate", level="severity"),
                                                                    cn.xs("mild", level="severity")).pvalue)

# 8. rewrites per Peer trajectory and score change per changed session, by severity
per = defaultdict(lambda: {"changed": 0, "traj": set(), "d": []})
for f in sorted(RUNS.glob("peer/level_*/run_[0-9][0-9].json")):
    r = json.loads(f.read_text())
    cell = level2cell[int(f.parent.name.split("_")[1])]
    tid = f"{cell['persona']}_{cell['severity']}_peer_{f.stem}"
    sv = cell["severity"]; per[sv]["traj"].add(tid)
    for b in r["belief_history"][1:]:
        if b.get("changed"):
            per[sv]["changed"] += 1
            per[sv]["d"].append(score[(tid, b["session"])] - score[(tid, b["session"] - 1)])
out["rewrites_by_severity"] = {s: {"changed_per_traj": round(v["changed"] / len(v["traj"]), 2),
                                   "mean_d_per_changed": round(float(np.mean(v["d"])), 3)} for s, v in per.items()}
(HERE / "a7_checks.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("persona_unit_slope_diff", "persona_unit_severe_vs_mild_p",
      "persona_unit_moderate_vs_mild_p", "persona_unit_norm_severe_vs_mild_p", "persona_unit_norm_moderate_vs_mild_p",
      "rewrites_by_severity")}, indent=1))

# 9. LaTeX tables for Appendix C
TEX = HERE / "tables_final"
TEX.mkdir(exist_ok=True)
NAMES = [("hopeful_disclosure", "Positive self-disclosure"), ("treatment_modeling", "Role modeling"),
         ("helper_reversal", "Helper principle"), ("strengths_focused_support", "Strengths-focused support"),
         ("attuned_encouragement", "Empathic regard")]
qt_all = {k: v for k, v in out["quote_strategy"].items() if k != "unmatched"}
qn = sum(v[0] for v in qt_all.values())
lines = [r"\begin{tabular}{lrr}", r"\toprule",
         r"Strategy & Peer turns & Quotes behind accepted rewrites \\", r"\midrule"]
for code, name in NAMES:
    tn = out["peer_turn_strategy"].get(code, [0, 0])
    q = qt_all.get(code, [0, 0])
    lines.append(f"{name} & {tn[0]} ({100 * tn[0] / tot:.0f}\\%) & {q[0]} ({100 * q[0] / qn:.0f}\\%) \\\\")
lines += [r"\midrule", f"Total & {tot} & {qn} \\\\", r"\bottomrule", r"\end{tabular}"]
(TEX / "appendix_strategy_table.tex").write_text("\n".join(lines).replace("(0\\%)", "($<$1\\%)") + "\n")

SUBN = [("alienation", "Alienation"), ("stereotype_endorsement", "Stereotype endorsement"),
        ("discrimination_experience", "Discrimination experience"), ("social_withdrawal", "Social withdrawal"),
        ("stigma_resistance", "Stigma resistance (reverse-scored)"), ("total_noSR", "Total without stigma resistance")]
lines = [r"\begin{tabular}{lrr}", r"\toprule", r"ISMI subscale & Peer $\Delta$ & General $\Delta$ \\", r"\midrule"]
for k, name in SUBN:
    lines.append(f"{name} & ${out['subscale_delta']['peer'][k]:+.2f}$ & ${out['subscale_delta']['general'][k]:+.2f}$ \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TEX / "appendix_subscale_table.tex").write_text("\n".join(lines).replace("+0.00", "0.00").replace("-0.00", "0.00") + "\n")
print((TEX / "appendix_strategy_table.tex").read_text()); print((TEX / "appendix_subscale_table.tex").read_text())

# 10. quotes behind accepted Peer rewrites that are the Peer's own first-person account (first person, no "you"),
#     and the direction votes (confirms / contests / neither) recorded by the filter
FP = re.compile(r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself)\b", re.I)
SP = re.compile(r"\b(you|your|you're|you've|yours|yourself)\b", re.I)
own, total = 0, 0
for f in sorted(RUNS.glob("peer/level_*/run_[0-9][0-9].json")):
    for b in json.loads(f.read_text())["belief_history"][1:]:
        if not b.get("changed"):
            continue
        for e in b.get("engaged") or []:
            for q in e.get("grounded_in") or []:
                total += 1
                own += bool(FP.search(q) and not SP.search(q))
out["peer_own_story_quotes"] = {"own_first_person_no_you": own, "of": total}
(HERE / "a7_checks.json").write_text(json.dumps(out, indent=1))
print("own-story quotes", own, "of", total)

# 11. direction of every engaged sentence behind an accepted change (confirms / contests / label rewrite)
dirs = {c: Counter() for c in ("peer", "general")}
for c in dirs:
    for f in sorted(RUNS.glob(f"{c}/level_*/run_[0-9][0-9].json")):
        for b in json.loads(f.read_text())["belief_history"][1:]:
            for e in (b.get("engaged") or []) if b.get("changed") else []:
                dirs[c][e.get("direction") or "label rewrite"] += 1
out["engaged_directions"] = {c: dict(v) for c, v in dirs.items()}
(HERE / "a7_checks.json").write_text(json.dumps(out, indent=1))
print("engaged directions", out["engaged_directions"])
