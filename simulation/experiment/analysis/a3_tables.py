"""Table 3 (overall + severity), Tables 6/7 (persona-level trajectories) and Table 8 (gate / belief change) from
the released runs, in the paper's LaTeX layout. Session 0 uses the pooled pre (mean of every scoring of the same
B0), so both conditions of a persona x severity cell start from one number.

  python3 a3_tables.py [--data final]  ->  tables_<data>/*.tex + a3_tables_<data>.json
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
SRC = {"final": (HERE.parent / "results/ismi_long.csv", HERE.parent / "results/runs")}
SEV = ["mild", "moderate", "severe"]
PERSONAS = ["Avery", "Jordan", "Riley", "Ravi", "Noor"]
LEVEL = {s: i for i, s in enumerate(SEV)}  # level_1x0 severe .. 1x2 mild? -- resolved from the csv below


def load(path):
    d = pd.read_csv(path)
    d["peer"] = (d["condition"] == "peer").astype(int)
    d["severity"] = pd.Categorical(d["severity"], SEV)
    pre = d[d.session == 0].groupby(["persona", "severity"], observed=True)["ismi"].mean().rename("pre_pooled")
    d = d.join(pre, on=["persona", "severity"])
    d.loc[d.session == 0, "ismi"] = d.loc[d.session == 0, "pre_pooled"]
    return d


def fmt_p(p):
    return "$<.001$" if p < 0.001 else f"{p:.3f}".lstrip("0")


def m(x, nd=3):
    return f"${'-' if x < 0 else '+'}{abs(x):.{nd}f}$".replace("-", "-")


def table3(d):
    w = d.pivot_table(index=["trajectory_id", "severity", "condition"], columns="session", values="ismi",
                      observed=True).reset_index()
    w["delta"] = w[8] - w[0]
    dl = lambda sub, c: float(sub[sub.condition == c]["delta"].mean())
    rows = {}
    f3 = smf.mixedlm("ismi ~ peer*session", d, groups=d["trajectory_id"]).fit(reml=True)
    ci = f3.conf_int().loc["peer:session"]
    rows["Overall"] = dict(peer=dl(w, "peer"), general=dl(w, "general"), beta=float(f3.params["peer:session"]),
                           lo=float(ci[0]), hi=float(ci[1]), p=float(f3.pvalues["peer:session"]))
    f4 = smf.mixedlm("ismi ~ peer*session*C(severity, Treatment('mild'))", d, groups=d["trajectory_id"]).fit(reml=True)
    names = list(f4.model.exog_names)
    contrasts = {}
    for s in SEV:
        v = np.zeros(len(names))
        v[names.index("peer:session")] = 1
        if s != "mild":
            v[names.index(f"peer:session:C(severity, Treatment('mild'))[T.{s}]")] = 1
        t = f4.t_test(v[None, :])
        lo, hi = t.conf_int()[0]
        sub = w[w.severity == s]
        rows[s.capitalize()] = dict(peer=dl(sub, "peer"), general=dl(sub, "general"), beta=float(t.effect[0]),
                                    lo=float(lo), hi=float(hi), p=float(t.pvalue))
        if s != "mild":
            k = f"peer:session:C(severity, Treatment('mild'))[T.{s}]"
            contrasts[f"{s}_vs_mild"] = {"beta": float(f4.params[k]), "p": float(f4.pvalues[k])}
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{$\Delta$ISMI} & \multicolumn{3}{c}{Peer--General slope difference} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
             r"Initial severity & Peer & General & $\beta$ & 95\% CI & $p$ \\", r"\midrule"]
    for k, r in rows.items():
        lines.append(f"{k} & {m(r['peer'])} & {m(r['general'])} & {m(r['beta'])} & "
                     f"$[{r['lo']:.3f}, {r['hi']:.3f}]$ & {fmt_p(r['p'])} \\\\")
        if k == "Overall":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), {"rows": rows, "severity_contrasts_vs_mild": contrasts}


def table_traj(d, cond):
    g = d[d.condition == cond].groupby(["persona", "severity", "session"], observed=True)["ismi"].mean().unstack()
    lines = [r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
             r"Persona & Severity & pre & s1 & s2 & s3 & s4 & s5 & s6 & s7 & s8 \\", r"\midrule"]
    for i, p in enumerate(PERSONAS):
        for s in SEV:
            vals = " & ".join(f"{g.loc[(p, s), k]:.3f}" for k in range(9))
            lines.append(f"{p} & {s.capitalize()} & {vals} \\\\")
        if i < len(PERSONAS) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), {f"{p}|{s}": [round(float(g.loc[(p, s), k]), 3) for k in range(9)] for p in PERSONAS for s in SEV}


def table8(raw: Path, d):
    sev_of = d.drop_duplicates("level").set_index("level")["severity"].astype(str).to_dict()
    counts = {}
    for cond in ("peer", "general"):
        for f in sorted((raw / cond).glob("level_*/run_*.json")):
            if ".ismi29" in f.name:
                continue
            sev = sev_of[int(f.parent.name.split("_")[1])]
            c = counts.setdefault((cond, sev), [0, 0, 0])
            for b in json.loads(f.read_text())["belief_history"]:
                if b.get("session", 0) == 0:
                    continue
                c[0] += 1
                c[1] += bool(b.get("engaged"))
                c[2] += bool(b.get("changed"))
    pct = lambda a, n: f"{a}/{n} ({100 * a / n:.1f}\\%)"
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Peer} & \multicolumn{2}{c}{General} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Severity & Gate open & Belief changed & Gate open & Belief changed \\", r"\midrule"]
    tot = {c: [0, 0, 0] for c in ("peer", "general")}
    for s in ["severe", "moderate", "mild"]:
        cells = []
        for cond in ("peer", "general"):
            n, e, ch = counts[(cond, s)]
            tot[cond] = [x + y for x, y in zip(tot[cond], (n, e, ch))]
            cells += [pct(e, n), pct(ch, n)]
        lines.append(f"{s.capitalize()} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append("Overall & " + " & ".join(pct(tot[c][i], tot[c][0]) for c in ("peer", "general") for i in (1, 2)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), {f"{c}|{s}": v for (c, s), v in counts.items()}


ap = argparse.ArgumentParser()
ap.add_argument("--data", choices=SRC, default="final")
a = ap.parse_args()
csv_path, raw = SRC[a.data]
d = load(csv_path)
out = HERE / f"tables_{a.data}"
out.mkdir(exist_ok=True)
res = {}
builders = {"table3": table3(d), "table6_peer": table_traj(d, "peer"), "table7_general": table_traj(d, "general")}
if a.data == "final":  # Table 8 needs the gate records kept in the run files
    builders["table8_gate"] = table8(raw, d)
for name, (tex, info) in builders.items():
    (out / f"{name}.tex").write_text(tex + "\n")
    res[name] = info
(HERE / f"a3_tables_{a.data}.json").write_text(json.dumps(res, indent=1, default=float))
print((out / "table3.tex").read_text())
print(json.dumps(res["table3"]["severity_contrasts_vs_mild"], indent=1))
if a.data == "final":
    print((out / "table8_gate.tex").read_text())
