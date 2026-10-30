"""Appendix tables generated from the analysis outputs, so no number is typed by hand.

  tables_final/appendix_validity_table.tex  evaluation set: belief state vs accumulated dialogue, by split and
                                   trajectory type (change at session 8 and trajectory error)
  tables_final/appendix_sensitivity.tex     Peer x session slope difference under alternative specifications

  python3 a5_appendix_tables.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "tables_final"
TEX.mkdir(exist_ok=True)
V = json.loads((HERE.parent / "results/summaries/a4_validity.json").read_text())
A2 = json.loads((HERE / "a2_sensitivity.json").read_text()) if (HERE / "a2_sensitivity.json").exists() else json.loads((HERE.parent / "results/summaries/a2_sensitivity.json").read_text())


# 2. Evaluation set: belief state vs accumulated dialogue
S = V["summary"]
order = ["worsening", "reducing", "flat (dry)", "flat (empathic)"]
lines = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"Split & Input & Overall & Worsening & Reducing & Flat (dry) & Flat (empathic) \\",
         r" & & traj.\ error & \multicolumn{4}{c}{change at session 8 (trajectory error); target $+0.5$ / $-0.5$ / $0$ / $0$} \\",
         r"\midrule"]
for split, label in (("dev", "Development (16)"), ("test", "Test, held out (8)")):
    for arm, name in (("belief_final", "Belief state"), ("cumulative", "Accumulated dialogue")):
        s = S[split][arm]
        cells = " & ".join(f"${s['by_kind'][k]['delta8']:+.2f}$ ({s['by_kind'][k]['traj']:.2f})" for k in order)
        lines.append(f"{label if arm == 'belief_final' else ''} & {name} & {s['traj']:.3f} & {cells} \\\\")
    lines.append(r"\addlinespace")
lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}"]
(TEX / "appendix_validity_table.tex").write_text("\n".join(lines).replace("+0.00", "0.00").replace("-0.00", "0.00") + "\n")

# 3. Sensitivity of the Peer x session slope difference
spec = [("pooled_pre", "Random intercept, pooled session-0 score (Eq.~\\ref{eq:lmm}, as reported)"),
        ("intercept", "Random intercept, each trajectory's own session-0 score"),
        ("random_slope", "Random intercept and slope per trajectory (own session-0 score)"),
        ("gee_ar1", "GEE, AR(1) correlation across sessions (own session-0 score)")]
lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Specification & $\beta$ & 95\% CI & $p$ \\", r"\midrule"]
for key, name in spec:
    b = A2["final"][key]
    p = "$<.001$" if b["p"] < 0.001 else f"{b['p']:.3f}"
    name_ = name.replace("sessions (", f"sessions, $\\rho={b['rho']:.3f}$ (") if "rho" in b else name
    lines.append(f"{name_} & ${b['beta']:+.3f}$ & $[{b['ci'][0]:.3f}, {b['ci'][1]:.3f}]$ & {p} \\\\")
A7 = json.loads((HERE / "a7_checks.json").read_text()) if (HERE / "a7_checks.json").exists() else json.loads((HERE.parent / "results/summaries/a7_checks.json").read_text())
pu = A7["persona_unit_slope_diff"]
fp = lambda x: "$<.001$" if x < 0.001 else "$" + f"{x:.3f}".lstrip("0") + "$"
lines += [r"\midrule", r"\multicolumn{4}{l}{\textit{Persona as the unit: per-cell mean slope difference over the three runs}} \\",
          f"All 15 cells (Wilcoxon; {pu['all_cells']['n_negative']} of 15 negative) & ${pu['all_cells']['mean']:+.3f}$ & & {fp(pu['all_cells']['wilcoxon_p'])} \\\\",
          f"Five personas ($t$-test; {pu['personas']['n_negative']} of 5 negative) & ${pu['all_cells']['mean']:+.3f}$ & & {fp(pu['personas']['p'])} \\\\"]
for s_ in ("mild", "moderate", "severe"):
    lines.append(f"{s_.capitalize()} (five personas, $t$-test) & ${pu[s_]['mean']:+.3f}$ & & {fp(pu[s_]['p'])} \\\\")
lines.append(f"Severe vs.\\ mild, moderate vs.\\ mild (paired by persona) & & & {fp(A7['persona_unit_severe_vs_mild_p'])}, {fp(A7['persona_unit_moderate_vs_mild_p'])} \\\\")
sv = A2["final"]["severity_pooled"]
lines += [r"\midrule", r"\multicolumn{4}{l}{\textit{Peer change as a share of the distance from pre to the scale minimum}} \\"]
for s in ("mild", "moderate", "severe"):
    lines.append(f"{s.capitalize()} (pre {sv[s]['pre']:.2f}) & \\multicolumn{{3}}{{l}}{{${100 * sv[s]['norm_peer']:+.0f}\\%$ "
                 f"(General ${100 * sv[s]['norm_general']:+.0f}\\%$)}} \\\\")
nm = A2["final"]["norm_peer_vs_mild"]
lines.append(rf"\multicolumn{{4}}{{l}}{{Difference from mild: moderate ${100 * nm['moderate']['diff']:+.0f}$ points ($p=.{str(round(nm['moderate']['p'],2))[2:]}$; "
             rf"paired by persona ${fp(A7['persona_unit_norm_moderate_vs_mild_p']).strip('$')}$), "
             rf"severe ${100 * nm['severe']['diff']:+.0f}$ points ($p=.{str(round(nm['severe']['p'],3))[2:]}$; paired by persona "
             rf"${fp(A7['persona_unit_norm_severe_vs_mild_p']).strip('$')}$)}} \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TEX / "appendix_sensitivity.tex").write_text("\n".join(lines).replace("+0\\%", "0\\%").replace("-0\\%", "0\\%") + "\n")
for f in ("appendix_validity_table.tex", "appendix_sensitivity.tex"):
    print(f"== {f}\n" + (TEX / f).read_text())
