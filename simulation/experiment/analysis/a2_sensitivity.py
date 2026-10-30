"""Sensitivity checks of the Peer-General slope difference (Appendix G.4) on the released runs.

  python3 a2_sensitivity.py  ->  a2_sensitivity.json + printed tables
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
DATA = {"final": Path(__file__).resolve().parents[1] / "results/ismi_long.csv"}
SEV = ["mild", "moderate", "severe"]


def load(path):
    d = pd.read_csv(path)
    d["peer"] = (d["condition"] == "peer").astype(int)
    d["severity"] = pd.Categorical(d["severity"], SEV)
    return d


def pooled(d):
    """Replace each trajectory's session-0 score with the mean over all session-0 scorings of that B0."""
    d = d.copy()
    pre = d[d.session == 0].groupby(["persona", "severity"], observed=True)["ismi"].mean().rename("pre_pooled")
    d = d.join(pre, on=["persona", "severity"])
    d.loc[d.session == 0, "ismi"] = d.loc[d.session == 0, "pre_pooled"]
    return d


def beta(fit, name="peer:session"):
    ci = fit.conf_int().loc[name]
    return {"beta": round(float(fit.params[name]), 4), "ci": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
            "p": float(fit.pvalues[name])}


def deltas(d):
    w = d.pivot_table(index=["trajectory_id", "persona", "severity", "condition"], columns="session",
                      values="ismi", observed=True).reset_index()
    w["pre"], w["post"] = w[0], w[8]
    w["delta"] = w["post"] - w["pre"]
    w["norm"] = w["delta"] / (w["pre"] - 1.0)
    return w


def analyse(d):
    out = {}
    out["intercept"] = beta(smf.mixedlm("ismi ~ peer*session", d, groups=d["trajectory_id"]).fit(reml=True))
    dp = pooled(d)
    out["pooled_pre"] = beta(smf.mixedlm("ismi ~ peer*session", dp, groups=dp["trajectory_id"]).fit(reml=True))
    out["random_slope"] = beta(smf.mixedlm("ismi ~ peer*session", d, groups=d["trajectory_id"],
                                           re_formula="~session").fit(reml=True))
    gee = smf.gee("ismi ~ peer*session", groups="trajectory_id", data=d.sort_values(["trajectory_id", "session"]),
                  time=d.sort_values(["trajectory_id", "session"])["session"].values,
                  cov_struct=sm.cov_struct.Autoregressive(grid=True)).fit()
    out["gee_ar1"] = beta(gee) | {"rho": round(float(gee.cov_struct.dep_params), 3)}
    # per-severity Peer x session (paper eq. 4 split by level)
    out["by_severity"] = {s: beta(smf.mixedlm("ismi ~ peer*session", g, groups=g["trajectory_id"]).fit(reml=True))
                          for s, g in d.groupby("severity", observed=True)}
    # pooled-pre deltas and the bounded-scale check
    w = deltas(dp)
    sev = {}
    for s in SEV:
        p, g = w[(w.severity == s) & (w.condition == "peer")], w[(w.severity == s) & (w.condition == "general")]
        sev[s] = {"pre": round(p["pre"].mean(), 3), "delta_peer": round(p["delta"].mean(), 3),
                  "delta_general": round(g["delta"].mean(), 3),
                  "norm_peer": round(p["norm"].mean(), 3), "norm_general": round(g["norm"].mean(), 3)}
    out["severity_pooled"] = sev
    peer = w[w.condition == "peer"]
    fit = smf.ols("norm ~ C(severity, Treatment('mild'))", peer).fit()
    out["norm_peer_vs_mild"] = {k.split("[T.")[1].rstrip("]"): {"diff": round(float(fit.params[k]), 3),
                                                                "p": float(fit.pvalues[k])}
                                for k in fit.params.index if "T." in k}
    out["overall_pooled"] = {"delta_peer": round(w[w.condition == "peer"]["delta"].mean(), 3),
                             "delta_general": round(w[w.condition == "general"]["delta"].mean(), 3)}
    return out


res = {k: analyse(load(p)) for k, p in DATA.items()}
(Path(__file__).with_name("a2_sensitivity.json")).write_text(json.dumps(res, indent=1))
for k, r in res.items():
    print(f"== {k}")
    for m in ("intercept", "pooled_pre", "random_slope", "gee_ar1"):
        b = r[m]
        print(f"  {m:13s} beta {b['beta']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}] p {b['p']:.1e}"
              + (f" rho {b['rho']}" if "rho" in b else ""))
    for s, b in r["by_severity"].items():
        print(f"  sev {s:9s} beta {b['beta']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}] p {b['p']:.1e}")
    print(f"  pooled-pre delta  peer {r['overall_pooled']['delta_peer']:+.3f}  general {r['overall_pooled']['delta_general']:+.3f}")
    for s, v in r["severity_pooled"].items():
        print(f"  {s:9s} pre {v['pre']:.3f}  delta P {v['delta_peer']:+.3f} G {v['delta_general']:+.3f}"
              f"  norm P {v['norm_peer']:+.3f} G {v['norm_general']:+.3f}")
    print("  norm (peer) vs mild:", {s: f"{v['diff']:+.3f} p={v['p']:.3f}" for s, v in r["norm_peer_vs_mild"].items()})
