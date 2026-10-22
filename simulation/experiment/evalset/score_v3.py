"""Score belief-update results on dataset v3 (plan docs/superpowers/plans/2026-10-13-belief-eval-dataset.md §3).

Input: a directory of per-dialogue result JSONs in the harness format -- written by
harness.py (a belief-update module) or by judge_notes.py (any method's note sequences).
Change per run = S8 judged mean - the pooled mean of that exact session-0 note
(testset_v3/s0_pooled.json, or SEALED/s0_pooled.json for the test split).

Main metric (one number, lower is better):
  E = mean over dialogues of |mean change - target change|   (targets: +0.5 / -0.5 / 0)
Guards (a method that breaks one fails, whatever its E):
  reinforcing: mean change > 0 in each reinforcing cell
  reducing:    mean change < 0 in each reducing cell
  neutral:     |mean change| <= 0.15 in each of the four neutral cells (dry 2.5/3.0, general-like 2.5/3.0)
Reported separately: drift on dry neutral ("no-signal drift") and on general-like neutral
("over-reaction to sympathy").

  python3 score_v3.py results/v3_runs/base_paper [--json out.json] [--split test]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import v3data

NEUTRAL_MAX = 0.15
KIND_LABEL = {"stigma_reinforcing": "악화", "stigma_reducing": "감소",
              "neutral_dry": "유지·건조", "general_like": "유지·General형"}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def pooled(split: str) -> dict:
    p = (v3data.SEALED if split == "test" else v3data.DEV_DIR) / "s0_pooled.json"
    return json.loads(p.read_text()) if p.exists() else {}


def item_metrics(r: dict, snap: dict) -> dict:
    s0 = [x for run in r["runs"] for x in run["sessions"] if x["session"] == 0]
    base = snap.get(s0[0].get("text_sha"), {}).get("mean") if s0 else None
    if base is None:  # fall back to this file's own session-0 judgments
        base = mean([v for x in s0 for v in (x.get("per_rep") or [x.get("ismi")])])
    deltas, opens, words = [], [], []
    for run in r["runs"]:
        s = sorted(run["sessions"], key=lambda x: x["session"])
        if s[-1].get("ismi") is not None and base is not None:
            deltas.append(s[-1]["ismi"] - base)
        post = s[1:]
        opens.append(sum(bool(x.get("changed")) for x in post) / len(post) if post else 0.0)
        words.append(s[-1].get("words"))
    d = mean(deltas)
    target = round(r["target_final_ismi"] - r["initial_ismi"], 3)
    return {"test_id": r["test_id"], "kind": r["trajectory_type"], "start": r["initial_ismi"],
            "cell": r["test_id"].split("_")[0], "s0": base, "delta": d, "delta_runs": deltas,
            "target": target, "err": abs(d - target) if d is not None else None,
            "changed_share": mean(opens), "final_words": mean(words), "n_runs": len(deltas)}


def summarize(result_dir: Path, split: str = "dev") -> dict:
    snap = pooled(split)
    results = [json.loads(p.read_text()) for p in sorted(result_dir.glob("*.json")) if not p.name.startswith("_")]
    items = [item_metrics(r, snap) for r in results]
    expected = {x["test_id"] for x in v3data.items(split)}
    missing = sorted(expected - {i["test_id"] for i in items})
    cells = {}
    for i in items:
        cells.setdefault((i["cell"], i["kind"], i["start"]), []).append(i)
    guards, cell_rows = {}, []
    for (cell, kind, start), rows in sorted(cells.items()):
        m = mean([r["delta"] for r in rows])
        if kind == "stigma_reinforcing":
            ok = m is not None and m > 0
        elif kind == "stigma_reducing":
            ok = m is not None and m < 0
        else:
            ok = m is not None and abs(m) <= NEUTRAL_MAX
        guards[cell] = ok
        cell_rows.append({"cell": cell, "kind": kind, "start": start, "mean_delta": m, "ok": ok,
                          "n": len(rows)})
    kinds = {}
    for k in v3data.KINDS:
        rows = [i for i in items if i["kind"] == k]
        if rows:
            kinds[k] = {"mean_delta": mean([r["delta"] for r in rows]), "mean_err": mean([r["err"] for r in rows]),
                        "changed_share": mean([r["changed_share"] for r in rows])}
    E = mean([i["err"] for i in items])
    return {"E": E, "guards_pass": bool(guards) and all(guards.values()) and not missing,
            "missing": missing, "cells": cell_rows, "kinds": kinds, "items": items}


def fmt(x, nd=3, sign=True):
    return "—" if x is None else (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}")


def report(name: str, s: dict) -> str:
    out = [f"## {name}", "", f"**E = {fmt(s['E'], sign=False)}** (낮을수록 좋음) · 지킴 조건 "
           f"{'통과' if s['guards_pass'] else '**실패**'}" + (f" · 빠진 대화 {s['missing']}" if s["missing"] else ""), "",
           "| 칸 | 유형 | 출발 | 평균 Δ | 조건 |", "|---|---|---|---|---|"]
    for c in s["cells"]:
        out.append(f"| {c['cell']} | {KIND_LABEL.get(c['kind'], c['kind'])} | {c['start']} | {fmt(c['mean_delta'])} | "
                   f"{'ok' if c['ok'] else '**FAIL**'} |")
    out += ["", "| 유형 | 평균 Δ | 평균 오차 | 노트 바뀐 세션 비율 |", "|---|---|---|---|"]
    for k, v in s["kinds"].items():
        out.append(f"| {KIND_LABEL.get(k, k)} | {fmt(v['mean_delta'])} | {fmt(v['mean_err'], sign=False)} | "
                   f"{fmt(v['changed_share'], 2, False)} |")
    out += ["", "| 대화 | 세션0 실측 | 목표 Δ | Δ | 런별 Δ | 오차 | 최종 단어 |", "|---|---|---|---|---|---|---|"]
    for i in s["items"]:
        out.append(f"| {i['test_id']} | {fmt(i['s0'], 2, False)} | {i['target']:+.1f} | {fmt(i['delta'])} | "
                   f"{', '.join(f'{d:+.2f}' for d in i['delta_runs'])} | {fmt(i['err'], sign=False)} | "
                   f"{fmt(i['final_words'], 0, False)} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("--compare", type=Path, nargs="*", default=[])
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    s = summarize(a.result_dir, a.split)
    print(report(str(a.result_dir), s))
    for other in a.compare:
        print(report(str(other), summarize(other, a.split)))
    if a.json:
        a.json.write_text(json.dumps(s, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
