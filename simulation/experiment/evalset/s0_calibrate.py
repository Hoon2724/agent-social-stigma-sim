"""Session-0 note calibration for dataset v3 (plan P1). One person@start per call.

The nominal start (2.0 / 2.5 / 3.0) is only a generation instruction; the same judge read
earlier session-0 notes up to 1.1 points away from it. So candidates are written with the
paper's PRE_PROMPT (harness.PRE_PROMPT, qwen3:14b, T0.7 -- the updater's own model, so the
first update is not a change of writing style) and the one whose judged mean is closest to
the nominal start is kept:
  8 candidates: start 2.0 -> 8 with the 2.0 register (mild is already the lowest template);
                2.5 / 3.0 -> 4 with their own register + 4 with the register 0.25 lower
                (judge readings of earlier notes ran high, not low)
  6 judge reps each -> 3 closest get 12 more -> closest of those gets 12 more (30 in all).
Registers come from registers.json (sanitychecking2 _register_for, precomputed).

Output: one JSON per person@start in --out-dir (texts included). For test-split people the
out dir must be under SEALED; only numbers are printed either way.

  python3 s0_calibrate.py p_patient 2.5 --base-url http://localhost:11434/v1 --out-dir ~/belief_v2_runs/v3_s0
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import harness  # puts paper_rq1 on sys.path
import v3data

TOL = 0.15


def register_plan(start: float) -> list[float]:
    if start <= 2.0:
        return [start] * 8
    return [start] * 4 + [start - 0.25] * 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("persona_id")
    ap.add_argument("start", type=float)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--judge-model", default="qwen3:32b")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    if v3data.is_sealed(a.persona_id) and v3data.SEALED.resolve() not in a.out_dir.resolve().parents \
            and a.out_dir.resolve() != v3data.SEALED.resolve():
        raise SystemExit("test-split person: --out-dir must be under SEALED")

    from llm_client import OpenAICompatibleModel
    from measure_ismi29 import build_pre_innerstate_block, measure_once
    from strategy_selector import complete_json

    persona = v3data.personas()[a.persona_id]
    registers = json.loads((v3data.DEV_DIR / "registers.json").read_text())["registers"]
    gen = OpenAICompatibleModel(a.model, a.base_url, num_ctx=a.num_ctx, num_gpu=99)
    judge = OpenAICompatibleModel(a.judge_model, a.base_url, num_ctx=a.num_ctx, num_gpu=99)
    key = v3data.pre_key(a.persona_id, a.start)

    cands = []
    for reg in register_plan(a.start):
        prompt = harness.PRE_PROMPT.format(
            facts=persona["diagnosis_context"],
            intensity_hint="How this currently sits with you:\n" + registers[str(reg)])
        text = ""
        for _ in range(3):
            try:
                text = str(complete_json(gen, [{"role": "user", "content": prompt}],
                                         temperature=0.7).get("self_view", "")).strip()
            except (ValueError, KeyError, TypeError):
                text = ""
            if text:
                break
        if text:
            cands.append({"register": reg, "text": text, "per_rep": []})
    print(f"[{key}] {len(cands)} candidates written", flush=True)

    def judge_more(c: dict, n: int) -> None:
        c["per_rep"] += harness.judge_text(judge, c["text"], n, 0.3,
                                           (measure_once, build_pre_innerstate_block))["per_rep"]
        c["mean"] = round(statistics.fmean(c["per_rep"]), 4) if c["per_rep"] else None

    def dist(c: dict) -> float:
        return abs(c["mean"] - a.start) if c.get("mean") is not None else 9.0

    for c in cands:
        judge_more(c, 6)
    print(f"[{key}] round1 " + " ".join(f"{c['register']}:{c['mean']}" for c in cands), flush=True)
    top = sorted(cands, key=dist)[:3]
    for c in top:
        judge_more(c, 12)
    best = min(top, key=dist)
    judge_more(best, 12)
    best["chosen"] = True
    sd = statistics.stdev(best["per_rep"]) if len(best["per_rep"]) > 1 else 0.0
    ok = dist(best) <= TOL
    print(f"[{key}] round2 " + " ".join(f"{c['register']}:{c['mean']}(n={len(c['per_rep'])})" for c in top), flush=True)
    print(f"[{key}] CHOSEN mean={best['mean']} sd={sd:.3f} n={len(best['per_rep'])} "
          f"nominal={a.start} diff={best['mean'] - a.start:+.3f} within_tol={ok}", flush=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"{key}.json").write_text(json.dumps({
        "key": key, "persona_id": a.persona_id, "start": a.start, "tol": TOL, "within_tol": ok,
        "gen_model": a.model, "judge_model": a.judge_model, "judge_temperature": 0.3,
        "chosen_mean": best["mean"], "chosen_sd": round(sd, 4), "chosen_n": len(best["per_rep"]),
        "candidates": cands}, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
