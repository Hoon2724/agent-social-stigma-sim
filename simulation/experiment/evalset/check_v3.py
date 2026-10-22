"""Acceptance checks for dataset v3 dialogues (plan P3) and assembly of the final sets.

  direction  blind relative direction rating by Sonnet (direction_check.py prompt and rule)
  oracle     2 Claude Opus ideal-note sequences per dialogue ("others" rule, length cap,
             oracle_notes.py prompt), starting from the frozen session-0 note
  judge      the paper judge (qwen3:32b, 3 reps) on oracle notes S4 and S8;
             change = S8 - pooled session-0 mean
  review     a separate Claude reader returns pass/fail and problem codes only (no quotes)
  summary    one row per attempt; --assemble writes the accepted attempts to
             testset_v3/dev.jsonl or SEALED/test.jsonl
  failing    ids to regenerate: every attempt checked, none accepted (or never generated)

Every subcommand works per split. For --split test all outputs go under SEALED/check and
only ids, numbers and codes are printed (user decision 2026-10-13: nobody reads the test
dialogues before the final comparison).

  python3 check_v3.py direction --split dev
  python3 check_v3.py oracle --split dev
  python3 check_v3.py judge --split dev --base-url http://localhost:11888/v1
  python3 check_v3.py review --split dev
  python3 check_v3.py summary --split dev [--assemble]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import subprocess
import threading
from pathlib import Path

import direction_check as dc
import gen_v3
import oracle_notes as on
import v3data

HERE = Path(__file__).resolve().parent
ORACLE_MIN = 0.25      # |oracle change| a directional dialogue must reach
NEUTRAL_MAX = 0.15     # |oracle change| a neutral dialogue may show
REVIEW_CODES = {
    "reinforcing_not_cold": "a reinforcing commenter is not consistently cold/dismissive or turns supportive",
    "reducing_not_peer": "a reducing peer does not actually push back on the self-stigma (or lectures/counsels)",
    "neutral_carries_evidence": "a neutral commenter says something that supports or argues against how the poster sees themselves",
    "general_like_off_register": "a general-like commenter is not brief stock sympathy (too long, advice, own story, reassurance about worth/effort/future)",
    "dry_off_register": "a dry commenter is warm, validating or steering rather than brief and plain",
    "patient_off_trajectory": "the poster's own self-view does not move the intended way across weeks (or moves when it should stay)",
    "abrupt_change": "the poster's self-view jumps within a week or between weeks instead of shifting gradually",
    "incoherent": "continuity breaks, invented backreferences, contradictions, repeated lines",
    "unrealistic": "does not read like an anonymous support-forum thread",
    "leak": "mentions scores, scales, conditions or research language",
}
_lock = threading.Lock()


def check_dir(split: str) -> Path:
    d = (v3data.SEALED / "check") if split == "test" else (HERE / "results" / "v3_check")
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_json(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def save_json(p: Path, data) -> None:
    with _lock:
        p.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


def key(item: dict) -> str:
    return f"{item['test_id']}#a{item['attempt']}"


def attempts(split: str, ids: list[str] | None) -> list[dict]:
    rows = gen_v3.load(split)
    return [r for r in rows if not ids or r["test_id"] in ids]


def s0_text(item: dict) -> str:
    c = item["condition"]
    return gen_v3.s0_note(c["persona_id"], c["initial_ismi"])


def s0_mean(item: dict) -> float:
    import harness
    split = item["split"]
    pooled = load_json((v3data.SEALED if split == "test" else v3data.DEV_DIR) / "s0_pooled.json", {})
    return pooled[harness.text_hash(s0_text(item))]["mean"]


# ------------------------------------------------------------------ direction

def cmd_direction(a) -> None:
    path = check_dir(a.split) / "direction.json"
    res = load_json(path, {})
    todo = [it for it in attempts(a.split, a.ids) if key(it) not in res]

    def one(it: dict) -> str:
        ratings, whys = [0], ["week 1 is the reference"]
        for sess in it["visible_dialogue"][1:]:
            r = dc.ask(a.model, dc.PROMPT.format(thread=dc.render(it, sess["session"]), week=sess["session"]))
            ratings.append(max(-3, min(3, r["rating"])))
            whys.append(r.get("why", ""))
        ok = dc.verdict(v3data.GEN_KIND[it["condition"]["trajectory_type"]], ratings)
        res[key(it)] = {"ratings": ratings, "ok": ok, "why": whys, "rater": a.model}
        save_json(path, res)
        return f"  {key(it):34s} {it['condition']['trajectory_type']:18s} {ratings} -> {'OK' if ok else 'FAIL'}"

    print(f"direction: {len(todo)} to rate", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as pool:
        for line in pool.map(one, todo):
            print(line, flush=True)


# ------------------------------------------------------------------ oracle

def cmd_oracle(a) -> None:
    path = check_dir(a.split) / "oracle_notes.json"
    res = load_json(path, {})
    jobs = [(it, s) for it in attempts(a.split, a.ids) for s in range(a.samples)
            if f"{key(it)}/s{s}" not in res]

    def one(job) -> str:
        it, s = job
        pre = s0_text(it)
        cap = f", at most {max(130, int(1.3 * len(pre.split())))} words"
        notes = [pre]
        for sess in it["visible_dialogue"]:
            notes.append(on.ask(on.PROMPT.format(prev=notes[-1], week=on.render_week(sess),
                                                 rule=on.RULES["others"], cap=cap), a.model))
        res[f"{key(it)}/s{s}"] = {"notes": notes, "rule": "others_cap", "model": a.model}
        save_json(path, res)
        return f"  {key(it)} sample {s} written"

    print(f"oracle: {len(jobs)} sequences to write", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as pool:
        for line in pool.map(one, jobs):
            print(line, flush=True)


def cmd_judge(a) -> None:
    import harness
    from llm_client import OpenAICompatibleModel
    from measure_ismi29 import build_pre_innerstate_block, measure_once
    notes = load_json(check_dir(a.split) / "oracle_notes.json", {})
    path = check_dir(a.split) / "oracle_judged.json"
    res = load_json(path, {})
    judge = OpenAICompatibleModel("qwen3:32b", a.base_url, num_ctx=8192, num_gpu=99)
    by_item = {key(it): it for it in attempts(a.split, a.ids)}
    todo = [k for k in notes if k not in res and k.split("/")[0] in by_item]

    def one(k: str) -> str:
        seq = notes[k]["notes"]
        out = {}
        for s in (4, 8):
            out[f"S{s}"] = harness.judge_text(judge, seq[s], 3, 0.3, (measure_once, build_pre_innerstate_block))["ismi"]
        out["s0_pooled"] = s0_mean(by_item[k.split("/")[0]])
        out["delta"] = round(out["S8"] - out["s0_pooled"], 3) if out["S8"] is not None else None
        res[k] = out
        save_json(path, res)
        return f"  {k:38s} S0 {out['s0_pooled']:.2f}  S4 {out['S4']}  S8 {out['S8']}  delta {out['delta']:+.2f}"

    print(f"judge: {len(todo)} sequences", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as pool:
        for line in pool.map(one, todo):
            print(line, flush=True)


# ------------------------------------------------------------------ review

REVIEW_PROMPT = """You are checking one synthetic dataset item before it is used to test a model. It is an anonymous mental-health support thread over 8 weeks: one POSTER, and one other person who replies.

What this item is supposed to be:
- Type: {kind_desc}
- The poster starts at a {start_desc} level of self-stigma (treating their own struggling as a personal defect, expecting to be looked down on, seeing help-seeking as shameful) and should end {end_desc}. The shift, if any, is gradual across weeks.

Problem codes (use only these):
{codes}

THREAD:
{thread}

Read the whole thread. Decide whether it is fit for purpose. Minor stylistic imperfections are fine; flag a problem only when it would make the item a misleading test. Return ONLY JSON: {{"pass": true|false, "problems": [<codes>], "severity": "none"|"minor"|"major"}}. Do not quote the thread."""

KIND_DESC = {
    "stigma_reinforcing": "the replier is a cold, dismissive anonymous commenter (no own story) whose remarks make the poster's self-stigma land harder",
    "stigma_reducing": "the replier is a peer with lived experience who actively steers the poster toward less self-stigma (own story, questions, naming what the poster already does)",
    "neutral_dry": "the replier is a passing stranger leaving brief, plain, uninvested remarks; nothing they say bears on how the poster sees themselves",
    "general_like": "the replier is a passing stranger leaving brief stock sympathy ('that sounds tough', 'hope you get some rest'); nothing they say supports or argues against how the poster sees themselves",
}
def level_word(x: float) -> str:
    return "mild" if x < 2.4 else ("moderate" if x < 2.9 else "fairly high")
REVIEW_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["pass", "problems", "severity"],
                 "properties": {"pass": {"type": "boolean"},
                                "problems": {"type": "array", "items": {"type": "string", "enum": sorted(REVIEW_CODES)}},
                                "severity": {"type": "string", "enum": ["none", "minor", "major"]}}}


def cmd_review(a) -> None:
    path = check_dir(a.split) / "review.json"
    res = load_json(path, {})
    todo = [it for it in attempts(a.split, a.ids) if key(it) not in res]
    env = {"HOME": os.environ["HOME"], "PATH": os.environ["PATH"], "LANG": "C.UTF-8"}
    cmd = ["claude", "-p", "--model", a.model, "--system-prompt", "You check datasets. Output JSON only.",
           "--tools", "", "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
           "--exclude-dynamic-system-prompt-sections", "--settings", '{"alwaysThinkingEnabled": false}',
           "--output-format", "json", "--json-schema", json.dumps(REVIEW_SCHEMA)]

    def one(it: dict) -> str:
        c = it["condition"]
        kind, start, final = c["trajectory_type"], c["initial_ismi"], c["target_final_ismi"]
        a0 = c.get("target_anchor") or start          # measured session-0 level (v3.1), else nominal
        a1 = a0 + (final - start)
        end = ("at about the same level" if final == start else
               f"{'higher' if final > start else 'lower'} -- at a {level_word(a1)} level")
        prompt = REVIEW_PROMPT.format(kind_desc=KIND_DESC[kind], start_desc=level_word(a0), end_desc=end,
                                      codes="\n".join(f"- {k}: {v}" for k, v in REVIEW_CODES.items()),
                                      thread=dc.render(it, 99))
        out = None
        for _ in range(3):
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=env, timeout=900)
            try:
                e = json.loads(proc.stdout)
                out = e.get("structured_output") or json.loads(e["result"])
                break
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
        if out is None:
            return f"  {key(it)} review FAILED"
        res[key(it)] = {**out, "model": a.model}
        save_json(path, res)
        return f"  {key(it):34s} {'PASS' if out['pass'] else 'FAIL'} {out['severity']} {out['problems']}"

    print(f"review: {len(todo)} to read", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as pool:
        for line in pool.map(one, todo):
            print(line, flush=True)


# ------------------------------------------------------------------ summary

def verdicts(a) -> list[dict]:
    d = check_dir(a.split)
    direc, judged, review = (load_json(d / f, {}) for f in ("direction.json", "oracle_judged.json", "review.json"))
    rows = []
    for it in attempts(a.split, a.ids):
        k, kind = key(it), it["condition"]["trajectory_type"]
        deltas = [v["delta"] for kk, v in judged.items() if kk.split("/")[0] == k and v.get("delta") is not None]
        od = statistics.fmean(deltas) if deltas else None
        if od is None:
            o_ok = None
        elif kind == "stigma_reinforcing":
            o_ok = od >= ORACLE_MIN
        elif kind == "stigma_reducing":
            o_ok = od <= -ORACLE_MIN
        else:
            o_ok = abs(od) <= NEUTRAL_MAX
        q = it["qc"]
        qc_ok = not q["schema_problems"] and not q["leak_hits"] and not q["ungrounded_quotes"]
        dr, rv = direc.get(k), review.get(k)
        checks = [qc_ok, dr and dr["ok"], o_ok, rv and rv["pass"]]
        rows.append({"key": k, "test_id": it["test_id"], "attempt": it["attempt"], "cell": it["cell"],
                     "kind": kind, "qc": qc_ok, "direction": dr["ratings"] if dr else None,
                     "direction_ok": dr["ok"] if dr else None, "oracle_deltas": deltas, "oracle_delta": od,
                     "oracle_ok": o_ok, "review_ok": rv["pass"] if rv else None,
                     "review_problems": rv["problems"] if rv else None,
                     "accepted": all(x is True for x in checks), "complete": all(x is not None for x in checks)})
    return rows


def cmd_summary(a) -> None:
    rows = verdicts(a)
    f = lambda x: "—" if x is None else ("ok" if x is True else ("FAIL" if x is False else x))  # noqa: E731
    print("| dialogue | type | qc | direction | oracle Δ (2 samples) | review | accepted |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        od = "—" if r["oracle_delta"] is None else f"{r['oracle_delta']:+.2f} ({', '.join(f'{x:+.2f}' for x in r['oracle_deltas'])}) {f(r['oracle_ok'])}"
        rv = f(r["review_ok"]) + (f" {r['review_problems']}" if r["review_problems"] else "")
        print(f"| {r['key']} | {r['kind']} | {f(r['qc'])} | {r['direction']} {f(r['direction_ok'])} | {od} | {rv} | "
              f"{'**yes**' if r['accepted'] else ('no' if r['complete'] else 'pending')} |")
    if a.assemble:
        items = {key(it): it for it in attempts(a.split, None)}
        keep = {}
        for r in rows:
            if r["accepted"]:
                keep[r["test_id"]] = items[r["key"]]   # latest accepted attempt wins
        # User exceptions (10-14): an attempt kept although a check failed; recorded in EXCEPTIONS.json.
        exc_path = (v3data.SEALED if a.split == "test" else v3data.DEV_DIR) / "EXCEPTIONS.json"
        exc = load_json(exc_path, {})
        for k in a.include or []:
            tid = k.split("#")[0]
            if tid in keep:
                raise SystemExit(f"{tid} already has an accepted attempt -- no exception needed")
            row = next(r for r in rows if r["key"] == k)
            keep[tid] = items[k]
            exc[tid] = {"key": k, "reason": a.reason or "", "failed": [c for c in ("qc", "direction_ok", "oracle_ok", "review_ok")
                                                                        if row[c] is not True]}
        if a.include:
            save_json(exc_path, exc)
        order = [x["test_id"] for x in v3data.items(a.split)]
        out = (v3data.SEALED / "test.jsonl") if a.split == "test" else (v3data.DEV_DIR / "dev.jsonl")
        out.write_text("".join(json.dumps(keep[t], ensure_ascii=False) + "\n" for t in order if t in keep))
        missing = [t for t in order if t not in keep]
        print(f"\nassembled {len(keep)}/{len(order)} -> {out}" + (f"; missing {missing}" if missing else ""))


def cmd_failing(a) -> None:
    """test_ids whose every attempt is fully checked and none accepted (to regenerate)."""
    rows = verdicts(a)
    ids = sorted({r["test_id"] for r in rows}
                 - {r["test_id"] for r in rows if r["accepted"] or not r["complete"]})
    missing = sorted({x["test_id"] for x in v3data.items(a.split)} - {r["test_id"] for r in rows})
    print(" ".join(ids + missing))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("direction", "oracle", "judge", "review", "summary", "failing"))
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--model")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-url")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--include", nargs="*", help="with --assemble: attempt keys (id#aN) kept as user exceptions")
    ap.add_argument("--reason", help="with --include: why (written to EXCEPTIONS.json)")
    a = ap.parse_args()
    # 10-13 22:50: oracle and review moved to Sonnet to save Max usage (generation stays Opus)
    a.model = a.model or {"direction": "sonnet", "oracle": "sonnet", "review": "sonnet"}.get(a.cmd)
    {"direction": cmd_direction, "oracle": cmd_oracle, "judge": cmd_judge,
     "review": cmd_review, "summary": cmd_summary, "failing": cmd_failing}[a.cmd](a)


if __name__ == "__main__":
    main()
