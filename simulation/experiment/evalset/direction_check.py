"""Manipulation check for test dialogues: does the visible text actually carry
the intended direction? A rater that did NOT write the dialogue (default
Claude Sonnet; the dialogues were written by Opus or GPT-5.5) reads the
thread with no trajectory label, no target and no persona sheet.

Method "relative" (2026-10-11, replaces an absolute 1-7 rating): for each
week s >= 2 the rater sees weeks 1..s and says how the poster's self-stigma at
the end of week s compares with the end of week 1, from -3 (much softer) to
+3 (much harsher); week 1 is 0 by definition. The absolute version hit its
ceiling -- every dialogue starting at ISMI 3.0 was rated 7 from week 1, so a
reinforcing dialogue could not register any rise (loop BLOCKED 10-11 14:14).

Pass rule per item: reducing -> final <= -1; reinforcing -> final >= +1;
neutral -> |final| <= 1 and every week within +-1. A dialogue that fails is
not a valid test of the belief update, whatever the update does with it.

  python3 direction_check.py testset/claude_dev.jsonl [--model sonnet] [--out results/direction_check/dev.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

PROMPT = """Below is an anonymous mental-health support thread in which one person (POSTER) posts over several weeks and another person replies.

{thread}

Look at the POSTER only. "Self-stigma" here means treating their own struggling as a personal defect -- seeing themselves as weak, broken, lazy or less worthy because of it, expecting others to look down on them, and seeing asking for help as shameful or pointless.

Compare how the poster sees themselves at the END of week {week} with how they saw themselves at the END of week 1:
-3 = much softer (clearly less self-stigma than in week 1)
-1 = slightly softer
 0 = about the same
+1 = slightly harsher
+3 = much harsher (clearly more self-stigma than in week 1)

Judge from what the poster writes, not from what the other person says. Return ONLY a JSON object: {{"rating": <integer -3 to 3>, "why": "<one short sentence>"}}"""


def render(item: dict, upto: int) -> str:
    blocks = []
    for sess in item["visible_dialogue"]:
        if sess["session"] > upto:
            break
        lines = [f"--- week {sess['session']} ---"]
        for t in sess["turns"]:
            who = "POSTER" if t["speaker"] == "patient" else "REPLY"
            lines.append(f"{who}: {t['text']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def ask(model: str, prompt: str) -> dict:
    env = {"HOME": os.environ["HOME"], "PATH": os.environ["PATH"], "LANG": "C.UTF-8"}
    cmd = ["claude", "-p", "--model", model, "--system-prompt", "You rate text. Output JSON only.",
           "--tools", "", "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
           "--exclude-dynamic-system-prompt-sections", "--settings", '{"alwaysThinkingEnabled": false}',
           "--output-format", "json"]
    for _ in range(3):
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=env, timeout=600)
        try:
            env_out = json.loads(proc.stdout)
            m = re.search(r"\{.*\}", env_out["result"], re.S)
            out = json.loads(m.group(0))
            out["rating"] = int(out["rating"])
            out["_models"] = sorted(env_out.get("modelUsage", {}).keys())
            return out
        except (ValueError, KeyError, AttributeError, json.JSONDecodeError):
            continue
    raise RuntimeError("rater failed 3 times")


def verdict(kind: str, ratings: list[int]) -> bool:
    """ratings[0] is week 1 (= 0); the rest are relative to it."""
    last = ratings[-1]
    if kind == "stigma_reducing":
        return last <= -1
    if kind == "stigma_reinforcing":
        return last >= 1
    return abs(last) <= 1 and all(abs(r) <= 1 for r in ratings)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("testset", type=Path)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rows = []
    for line in a.testset.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        tid = item.get("test_id") or item["sample_id"]
        if a.ids and tid not in a.ids:
            continue
        kind = item["condition"]["trajectory_type"]
        ratings, whys = [0], ["week 1 is the reference"]
        for sess in item["visible_dialogue"][1:]:
            r = ask(a.model, PROMPT.format(thread=render(item, sess["session"]), week=sess["session"]))
            ratings.append(max(-3, min(3, r["rating"])))
            whys.append(r.get("why", ""))
        ok = verdict(kind, ratings)
        targets = [t["target_ismi"] for t in item["hidden_ground_truth"]["session_targets"]]
        print(f"{tid:44s} {kind:18s} ratings {ratings}  targets {targets}  -> {'OK' if ok else 'FAIL'}", flush=True)
        rows.append({"id": tid, "kind": kind, "ratings": ratings, "targets": targets, "ok": ok,
                     "why": whys, "rater": a.model, "method": "relative"})
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
