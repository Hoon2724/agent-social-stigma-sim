"""Oracle belief notes: what this measurement can register for each test dialogue.

No belief-update code is involved. Claude (Opus) reads each dialogue session by
session and keeps the patient's first-person note up to date, blind to the
trajectory type and the hidden targets, told only to follow the exchange
(harsher, unchanged or softer). The notes are then scored with the same
belief-only ISMI-29 judge (judge_notes.py). The resulting change per direction
is the range a good updater could reach on these dialogues under this
measurement -- the basis for fair, direction-specific criteria (user decision
2026-10-12, replacing the flat +-0.30 that assumed symmetric judge response).

Two evidence rules:
  all     -- anything in the week counts, the person's own replies included
             (how a remark landed is often only visible in their own words)
  others  -- only what other people said or did counts, as in the paper's
             belief update ("your own posts do not count as new evidence")
The gap between them shows whether that rule limits the reachable change.

Session 0 is the frozen pre text (testset/pre/<persona>@<start>.txt), as for
every impl.

  python3 oracle_notes.py --rule all --sets gold heldout reference --samples 2 1 1
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "oracle"

RULES = {
    "all": ("Base any change on what happened in this week's exchange, including how this person "
            "themselves responded in it -- their own replies are often the clearest sign of how "
            "something landed."),
    "others": ("Base any change only on what OTHER people said or did this week. This person's own "
               "posts and replies are not new evidence about them -- they already knew what they "
               "think; only another person's words or actions can move the note."),
}

PROMPT = """You keep a private first-person note for a person who posts anonymously online about how they are doing. The note says, in their own voice, how they see themselves right now: what they believe is true about them, what they let people know, and what they expect would happen if others knew.

[THE NOTE AS IT STOOD BEFORE THIS WEEK]
{prev}

[THIS WEEK, ONLINE]  ("me" is this person; "someone" is anyone else)
{week}

Update the note so it says where this person stands at the end of this week.
- {rule}
- A week can leave how they see themselves harsher, unchanged, or softer. Do not assume any of these -- follow the exchange. If nothing this week touched how they see themselves, return the note unchanged, word for word.
- What other people say about their own lives is not this person's experience; never write it as if it happened to this person.
- Keep it the same kind of note: plain first person, present tense, 5-7 sentences{cap}. Do not narrate the week as a story and do not compare with before (no "now", "starting to", "anymore", "than before"). No metaphors, no therapy or questionnaire language.

Return only JSON: {{"note": "<5-7 sentences>"}}"""

CAP = False

SCHEMA = {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"],
          "additionalProperties": False}


def render_week(session: dict) -> str:
    lines = []
    for t in session["turns"]:
        who = "me" if t["speaker"] == "patient" else "someone"
        lines.append(f"  {who}: {t['text'].strip()}")
    return "\n".join(lines)


def ask(prompt: str, model: str) -> str:
    env = {"HOME": os.environ["HOME"], "PATH": os.environ["PATH"], "LANG": "C.UTF-8"}
    cmd = ["claude", "-p", "--model", model, "--system-prompt", "You write JSON only.",
           "--tools", "", "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
           "--exclude-dynamic-system-prompt-sections", "--settings", '{"alwaysThinkingEnabled": false}',
           "--output-format", "json", "--json-schema", json.dumps(SCHEMA)]
    last = None
    for _ in range(3):
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=env, timeout=600)
        try:
            envelope = json.loads(proc.stdout)
            if envelope.get("is_error"):
                raise ValueError(envelope.get("result"))
            note = (envelope.get("structured_output") or json.loads(envelope["result"]))["note"].strip()
            if note:
                return note
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            last = exc
    raise RuntimeError(f"oracle call failed 3 times: {last}")


def run_item(item: dict, rule: str, sample: int, model: str) -> dict:
    c = item["condition"]
    pre = (HERE / "testset" / "pre" / f"{c['persona_id']}@{c['initial_ismi']}.txt").read_text().strip()
    notes = [pre]
    for session in item["visible_dialogue"]:
        cap = (f", at most {max(130, int(1.3 * len(pre.split())))} words" if CAP else "")
        notes.append(ask(PROMPT.format(prev=notes[-1], week=render_week(session), rule=RULES[rule], cap=cap), model))
    print(f"  {item['test_id']} rule={rule} sample={sample} done", flush=True)
    return {"test_id": item["test_id"], "rule": rule, "sample": sample, "notes": notes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=sorted(RULES), required=True)
    ap.add_argument("--sets", nargs="+", default=["gold", "heldout", "reference"])
    ap.add_argument("--samples", nargs="+", type=int, default=[2, 1, 1],
                    help="independent note sequences per item, one number per set")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cap", action="store_true",
                    help="length cap = max(130, 1.3 x session-0 words), same as oracle_impl.py; "
                         "output notes_<rule>_cap.json (checks whether the oracle rise is a length effect)")
    a = ap.parse_args()
    global CAP
    CAP = a.cap
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, n in zip(a.sets, a.samples):
        for line in (HERE / "testset" / f"{name}.jsonl").read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                jobs += [(item, s) for s in range(n)]
    out_path = OUT / (f"notes_{a.rule}_cap.json" if a.cap else f"notes_{a.rule}.json")
    done = json.loads(out_path.read_text()) if out_path.exists() else []
    have = {(d["test_id"], d["sample"]) for d in done}
    todo = [(i, s) for i, s in jobs if (i["test_id"], s) not in have]
    print(f"rule={a.rule}: {len(todo)} sequences to write ({len(have)} already done)", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as pool:
        for result in pool.map(lambda js: run_item(js[0], a.rule, js[1], a.model), todo):
            done.append(result)
            out_path.write_text(json.dumps(done, indent=1, ensure_ascii=False) + "\n")
    print(f"-> {out_path} ({len(done)} sequences)")


if __name__ == "__main__":
    main()
