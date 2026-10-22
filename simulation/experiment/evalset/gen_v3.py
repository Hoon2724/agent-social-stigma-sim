"""Generate dataset v3 dialogues with Claude Opus (plan P2).

Same v17 sequential pipeline as gen_claude.py (sanitychecking2 prompts, one session per
call, the previous session verbatim as context, --json-schema structured output), with the
v3 changes from v3data.patch_prompt: +-0.5 final targets, the general_like commenter, and
the person's frozen session-0 note in the session-1 prompt.

Every attempt is appended (never overwritten) with an attempt number; check_v3.py decides
which attempt is kept. Dev dialogues go to testset_v3/candidates_dev.jsonl; test dialogues
go to SEALED/candidates_test.jsonl and nothing from them is printed beyond ids and qc counts.

  python3 gen_v3.py --split dev [--only C1_p_patient ...] [--workers 4]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import threading
from pathlib import Path

import gen_claude  # noqa: F401  (puts sanitychecking2 and the persona_simple shim on sys.path)
from gen_claude import ClaudeClient

import generate  # sanitychecking2
import prompts
import trajectories
from schema import SESSION_OUTPUT_JSON_SCHEMA, find_leak_terms, find_ungrounded_quotes, validate_item

import v3data

TURNS = 11
SESSIONS = 8
_lock = threading.Lock()


def out_path(split: str) -> Path:
    return (v3data.SEALED / "candidates_test.jsonl") if split == "test" else (v3data.DEV_DIR / "candidates_dev.jsonl")


def load(split: str) -> list[dict]:
    p = out_path(split)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def s0_note(pid: str, start: float) -> str:
    p = v3data.pre_dir(pid) / f"{v3data.pre_key(pid, start)}.txt"
    if not p.exists():
        raise SystemExit(f"{p} missing -- run P1 (s0_calibrate + s0_select) first")
    return p.read_text().strip()


def session_targets(kind: str, start: float, anchor: float | None = None) -> list[dict]:
    """Hidden per-session targets. With anchor (the measured session-0 mean, rounded to 0.1) the
    trajectory starts where the judge actually reads the person's note and keeps the cell's change
    (+-0.5 / 0), so the patient's session-1 voice matches the note (10-13: a 2.0-nominal note read
    as 2.60 was paired with a mild-register patient and the reinforcing dialogue measured downward)."""
    gk = v3data.GEN_KIND[kind]
    change = v3data.FINAL[kind][start] - start
    s = float(start) if anchor is None else round(float(anchor), 1)
    trajectories._FINAL_ISMI.setdefault(gk, {})[s] = round(s + change, 1)
    return trajectories.build_session_targets(gk, s, SESSIONS)


def s0_anchor(pid: str, start: float) -> float | None:
    import harness
    pooled_path = (v3data.SEALED if v3data.is_sealed(pid) else v3data.DEV_DIR) / "s0_pooled.json"
    pooled = json.loads(pooled_path.read_text()) if pooled_path.exists() else {}
    m = pooled.get(harness.text_hash(s0_note(pid, start)), {}).get("mean")
    return None if m is None else round(m, 1)


def generate_one(row: dict, persona: dict, client: ClaudeClient, attempt: int) -> dict:
    kind, start = row["kind"], row["start"]
    gk = v3data.GEN_KIND[kind]
    note = s0_note(row["persona_id"], start)
    anchor = s0_anchor(row["persona_id"], start)
    targets = session_targets(kind, start, anchor)
    spec = {"trajectory_type": gk, "initial_ismi": targets[0]["target_ismi"],
            "target_final_ismi": targets[-1]["target_ismi"],
            "session_count": SESSIONS, "turns_per_session": TURNS, "persona_id": row["persona_id"]}
    cond = generate._condition_block(spec, persona)
    if kind == "general_like":
        cond["peer_style"] = "general_like_commenter"
    dry = prompts._PEER_INSTRUCTIONS["general_neutral"]
    dialogue, markers, prior = [], [], None
    for i, target in enumerate(targets):
        user = prompts.build_session_user_prompt(cond, persona, targets, i, prior)
        user = v3data.patch_prompt(user, kind, i, note, dry)
        out = client.complete_json(prompts.build_system_prompt(), user, SESSION_OUTPUT_JSON_SCHEMA)
        dialogue.append({"session": target["session"], "turns": out["turns"]})
        markers.append({"session": target["session"], "evidence": out["evidence"]})
        prior = out["turns"]
    item = {
        "dataset_id": "belief_v3", "test_id": row["test_id"], "cell": row["cell"], "split": row["split"],
        "attempt": attempt, "prompt_version": v3data.PROMPT_TAG,
        "model": f"claude:{client.model}",
        "generation_params": {"temperature": "cli-default", "resolved_models": sorted(client.used_models)},
        "condition": {**cond, "trajectory_type": kind, "gen_trajectory_type": gk,
                      "initial_ismi": start, "target_final_ismi": row["final"], "target_anchor": anchor,
                      "s0_key": v3data.pre_key(row["persona_id"], start)},
        "hidden_ground_truth": {"session_targets": targets},
        "visible_dialogue": dialogue,
        "author_evidence": {"markers_by_session": markers},
    }
    # validate_item knows only the v17 type names; check a copy under the generator's name.
    probe = json.loads(json.dumps(item))
    probe["condition"]["trajectory_type"] = gk
    probe["schema_version"] = "1.0"
    for k in ("condition_id", "sample_id", "seed", "created_at"):
        probe.setdefault(k, None)
    item["qc"] = {"schema_problems": validate_item(probe), "leak_hits": find_leak_terms(probe),
                  "ungrounded_quotes": find_ungrounded_quotes(probe)}
    return item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--only", nargs="*", help="test_ids to (re)generate as a new attempt")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    rows = v3data.items(a.split)
    have = load(a.split)
    if a.only:
        unknown = set(a.only) - {r["test_id"] for r in rows}
        if unknown:
            raise SystemExit(f"unknown test_ids: {sorted(unknown)}")
        rows = [r for r in rows if r["test_id"] in a.only]
    else:
        done = {d["test_id"] for d in have}
        rows = [r for r in rows if r["test_id"] not in done]
    people = v3data.personas()
    out = out_path(a.split)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{a.split}] {len(rows)} dialogue(s) to write", flush=True)

    def work(row: dict) -> str:
        attempt = 1 + sum(1 for d in load(a.split) if d["test_id"] == row["test_id"])
        item = generate_one(row, people[row["persona_id"]], ClaudeClient(a.model), attempt)
        with _lock:
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        q = item["qc"]
        return (f"  {row['test_id']} attempt {attempt}: schema {len(q['schema_problems'])}, "
                f"leaks {len(q['leak_hits'])}, ungrounded {len(q['ungrounded_quotes'])}")

    with cf.ThreadPoolExecutor(a.workers) as pool:
        for line in pool.map(work, rows):
            print(line, flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    sys.exit(main())
