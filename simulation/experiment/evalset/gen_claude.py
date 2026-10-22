"""Generate controlled-trajectory test dialogues with Claude (Max login, `claude -p`)
through the SAME pipeline as the paper's Section 4.2 data: sanitychecking2's
prompts.py (v17), trajectories.py, schema.py and generate.py's sequential
one-session-per-call loop. Only the client differs -- GPT-5.5 credits are not
available, so a Claude model writes the sessions.

Claude is called with a clean environment and every tool, setting source,
MCP server and thinking turned off (memory: project_1007_claude_bystander_pilot),
so the desktop app's own context does not leak into the dialogue. The CLI has
no temperature or seed control; generation_params records that.

  python3 gen_claude.py --set dev      # gold: p_patient @3.0 x reducing/neutral/reinforcing
  python3 gen_claude.py --set heldout  # 2 other personas x 3 types
  python3 gen_claude.py --set heldout --only general_neutral --persona p_retail_social   # redo one item
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SANITY2 = HERE.parent.parent / "sanitychecking2"   # simulation/sanitychecking2
sys.path.insert(0, str(SANITY2))
sys.path.append(str(HERE / "_shim"))  # persona_simple (see _shim/persona_simple.py)

import generate  # noqa: E402  (sanitychecking2)
import prompts  # noqa: E402
from schema import find_leak_terms, find_ungrounded_quotes, validate_item  # noqa: E402
from trajectories import build_session_targets  # noqa: E402

PERSONAS = HERE / "testset" / "personas.json"

SETS = {
    # Gold: one per direction, same person, same start -- only the other party differs.
    # Reinforcing starts at 2.0 (user decision 10-12): at 3.0 the judge already read the
    # session-0 note as ~3.5 and first-person notes top out near ~3.65, so a rise could not
    # be measured (loop BLOCKED 10-11 18:37). The 3.0 item is kept in claude_dev_U30_archive.jsonl.
    "dev": [("stigma_reducing", 3.0, "p_patient"),
            ("general_neutral", 3.0, "p_patient"),
            ("stigma_reinforcing", 2.0, "p_patient")],
    # Round-2 dev (10-12): the round-1 candidate passed the gold reinforcing item but left the
    # held-out reinforcing notes almost unchanged; the loop cannot see held-out, so two more
    # reinforcing dialogues with people outside the held-out set give it that signal.
    "dev2": [("stigma_reinforcing", 2.5, "p_office_anxiety"),
             ("stigma_reinforcing", 2.5, "d_line_cook")],
    # Held-out: people the loop never sees, two different starting points.
    "heldout": [(kind, start, pid)
                for pid, start in (("p_retail_social", 3.0), ("h_night_warehouse", 2.5))
                for kind in ("stigma_reducing", "general_neutral", "stigma_reinforcing")],
}
TURNS = 11  # matches the paper's gold items and the production 1:1 thread


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in output")
    return json.loads(text[start:end + 1])


class ClaudeClient:
    def __init__(self, model: str) -> None:
        self.model = model
        self.used_models: set[str] = set()

    def complete_json(self, system: str, user: str, json_schema: dict, temperature: float = 0.7,
                      max_output_tokens: int = 8000, seed: int | None = None) -> dict:
        del temperature, max_output_tokens, seed  # the CLI exposes none of these
        schema = json.dumps(json_schema["schema"], indent=1)
        system_full = (system + "\n\nReturn ONLY one JSON object that validates against this "
                       "JSON Schema -- no prose, no code fence:\n" + schema)
        env = {"HOME": os.environ["HOME"], "PATH": os.environ["PATH"], "LANG": "C.UTF-8"}
        cmd = ["claude", "-p", "--model", self.model, "--system-prompt", system_full,
               "--tools", "", "--setting-sources", "", "--strict-mcp-config",
               "--no-session-persistence", "--exclude-dynamic-system-prompt-sections",
               "--settings", '{"alwaysThinkingEnabled": false}', "--output-format", "json",
               # Structured output: the CLI validates against the schema and returns
               # `structured_output` (10-11: without it one session came back 3x with
               # every turn missing its keys).
               "--json-schema", json.dumps(json_schema["schema"])]
        last = None
        for _ in range(3):
            proc = subprocess.run(cmd, input=user, capture_output=True, text=True, env=env, timeout=900)
            try:
                envelope = json.loads(proc.stdout)
                if envelope.get("is_error"):
                    raise ValueError(f"claude error: {envelope.get('result')}")
                self.used_models.update(envelope.get("modelUsage", {}).keys())
                out = envelope.get("structured_output") or _extract_json(envelope["result"])
                if not isinstance(out.get("turns"), list) or "evidence" not in out:
                    raise ValueError("missing turns/evidence")
                # Kept as a guard: a turn missing a key (seen 10-11 before --json-schema,
                # no "text") crashes the next session's prompt build, so retry instead.
                need = json_schema["schema"]["properties"]["turns"]["items"]["required"]
                bad = [t for t in out["turns"] if not isinstance(t, dict) or any(k not in t for k in need)]
                if bad:
                    raise ValueError(f"{len(bad)} turn(s) missing keys {need}")
                return out
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                last = exc
                print(f"    retry: {exc}", flush=True)
        raise RuntimeError(f"claude -p failed 3 times: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    ap.add_argument("--model", default="opus")
    ap.add_argument("--only", nargs="*", help="trajectory types to (re)generate")
    ap.add_argument("--resume", action="store_true", help="skip items already in the output file")
    ap.add_argument("--persona", help="with --only: regenerate just this persona's item(s)")
    a = ap.parse_args()

    personas = {p["persona_id"]: p for p in json.loads(PERSONAS.read_text())["personas"]}
    client = ClaudeClient(a.model)
    out_path = HERE / "testset" / f"claude_{a.set}.jsonl"
    existing = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                existing[(d["condition"]["trajectory_type"], d["condition"]["persona_id"])] = d

    for n, (kind, start, pid) in enumerate(SETS[a.set]):
        if a.only and (kind not in a.only or (a.persona and pid != a.persona)) and (kind, pid) in existing:
            continue
        if a.resume and not a.only and (kind, pid) in existing:
            print(f"[{a.set}] {kind} {pid}: already generated, skipped", flush=True)
            continue
        spec = {"trajectory_type": kind, "initial_ismi": start,
                "target_final_ismi": generate.target_final_ismi(kind, start),
                "session_count": 8, "turns_per_session": TURNS, "persona_id": pid}
        spec["condition_id"] = generate._condition_id(spec)
        spec["sample_id"] = f"claude_{a.set}__{spec['condition_id']}__{pid}"
        spec["seed"] = None
        persona = personas[pid]
        targets = build_session_targets(kind, start, 8)
        print(f"[{a.set}] {kind} {pid} @{start} -> {spec['target_final_ismi']}", flush=True)
        output = generate.generate_sample_sequentially(
            client, generate._condition_block(spec, persona), persona, targets,
            base_seed=0, temperature=0.7, max_output_tokens=8000)
        item = generate.assemble_item(spec, persona, targets, output,
                                      f"claude:{a.model}", f"belief_v2_claude_{a.set}",
                                      {"temperature": "cli-default", "max_output_tokens": None,
                                       "resolved_models": sorted(client.used_models)})
        item["qc"] = {"schema_problems": validate_item(item), "leak_hits": find_leak_terms(item),
                      "ungrounded_quotes": find_ungrounded_quotes(item)}
        print(f"  qc: {json.dumps(item['qc'])[:300]}", flush=True)
        existing[(kind, pid)] = item
        order = [(k, p) for k, _, p in SETS[a.set]]
        out_path.write_text("".join(json.dumps(existing[key], ensure_ascii=False) + "\n"
                                    for key in order if key in existing), encoding="utf-8")
    print(f"-> {out_path} (prompt {prompts.PROMPT_VERSION}, models {sorted(client.used_models)})")


if __name__ == "__main__":
    main()
