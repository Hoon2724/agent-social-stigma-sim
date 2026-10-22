"""Run one belief-update implementation over test-set dialogues and score
every session's belief text with the paper's ISMI-29 judge.

The implementation is called exactly the way the paper pipeline calls it
(RQ3_crowd/run_crowd.py, paper_rq1/run_dialogues.py): initial_selfreport()
at session 0, then belief_update_avg(model, facts, name, dialogue_log,
prev_beliefs=history) once per session, so the scored artifact is carried
forward the same way (previous_belief_scored), not the way the older
sanitychecking2 script did. The impl sees only visible_dialogue.

Judging matches measure_ismi29_run.py --belief-update-only: belief_update
text alone through build_pre_innerstate_block, qwen3:32b, temperature 0.3.
One difference, on purpose: within one generation run, identical belief
texts are judged once and the result is reused (a closed gate carries the
text byte-identical). That makes a closed-gate session contribute exactly 0
change instead of judge noise, and halves judge time. The judge noise floor
on identical text is a separate, known quantity (SD ~0.08, paper runs).

Session 0 (pre) text is generated once per persona@initial_ismi with the
sanitychecking2 prompt and frozen in testset/pre/ -- every impl starts from
the same text. A missing pre is generated only with --allow-pre-gen.

  python3 harness.py --impl paper --testset testset/dev.jsonl --ids R1 \
      --base-url http://localhost:11434/v1 --gen-runs 2 --judge-reps 3 \
      --out-dir results/baseline_paper
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PAPER = HERE.parent / "agents"
IMPLS = {
    "final": PAPER / "belief_update.py",   # the final update used in the paper
}
PRE_DIR = HERE / "pre"

# paper_rq1 uses flat imports (llm_client, strategy_selector, measure_ismi29);
# both impls import from it, so it has to be on the path before loading them.
if str(PAPER) not in sys.path:
    sys.path.insert(0, str(PAPER))

# Same prompt as sanitychecking2/sanity_check_belief.py _PRE_PROMPT (the
# paper's Section 4.2 check), copied so the pre step is identical.
PRE_PROMPT = """You are writing a private note to yourself about how you see yourself right now.

[WHO YOU ARE]
{facts}

Nobody has said anything to you about any of this. You have not talked to
anyone about it and you are not in treatment. This is just where you are.

{intensity_hint}

Write 5-7 first-person sentences saying how you see yourself and your
situation, in the present tense. Cover what you think is true about you, what
you do and do not let people know, and what you expect would happen if they
knew. Ground it in the specific beliefs above rather than restating them word
for word.

Write plainly, the way a person writes privately -- unpolished, no metaphors
or literary imagery. Do not compare this to how you used to be: no "better",
"worse", "than before", "lately", "starting to", "no longer". There is nothing
to compare to yet. Do not mention scores, questionnaires, therapy jargon, or
this instruction.

Return exactly one JSON object:
{{"self_view": "<5-7 first-person sentences>"}}"""


def load_impl(name: str):
    path = Path(IMPLS.get(name, name)).resolve()
    spec = importlib.util.spec_from_file_location(f"impl_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in ("initial_selfreport", "belief_update_avg", "narrative_of"):
        if not hasattr(module, attr):
            raise SystemExit(f"{path} lacks {attr}() -- the paper pipeline calls it")
    return module, path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def pre_key(item: dict) -> str:
    c = item["condition"]
    return f"{c['persona_id']}@{c['initial_ismi']}"


def pre_path(item: dict, pre_dir: Path = PRE_DIR) -> Path:
    return pre_dir / f"{pre_key(item)}.txt"


def make_pre(model, item: dict, complete_json) -> str:
    first = item["hidden_ground_truth"]["session_targets"][0]
    prompt = PRE_PROMPT.format(
        facts=item["condition"]["diagnosis_context"],
        intensity_hint=f"How this currently sits with you:\n{first.get('intended_register', '')}")
    for _ in range(3):
        try:
            text = str(complete_json(model, [{"role": "user", "content": prompt}],
                                     temperature=0.7).get("self_view", "")).strip()
            if text:
                return text
        except (ValueError, KeyError, TypeError):
            continue
    raise RuntimeError(f"pre generation failed for {pre_key(item)}")


def session_turns(session: dict) -> list[dict]:
    n = session["session"]
    return [{"speaker": t["speaker"], "text": t["text"], "content": t["text"], "session": n}
            for t in session["turns"]]


def generate_run(impl, gen_model, item: dict, pre_text: str) -> list[dict]:
    """One independent pass over the 8 sessions. Returns belief_history,
    session 0 included, exactly as the paper pipeline stores it."""
    c = item["condition"]
    facts, name = c["diagnosis_context"], c["persona_id"]
    history = [impl.initial_selfreport(gen_model, facts, name, pre_text)]
    for session in item["visible_dialogue"]:
        t0 = time.time()
        belief = impl.belief_update_avg(gen_model, facts, name, session_turns(session),
                                        prev_beliefs=history)
        belief["session"] = session["session"]
        belief["_seconds"] = round(time.time() - t0, 1)
        history.append(belief)
        print(f"    S{session['session']} changed={belief.get('changed')} "
              f"({belief['_seconds']}s)", flush=True)
    return history


def judge_text(judge_model, text: str, reps: int, temperature: float, measure) -> dict:
    measure_once, block = measure
    per_rep, failures = [], 0
    for _ in range(reps):
        try:
            per_rep.append(round(measure_once(judge_model, "", temperature, block(text))["total_mean"], 3))
        except (ValueError, KeyError, TypeError):
            failures += 1
    return {
        "ismi": round(statistics.fmean(per_rep), 3) if per_rep else None,
        "sd": round(statistics.stdev(per_rep), 3) if len(per_rep) > 1 else 0.0,
        "per_rep": per_rep, "failures": failures,
    }


KEEP = ("session", "changed", "belief_update", "belief_narrative", "reasoning", "grounded_in",
        "engaged", "engaged_stages", "disclosure", "_is_fallback", "_seconds")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="final", help="final | path/to/module.py")
    ap.add_argument("--testset", type=Path, default=HERE / "dev.jsonl")
    ap.add_argument("--ids", nargs="*", help="test_ids to run (default: all)")
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--judge-model", default="qwen3:32b")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--gen-base-url", help="server for the belief-update (generation) model when it must differ from the "
                    "judge server; default = --base-url")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--judge-num-ctx", type=int, default=8192)
    ap.add_argument("--num-gpu", type=int, default=99)
    ap.add_argument("--gen-runs", type=int, default=2)
    ap.add_argument("--judge-reps", type=int, default=3)
    ap.add_argument("--judge-temperature", type=float, default=0.3)
    ap.add_argument("--pre-dir", type=Path, default=PRE_DIR,
                    help="where session-0 texts live (frozen copy: testset/pre)")
    ap.add_argument("--allow-pre-gen", action="store_true",
                    help="generate a missing session-0 text (baseline only)")
    ap.add_argument("--out-dir", type=Path, required=True)
    # 10-20: split the two phases across processes so each GPU holds one model (14b servers for
    # generation, then 32b servers for judging). Judging and the output format are unchanged.
    ap.add_argument("--gen-only", type=Path, help="write phase-1 notes to this JSON and stop")
    ap.add_argument("--judge-from", type=Path, help="skip phase 1, judge notes saved by --gen-only")
    a = ap.parse_args()

    impl, impl_path = load_impl(a.impl)
    from llm_client import OpenAICompatibleModel
    from measure_ismi29 import build_pre_innerstate_block, measure_once
    from strategy_selector import complete_json

    items = [json.loads(l) for l in a.testset.read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.ids:
        items = [i for i in items if i["test_id"] in set(a.ids)]
        missing = set(a.ids) - {i["test_id"] for i in items}
        if missing:
            raise SystemExit(f"unknown test_ids: {sorted(missing)}")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    gen_model = OpenAICompatibleModel(a.model, a.gen_base_url or a.base_url, num_ctx=a.num_ctx, num_gpu=a.num_gpu)
    judge_model = OpenAICompatibleModel(a.judge_model, a.base_url,
                                        num_ctx=a.judge_num_ctx, num_gpu=a.num_gpu)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    # Phase 1: all generation with the 14b, so the GPU holds one model at a time.
    generated = {}
    if a.judge_from:
        generated = json.loads(a.judge_from.read_text(encoding="utf-8"))["generated"]
    for item in ([] if a.judge_from else items):
        tid = item["test_id"]
        print(f"[gen] {tid} {item['condition']['trajectory_type']}", flush=True)
        pp = pre_path(item, a.pre_dir)
        if not pp.exists():
            if not a.allow_pre_gen:
                raise SystemExit(f"{pp} missing -- pre texts are frozen; use --allow-pre-gen only for the first baseline")
            a.pre_dir.mkdir(parents=True, exist_ok=True)
            pp.write_text(make_pre(gen_model, item, complete_json) + "\n", encoding="utf-8")
            print(f"  pre generated -> {pp.name}", flush=True)
        pre_text = pp.read_text(encoding="utf-8").strip()
        generated[tid] = []
        for r in range(a.gen_runs):
            print(f"  run {r + 1}/{a.gen_runs}", flush=True)
            generated[tid].append(generate_run(impl, gen_model, item, pre_text))
    if a.gen_only:
        a.gen_only.parent.mkdir(parents=True, exist_ok=True)
        # keep only what phase 2 reads (KEEP); an impl may carry live objects in its belief dict
        slim = {tid: [[{k: b.get(k) for k in KEEP if k in b} for b in hist] for hist in runs_]
                for tid, runs_ in generated.items()}
        a.gen_only.write_text(json.dumps({"impl": a.impl, "gen_runs": a.gen_runs, "generated": slim},
                                         ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        print(f"[gen-only] -> {a.gen_only}", flush=True)
        return

    # Phase 2: judge with the 32b. Identical texts within a run share one judgment.
    for item in items:
        tid = item["test_id"]
        targets = {t["session"]: t["target_ismi"] for t in item["hidden_ground_truth"]["session_targets"]}
        targets[0] = item["condition"]["initial_ismi"]
        runs = []
        for history in generated[tid]:
            cache, sessions = {}, []
            for belief in history:
                text = str(belief.get("belief_update", "")).strip()
                h = text_hash(text)
                if h not in cache:
                    cache[h] = judge_text(judge_model, text, a.judge_reps, a.judge_temperature,
                                          (measure_once, build_pre_innerstate_block))
                row = {k: belief.get(k) for k in KEEP}
                row.update({"text_sha": h, "target_ismi": targets.get(belief["session"]),
                            "words": len(text.split()), **cache[h]})
                sessions.append(row)
            runs.append({"sessions": sessions})
            print(f"[judge] {tid} " + " ".join(f"{s['ismi']}" for s in sessions), flush=True)
        out = {
            "test_id": tid, "trajectory_type": item["condition"]["trajectory_type"],
            "persona_id": item["condition"]["persona_id"],
            "initial_ismi": item["condition"]["initial_ismi"],
            "target_final_ismi": item["condition"]["target_final_ismi"],
            "impl": a.impl, "impl_path": (str(impl_path.relative_to(REPO)) if impl_path.is_relative_to(REPO) else impl_path.name),
            "impl_sha256": sha256_file(impl_path), "git_commit": commit,
            "gen_model": a.model, "judge_model": a.judge_model,
            "gen_runs": a.gen_runs, "judge_reps": a.judge_reps,
            "judge_temperature": a.judge_temperature, "num_ctx": a.num_ctx,
            "judge_num_ctx": a.judge_num_ctx, "pre_key": pre_key(item),
            "runs": runs,
        }
        (a.out_dir / f"{tid}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
        print(f"[done] {tid} -> {a.out_dir / (tid + '.json')}", flush=True)


if __name__ == "__main__":
    main()
