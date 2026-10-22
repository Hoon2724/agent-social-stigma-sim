"""measure_ismi29_run.py — administer ISMI-29 to a COMPLETED run, per session.

The baseline script (measure_ismi29.py) measures a persona's stigma from the
backstory alone, before any intervention. This one measures the same 29 items
AFTER the persona has lived through the dialogue, so a pre/post delta shows what
the conversations moved.

Materials mirror the belief judge (score_session): backstory + the dialogue the
persona went through + the belief_update they wrote. The instrument itself is
untouched — same items, same anchors, same "answer from the persona's subjective
beliefs" instruction — so the only thing that differs between pre and post is how
much of the persona's life the prompt describes. Tuning the questionnaire to move
a number would make the measurement meaningless (see measure_ismi29.py header).

  pre           = backstory only (identical to the baseline measurement)
  post,session s = backstory + session s's dialogue + session s's belief_update

The dialogue is THIS SESSION ONLY, not sessions 1..s. An earlier version
accumulated it and the measurement broke: ISMI climbed across sessions even when
the dialogue clearly improved, because the judge kept quoting the distressed
opening sessions. ismi_eval/judge_prompts.py records the same finding from the
other direction — a recency-priority instruction did not fix cumulative
transcripts either. Past context is not lost by dropping the accumulation:
belief_update is itself cumulative (belief.py carries it forward session to
session), so it is the channel that carries history, while the transcript
carries what just happened. A side effect is that prompt size no longer grows
with session number, so nothing silently falls out of the context window.

Usage:
  OLLAMA_BASE_URL=http://localhost:11440/v1 python3 measure_ismi29_run.py \
    --run outputs/demo_riley_mid/c2_run_01.json \
    --model qwen3:32b --reps 3 \
    --out outputs/demo_riley_mid/c2_ismi29.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from ismi29_items import ISMI_SUBSCALES
from llm_client import OpenAICompatibleModel
from measure_ismi29 import (
    build_experience_block, build_pre_innerstate_block, interpret, measure_once,
)
from reddit_persona_profiles import build_reddit_backstory_for_measurement

HERE = Path(__file__).resolve().parent


def _aggregate(results: list[dict]) -> dict:
    """Mean/SD over reps, matching measure_ismi29.measure_persona's output."""
    totals = [r["total_mean"] for r in results]
    return {
        "n_ok": len(results),
        "ismi29_total_mean": round(statistics.fmean(totals), 3),
        "ismi29_total_sd": round(statistics.stdev(totals), 3) if len(totals) > 1 else 0.0,
        "tier": interpret(statistics.fmean(totals)),
        "subscale_means": {
            s: round(statistics.fmean(r["subscale_means"][s] for r in results), 3)
            for s in ISMI_SUBSCALES
        },
        "per_rep_totals": [round(t, 3) for t in totals],
        "per_rep_subscales": [r["subscale_means"] for r in results],
    }


def _persona_key_for_level(personas_path: Path, level: int) -> str | None:
    personas = json.loads(personas_path.read_text(encoding="utf-8"))["personas"]
    return next((p["persona_key"] for p in personas if p["level"] == level), None)


def _pre_from_baseline(baseline_path: Path, personas_path: Path,
                       level: int) -> dict | None:
    """The persona's pre-measured baseline row, reshaped like a _measure result.

    Re-measuring pre for every run wastes calls and, worse, lets the two
    conditions land on different baselines: in a two-session pilot the same
    backstory scored 2.517 under one run and 2.713 under another, purely from
    judge noise (baseline rep SD averages 0.152). A delta against a per-run pre
    would carry that difference into the condition comparison. Reusing one
    baseline measured at n=5 gives every run the same origin.
    """
    key = _persona_key_for_level(personas_path, level)
    if key is None:
        return None
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    row = next((r for r in baseline["results"] if r.get("persona_key") == key), None)
    if row is None or row.get("ismi29_total_mean") is None:
        return None
    # measure_ismi29.py's baseline rows use "n_reps"; measure_ismi29_pre_innerstate.py's
    # use "n_ok". Fall back across both so --pre-from works against either baseline
    # file without silently producing a falsy n_ok (which made delta compute as
    # None for every row pulled from the innerstate baseline).
    return {
        "n_ok": row.get("n_ok") or row.get("n_reps"),
        "n_failed": 0,
        "ismi29_total_mean": row["ismi29_total_mean"],
        "ismi29_total_sd": row.get("ismi29_total_sd", 0.0),
        "tier": interpret(row["ismi29_total_mean"]),
        "subscale_means": row.get("subscale_means", {}),
        "_source": f"{baseline_path.name}:{key}",
    }


def _measure(model, backstory, experience, reps, temperature) -> dict:
    results, failures = [], 0
    for rep in range(reps):
        try:
            results.append(measure_once(model, backstory, temperature, experience))
        except (ValueError, KeyError, TypeError) as error:
            failures += 1
            print(f"    [rep {rep + 1} failed] {error}")
    if not results:
        return {"n_ok": 0, "n_failed": failures, "error": "no usable measurement"}
    out = _aggregate(results)
    out["n_failed"] = failures
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model", default="qwen3:32b")
    ap.add_argument("--base-url")
    ap.add_argument("--num-ctx", type=int, default=4096,
                    help="forced context size, so the judge model stays fully GPU-resident "
                         "instead of ollama's flaky automatic split (see llm_client.py).")
    ap.add_argument("--num-gpu", type=int, default=99,
                    help="force every layer onto GPU (see llm_client.OpenAICompatibleModel).")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--pre-from", type=Path,
                    help="reuse a persona's baseline as pre instead of re-measuring "
                         "(e.g. outputs/ismi29_baseline/summary.json)")
    ap.add_argument("--pre-from-session0", action="store_true",
                    help="measure this run's own session-0 belief_update as pre. "
                         "Useful with --belief-update-only, because post is also "
                         "scored from belief_update text only; avoids mixing a "
                         "shared baseline narrative with this run's sampled "
                         "trajectory narrative.")
    ap.add_argument("--personas", type=Path,
                    default=HERE / "outputs_v2" / "personas.json",
                    help="level -> persona_key lookup, for --pre-from")
    ap.add_argument("--no-backstory", action="store_true",
                    help="score each session from the dialogue + belief_update only, "
                         "without the fixed backstory paragraph (the backstory is "
                         "identical every session and anchors the judge, blunting "
                         "session-to-session change)")
    ap.add_argument("--dialogue-only", action="store_true",
                    help="score each session from that session's DIALOGUE alone, "
                         "with no belief_update and no backstory. The mirror of "
                         "--belief-update-only: together the two say which "
                         "channel an effect travelled down. A change that shows "
                         "up in the self-report but not in the dialogue means "
                         "the persona's private account moved while what they "
                         "actually said to people did not, which is a different "
                         "claim from the one a combined score would support.")
    ap.add_argument("--belief-update-only", action="store_true",
                    help="score each session from that session's belief_update ALONE, "
                         "using the same block builder as the pre baseline "
                         "(build_pre_innerstate_block). Implies --no-backstory. "
                         "Exists because the default post block also hands the judge "
                         "the whole dialogue thread while pre sees only an "
                         "belief_update: with different material on each side, a "
                         "pre->s1 delta partly measures the change in what the judge "
                         "was shown. Observed with the default: pre spread SD=0.533 "
                         "collapses to SD=0.213 at post, with 78%% of all sessions "
                         "landing in one 0.25-wide bin, and low-tier personas moving "
                         "UP. Use this flag to put both sides on one ruler.")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    run = json.loads(a.run.read_text(encoding="utf-8"))
    # With --no-backstory the post prompt carries only dialogue + belief_update.
    # pre still needs the backstory (it is the only description of the person
    # before any dialogue exists), so keep it for the pre measurement.
    if a.belief_update_only and a.dialogue_only:
        raise SystemExit("--belief-update-only and --dialogue-only are exclusive; "
                         "run the script twice to get both channels")
    if a.belief_update_only or a.dialogue_only:
        a.no_backstory = True
    # Measurement-only backstory (2026-08-02), not run["backstory"]. The saved
    # run backstory is build_reddit_backstory_longitudinal's text -- correct
    # for prompting the DIALOGUE GENERATOR, but its closing paragraph
    # ("Write each week from where they actually stand...") is an instruction
    # to a generator, not persona content, and was leaking into this
    # self-report judge prompt at both pre and post. Regenerated from
    # persona_key rather than read from the run so it's guaranteed to match
    # what measure_ismi29.py's baseline uses -- the whole point of
    # build_reddit_backstory_for_measurement existing is that baseline/pre/post
    # are the same measurement material (see module docstring above).
    persona_key = _persona_key_for_level(a.personas, run.get("level"))
    backstory = (build_reddit_backstory_for_measurement(persona_key)
                if persona_key else run["backstory"])
    post_backstory = "" if a.no_backstory else backstory
    turns = run["turns"]
    beliefs = {b["session"]: b for b in run.get("belief_history", [])}
    sessions = sorted({t.get("session", 1) for t in turns})

    model = OpenAICompatibleModel(a.model, a.base_url, num_ctx=a.num_ctx, num_gpu=a.num_gpu)

    # pre: either this run's own session-0 belief_update (same material shape as
    # --belief-update-only post), a shared baseline, or a fresh backstory
    # measurement. Session-0 pre is preferred for within-run trajectories; the
    # shared baseline is useful for tier context and for avoiding judge noise in
    # between-condition contrasts.
    pre = None
    if a.pre_from_session0:
        seed_belief = next((b for b in run.get("belief_history", [])
                            if int(b.get("session", -1)) == 0), None)
        seed_text = str((seed_belief or {}).get("belief_update", "")).strip()
        if seed_text:
            print(f"[ismi29-run] pre from this run's session-0 belief_update "
                  f"({a.reps} reps)...", flush=True)
            pre = _measure(model, "", build_pre_innerstate_block(seed_text),
                           a.reps, a.temperature)
            pre["_source"] = f"{a.run.name}:belief_history.session0"
            if pre.get("n_ok"):
                print(f"    ISMI-29 = {pre['ismi29_total_mean']} "
                      f"(SD {pre['ismi29_total_sd']}) -> {pre['tier']}")
        else:
            print(f"[ismi29-run] no session-0 belief_update in {a.run}; "
                  f"falling back", flush=True)
    if pre is None and a.pre_from:
        pre = _pre_from_baseline(a.pre_from, a.personas, run.get("level"))
        if pre:
            print(f"[ismi29-run] pre from baseline ({pre['_source']}): "
                  f"{pre['ismi29_total_mean']} -> {pre['tier']}", flush=True)
        else:
            print(f"[ismi29-run] no baseline row for level {run.get('level')}; "
                  f"measuring pre instead", flush=True)
    if pre is None:
        print(f"[ismi29-run] pre (backstory only, {a.reps} reps)...", flush=True)
        pre = _measure(model, backstory, "", a.reps, a.temperature)
        if pre.get("n_ok"):
            print(f"    ISMI-29 = {pre['ismi29_total_mean']} "
                  f"(SD {pre['ismi29_total_sd']}) -> {pre['tier']}")

    # post per session: THIS session's dialogue only, paired with that session's
    # belief_update (which is itself cumulative). A trajectory, not just a single
    # endpoint. See the module header for why the dialogue is not accumulated.
    post_rows = []
    for s in sessions:
        dialogue = [t for t in turns if t.get("session", 1) == s]
        inner = str(beliefs.get(s, {}).get("belief_update", ""))
        if a.belief_update_only:
            experience = build_pre_innerstate_block(inner)
        elif a.dialogue_only:
            experience = build_experience_block(dialogue, "",
                                                with_backstory=False)
        else:
            experience = build_experience_block(dialogue, inner,
                                                with_backstory=not a.no_backstory)
        shown = ("belief_update only" if a.belief_update_only
                 else f"{len(dialogue)} turns, no belief_update"
                 if a.dialogue_only else f"{len(dialogue)} turns")
        print(f"[ismi29-run] post session {s} "
              f"({shown}, {a.reps} reps)...", flush=True)
        row = _measure(model, post_backstory, experience, a.reps, a.temperature)
        row["session"] = s
        post_rows.append(row)
        if row.get("n_ok"):
            print(f"    ISMI-29 = {row['ismi29_total_mean']} "
                  f"(SD {row['ismi29_total_sd']}) -> {row['tier']}")

    last = post_rows[-1] if post_rows else {}
    delta = (round(last["ismi29_total_mean"] - pre["ismi29_total_mean"], 3)
             if pre.get("n_ok") and last.get("n_ok") else None)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "run": str(a.run),
        "level": run.get("level"),
        "agent_type": run.get("agent_type"),
        "model": a.model,
        "reps": a.reps,
        "temperature": a.temperature,
        "scale": "ISMI-29, 1-4, stigma_resistance reverse-scored; "
                 "higher = more internalized stigma. delta = post(last) - pre.",
        "post_materials": (
            "this session's belief_update ONLY (same block as pre; no dialogue, "
            "no backstory)" if a.belief_update_only
            else ("this session's dialogue (not cumulative) + this "
                  "session's belief_update, no backstory") if a.no_backstory
            else ("backstory + this session's dialogue (not "
                  "cumulative) + this session's belief_update")),
        "pre_source": pre.get("_source", "measured in this run"),
        "ismi29_pre": pre,
        "ismi29_post_by_session": post_rows,
        "ismi29_delta_total": delta,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ismi29-run] delta (post-pre) = {delta}")
    print(f"[ismi29-run] saved -> {a.out}")


if __name__ == "__main__":
    main()
