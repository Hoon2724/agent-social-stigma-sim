"""measure_ismi29.py — administer the official ISMI-29 to a persona backstory.

The persona answers all 29 items in the first person, from the backstory alone;
reverse scoring and subscale means are computed in code (ismi29_items). This is
the only module allowed to import the item text — see ismi29_items' header for
why. Nothing here is used during dialogue generation; it runs offline, before
the experiment, to establish each persona's baseline stigma tier.

The prompt is adapted from simulation/ismi_sanity_check/run_measure_ismi.py,
which recovered a 10-level stigma ladder at Spearman rho = -0.976, extended
from ISMI-10 to the full 29 items. Its key instruction is preserved verbatim in
spirit: answer from the persona's subjective beliefs, and do NOT correct their
stigma into a healthier or more socially desirable view. Without that, models
answer as a well-adjusted person regardless of the backstory.

The prompt and item set are FIXED. Tiers are hit by rewriting persona prose and
re-measuring, never by tuning this file -- adjusting the instrument to produce a
wanted number would make the measurement meaningless.

Usage:
  OLLAMA_BASE_URL=http://localhost:11440/v1 python3 measure_ismi29.py \
    --model qwen3:32b --reps 5 --out outputs/ismi29_baseline/summary.json

  # single tier / subset
  ... --personas depression_severe,depression_moderate,depression_mild
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from ismi29_items import ISMI_29_ITEMS, ISMI_SUBSCALES, score_ismi29
from llm_client import ChatModel, OpenAICompatibleModel
from reddit_persona_profiles import REDDIT_PERSONA_PROFILES, build_reddit_backstory_for_measurement
from strategy_selector import complete_json

HERE = Path(__file__).resolve().parent

# RQ1's own 3-band cutoffs (2026-08-08), NOT the standard Lysaker et al.
# 2007 4-category method (which splits 1.00-2.00 minimal | 2.01-2.50 mild |
# 2.51-3.00 moderate | 3.01-4.00 severe) -- collapsed to 3 bands to match the
# 3 persona tiers this study actually runs (severe/moderate/mild, no
# "minimal" persona), with round-number cutoffs at 2.5 and 3.0 rather than
# the standard's .01-offset boundaries. Ordered high to low so the first
# match wins; lower bound inclusive, upper bound exclusive, so a value
# sitting exactly on a cutoff (3.0, 2.5) falls into the higher band. A
# 2-category method also exists (Ritsher [Boyd] & Phelan 2004: <=2.50 "does
# not report high internalized stigma" / >2.50 "reports high"), deliberately
# NOT used for the primary analysis here: measured session values cluster
# tightly around that single cutoff (median 2.598 over 240 sessions, 47%
# within +/-0.11 of it), so dichotomizing there amplifies judge noise instead
# of absorbing it. Reported outcomes stay on the continuous mean; these bands
# are labels, not the metric.
_TIERS = (
    ("severe", 3.00, 4.01),
    ("moderate", 2.50, 3.00),
    ("mild", 0.00, 2.50),
)


def interpret(total_mean: float) -> str:
    for label, lo, hi in _TIERS:
        if lo <= total_mean < hi:
            return label
    return "mild"


def build_prompt(backstory: str, experience: str = "") -> str:
    """ISMI-29 administration prompt.

    `backstory` is the fixed persona paragraph; `experience` is the optional
    dialogue + belief-update block (build_experience_block). When `backstory` is
    empty the persona is described entirely by `experience`, with no fixed
    backstory paragraph. That paragraph is the same across every session, so
    including it anchors the judge on how the person started and blunts
    session-to-session change; dropping it lets each session be scored from what
    actually happened in it, the way the belief judge and a co-author's earlier
    window/reflection modes do.
    """
    items = "\n".join(f'{it["id"]}. {it["text"]}' for it in ISMI_29_ITEMS)
    fields = ",\n".join(f'  "{it["id"]}": <1|2|3|4>' for it in ISMI_29_ITEMS)
    if backstory.strip():
        source_line = "You are the person described in the backstory below."
        backstory_block = f"\n[Backstory]\n{backstory}\n"
        term_ref = "your mental health difficulty in the backstory"
    else:
        source_line = ("You are the person whose recent life is described below "
                       "(their conversations and their own private notes).")
        backstory_block = ""
        term_ref = "your mental health difficulty as it shows in the material below"
    return f"""{source_line}
Internalize it as your own lived experience. Do not answer as an
outside observer, clinician, or survey designer.
Answer from the persona's subjective beliefs and fears, even when those beliefs
are unfair, self-stigmatizing, or not clinically accurate. Do not correct the
persona's stigma into a healthier or more socially desirable view.

If the material below includes a conversation transcript, answer from what
the person actually concludes about themselves -- the substance of the
belief -- not from the casual speech habits they happen to say it in. People
have different natural ways of talking ("I guess", "idk", "honestly"), and
the same underlying belief can be said plainly, warmly, or with that kind of
filler without meaning something different about how much they believe it --
read past those verbal tics to what is actually being claimed.

This is different from actual expressed uncertainty about the belief itself
-- "some days I think X, other days I don't", "I don't know if that's true
or not", "a real fifty-fifty" -- which is not a speech habit, it IS the
claim: a person who genuinely wavers on whether something is true of them
holds it less strongly than a person who states it flatly, and that
difference is exactly what should move your answer toward the middle of the
scale rather than the strong-agree end. Do not flatten "I'm not sure if I'm
broken" into the same answer as "I am broken" just because both mention
being broken -- the uncertainty is content, not decoration.

When the material includes both a conversation transcript and a section
describing how the person has come to see themselves, that second section is
their real, settled inner belief -- reconstructed from everything that
happened, not just a summary of the conversation. Weigh it as the primary
evidence for how they see themselves; the transcript is context for how that
belief came to be, not a separate or competing account of it.
{backstory_block}{experience}
The questionnaire uses the term "mental illness"; think of it as whatever term
best fits {term_ref}, even if you have never
been given a diagnosis and would not use that word about yourself.

For each item, choose:
1 = strongly disagree
2 = disagree
3 = agree
4 = strongly agree

Items:
{items}

Output exactly one JSON object with integer scores only, no other text:
{{
{fields}
}}"""


def build_experience_block(dialogue_turns: list[dict], belief_update: str,
                           with_backstory: bool = True) -> str:
    """Render the intervening experience — the dialogue thread this persona
    lived through, then the belief update they carry out of it — as extra context
    to append to the backstory for a POST measurement.

    The instrument (item text, response anchors, the "answer from the persona's
    subjective beliefs" instruction) is untouched: pre and post are the same
    ISMI-29 administration, differing only in how much of the persona's life the
    prompt describes. Pre sees the backstory alone; post sees the backstory plus
    what has happened since. Anything else would make the delta an artifact of a
    changed instrument rather than a changed persona.
    """
    lines = []
    for t in dialogue_turns:
        speaker = t.get("speaker", "someone")
        # Condition-neutral by design (2026-08-02): this used to render
        # {"peer": "A peer replied", "general": "A stranger replied"}, which
        # printed the experimental condition into the outcome instrument
        # itself — the persona answering ISMI-29/ATSPPH-29 could see whether
        # it had been talking to "a peer" or "a stranger". That is a demand
        # characteristic on the dependent variable, and it is not defensible
        # the way an in-dialogue cue is (a real feed shows profiles; a
        # questionnaire does not). Who said what is still recoverable from the
        # run JSON for any analysis that needs it.
        who = {"patient": "You wrote"}.get(speaker, "Someone replied")
        lines.append(f"  {who}: {t.get('content', '').strip()}")
    thread = "\n".join(lines) if lines else "(no dialogue)"
    since = ("conversations you have had online in the weeks after the backstory "
             "above" if with_backstory else
             "conversations you have had online with someone over the past weeks")
    block = f"\n[What has happened — {since}]\n{thread}\n"
    # An empty belief_update means the caller wants the dialogue channel scored
    # on its own (--dialogue-only), so the section is omitted entirely rather
    # than filled with a placeholder. Printing "(no belief update recorded)"
    # would tell the judge that something it was meant to see is missing, which
    # is itself information about the persona and not the same prompt the other
    # channel is scored under.
    if belief_update.strip():
        block += (
            f"\n[How you have come to see yourself and your situation now, after "
            f"those conversations]\n{belief_update.strip()}\n"
        )
    return block


def build_pre_innerstate_block(inner_state: str) -> str:
    """PRE-measurement block: no dialogue has happened yet, so this omits the
    "[What has happened]" / "after those conversations" framing that
    build_experience_block uses for POST — that framing would be false before
    any conversation exists. Pairs with a session-0 inner_state generated by
    measure_pre_belief.pre_belief_narrative(), so pre and every post session
    are scored from the same material shape (inner_state only, no backstory
    paragraph) and a pre->s1 delta reflects the conversation, not a change in
    what the judge was shown.
    """
    inner = inner_state.strip() or "(no inner state recorded)"
    return f"\n[How you see yourself and your situation right now]\n{inner}\n"


def build_cumulative_dialogue_block(dialogue_turns: list[dict],
                                    with_backstory: bool = True) -> str:
    """Render sessions 1..s of dialogue, labeled per session, with NO
    inner_state section — an earlier judge_prompts.py "cumulative" transcript
    mode (--window=None), adapted to this repo's speaker labels.

    This repo's own per-session measurement (measure_ismi29_run.py) does NOT
    use this: an earlier version accumulated dialogue and the measurement
    broke (ISMI kept climbing across sessions because the judge kept quoting
    the distressed opening sessions, even as the dialogue clearly improved;
    That earlier judge_prompts.py records the same failure from the other
    direction — a recency-priority instruction did not fix it either). This
    function exists to reproduce that comparison point on demand, not because
    it is the recommended measurement.
    """
    lines = []
    for t in dialogue_turns:
        speaker = t.get("speaker", "someone")
        # Condition-neutral by design (2026-08-02): this used to render
        # {"peer": "A peer replied", "general": "A stranger replied"}, which
        # printed the experimental condition into the outcome instrument
        # itself — the persona answering ISMI-29/ATSPPH-29 could see whether
        # it had been talking to "a peer" or "a stranger". That is a demand
        # characteristic on the dependent variable, and it is not defensible
        # the way an in-dialogue cue is (a real feed shows profiles; a
        # questionnaire does not). Who said what is still recoverable from the
        # run JSON for any analysis that needs it.
        who = {"patient": "You wrote"}.get(speaker, "Someone replied")
        lines.append(f"  [Session {t.get('session', '?')}] {who}: {t.get('content', '').strip()}")
    thread = "\n".join(lines) if lines else "(no dialogue)"
    since = ("conversations you have had online in the weeks after the backstory "
             "above" if with_backstory else
             "conversations you have had online with someone over the past weeks")
    return f"\n[What has happened — {since}]\n{thread}\n"


def measure_once(model: ChatModel, backstory: str, temperature: float,
                 experience: str = "") -> dict:
    """One full ISMI-29 administration. Returns scored result, or raises.

    `experience` is appended to the backstory for a POST measurement; empty for
    a baseline (pre). Both go through build_prompt unchanged."""
    raw = complete_json(
        model, [{"role": "user", "content": build_prompt(backstory, experience)}],
        temperature=temperature,
    )
    # complete_json only guarantees valid JSON, not that every item came back.
    responses = {}
    for item in ISMI_29_ITEMS:
        key = str(item["id"])
        value = raw.get(key, raw.get(item["id"]))
        if value is None:
            raise ValueError(f"missing item {item['id']}")
        responses[item["id"]] = int(value)
    return score_ismi29(responses)


def _aggregate(key: str, results: list[dict], failures: int) -> dict:
    if not results:
        # No imputed scores: a fabricated baseline would propagate into every
        # delta computed against it.
        return {"persona_key": key, "n_ok": 0, "n_failed": failures,
                "error": "no usable measurement"}

    totals = [r["total_mean"] for r in results]
    mean_total = round(statistics.fmean(totals), 3)
    subscales = {
        s: round(statistics.fmean(r["subscale_means"][s] for r in results), 3)
        for s in ISMI_SUBSCALES
    }
    return {
        "persona_key": key,
        "n_ok": len(results),
        "n_failed": failures,
        "ismi29_total_mean": mean_total,
        "ismi29_total_sd": round(statistics.stdev(totals), 3) if len(totals) > 1 else 0.0,
        "tier": interpret(mean_total),
        "subscale_means": subscales,
        "per_rep_totals": [round(t, 3) for t in totals],
        # Per-rep item responses, so a later run can add repetitions to this
        # one instead of re-measuring from scratch, and so a subscale can be
        # recomputed after the fact. Storing only the aggregate meant an
        # earlier n=3 pass could not be topped up to n=5.
        "per_rep_items": [r["items_scored"] for r in results],
        "per_rep_subscales": [r["subscale_means"] for r in results],
    }


def measure_persona(model: ChatModel, key: str, reps: int,
                    temperature: float) -> dict:
    """Sequential, single-endpoint measurement. Kept for scripts/notebooks
    that want one persona measured on one model without pool machinery."""
    backstory = build_reddit_backstory_for_measurement(key)
    results, failures = [], 0
    for rep in range(reps):
        try:
            results.append(measure_once(model, backstory, temperature))
        except (ValueError, KeyError, TypeError) as error:
            failures += 1
            print(f"    [rep {rep + 1} failed] {error}")
    return _aggregate(key, results, failures)


def _measure_all_parallel(keys: list[str], reps: int, model_name: str,
                          temperature: float, endpoints: list[str],
                          workers_per_endpoint: int, num_ctx: int) -> list[dict]:
    """All (persona, rep) work items flattened into one pool, round-robined
    across `endpoints`. 9 personas x 5 reps = 45 independent ISMI-29
    administrations with no shared state between them, so this parallelizes
    cleanly -- unlike the dialogue pipeline (patient/peer/general), there is
    no session-to-session or turn-to-turn dependency here to preserve order
    for.

    num_ctx is passed explicitly (not left as None/server-default) because the
    4 endpoints were found already loaded with DIFFERENT context sizes (8192
    vs 32768) -- same model digest, different footprint, so half the pool sat
    at 0% GPU utilization while the other half spilled onto CPU (2700%+ CPU on
    those runners). qwen3:32b's KV cache at 32768 doesn't fit this card's 24GB
    alongside the weights; at num_ctx=4096 (comfortably above this prompt's
    length -- 29 ISMI items + backstory) all 4 stay fully GPU-resident.
    """
    import itertools
    from concurrent.futures import ThreadPoolExecutor, as_completed

    models = [OpenAICompatibleModel(model_name, ep, num_ctx=num_ctx) for ep in endpoints]
    backstories = {key: build_reddit_backstory_for_measurement(key) for key in keys}

    work = [(key, rep) for key in keys for rep in range(reps)]
    cycle = itertools.cycle(range(len(models)))

    def run_one(item):
        key, rep = item
        model = models[next(cycle)]
        return key, rep, measure_once(model, backstories[key], temperature)

    by_key: dict[str, list] = {key: [] for key in keys}
    failures: dict[str, int] = {key: 0 for key in keys}
    done = 0
    with ThreadPoolExecutor(max_workers=len(models) * workers_per_endpoint) as pool:
        futures = {pool.submit(run_one, item): item for item in work}
        for future in as_completed(futures):
            key, rep = futures[future]
            try:
                _, _, result = future.result()
                by_key[key].append(result)
            except (ValueError, KeyError, TypeError) as error:
                failures[key] += 1
                print(f"    [{key} rep {rep + 1} failed] {error}")
            done += 1
            if done % 5 == 0 or done == len(work):
                print(f"[ismi29] {done}/{len(work)} administrations done", flush=True)

    return [_aggregate(key, by_key[key], failures[key]) for key in keys]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", help="comma-separated keys (default: all)")
    ap.add_argument("--model", default="qwen3:32b")
    ap.add_argument("--base-url")
    ap.add_argument("--endpoints",
                    help="comma-separated OpenAI-compatible base URLs to round-robin "
                         "across (e.g. one per GPU). Overrides --base-url; when neither "
                         "is given, falls back to the single-endpoint OLLAMA_BASE_URL "
                         "default in llm_client.py.")
    ap.add_argument("--workers-per-endpoint", type=int, default=4)
    ap.add_argument("--num-ctx", type=int, default=4096,
                    help="forced context size for every endpoint in --endpoints, so "
                         "they all load identically and stay fully GPU-resident "
                         "(qwen3:32b's weights+KV cache exceed 24GB at 32768; this "
                         "prompt -- 29 ISMI items + backstory -- fits well under 4096)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--out", type=Path,
                    default=HERE / "outputs" / "ismi29_baseline" / "summary.json")
    a = ap.parse_args()

    if a.personas:
        keys = [k.strip() for k in a.personas.split(",") if k.strip()]
        unknown = [k for k in keys if k not in REDDIT_PERSONA_PROFILES]
        if unknown:
            raise SystemExit(f"unknown persona key(s): {unknown}")
    else:
        # Skip aliases (e.g. "depression" -> "depression_moderate") so nothing is measured twice.
        seen, keys = set(), []
        for key, profile in REDDIT_PERSONA_PROFILES.items():
            if id(profile) in seen:
                continue
            seen.add(id(profile))
            keys.append(key)

    if a.endpoints:
        endpoints = [e.strip() for e in a.endpoints.split(",") if e.strip()]
        print(f"[ismi29] {len(keys)} personas x {a.reps} reps across "
              f"{len(endpoints)} endpoints ({a.workers_per_endpoint} workers each)...",
              flush=True)
        rows = _measure_all_parallel(keys, a.reps, a.model, a.temperature,
                                     endpoints, a.workers_per_endpoint, a.num_ctx)
        for row in rows:
            if row.get("n_ok"):
                print(f"  {row['persona_key']}: ISMI-29 = {row['ismi29_total_mean']} "
                      f"(SD {row['ismi29_total_sd']}) -> {row['tier']}")
            else:
                print(f"  {row['persona_key']}: NO USABLE MEASUREMENT")
    else:
        model = OpenAICompatibleModel(a.model, a.base_url, num_ctx=a.num_ctx)
        rows = []
        for key in keys:
            print(f"[ismi29] {key} ({a.model}, {a.reps} reps)...", flush=True)
            row = measure_persona(model, key, a.reps, a.temperature)
            rows.append(row)
            if row.get("n_ok"):
                print(f"    ISMI-29 = {row['ismi29_total_mean']} "
                      f"(SD {row['ismi29_total_sd']}) -> {row['tier']}")
                print(f"    subscales: {row['subscale_means']}")
            else:
                print("    NO USABLE MEASUREMENT")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "model": a.model,
        "reps": a.reps,
        "temperature": a.temperature,
        "scale": "ISMI-29, 1-4, stigma_resistance reverse-scored; "
                 "higher = more internalized stigma",
        "results": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ismi29] saved -> {a.out}")


if __name__ == "__main__":
    main()
