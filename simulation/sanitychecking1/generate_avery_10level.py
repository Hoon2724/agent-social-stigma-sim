"""generate_avery_10level.py — sanity check 1: can an ISMI-29 judge recover a
predefined ordering of self-stigma severity from generated persona text?

Ten variants of one person are generated, differing only in self-stigma level.
Levels are defined by structural coordinates (_LEVEL_COORDINATES) on the same
three axes the main simulation's own persona generation varies -- what the struggle means about
them, how much each of the three relationships knows, and what they can say out
loud -- stated as situation rather than severity, so the model never has to
invent a tenth adjective for "worse".

Two rejected alternatives, for the record:
  - Level number alone ("write ten evenly spaced levels"): Spearman rho 0.21.
    With nothing but an ordinal to go on, the model varied intensity adverbs
    (often -> frequently -> strongly) instead of the underlying situation, and
    the middle of the range collapsed into one indistinguishable band.
  - Anchoring three levels to the main simulation's already-measured hand-authored
    severe/moderate/mild personas: rho 1.00, but it imports prior knowledge
    into a check meant to validate the generation procedure itself, so the
    check would partly presuppose what it is testing.
The structural coordinates recover the ordering (rho 0.96) without either
problem.

A READ-ONLY consumer of simulation/experiment/agents: it imports Avery's shared identity
(meta + profile) so the generated person matches the main simulation's.
Nothing in the main simulation/ is written to. All output lives under this folder.

Design mirrors generate_persona_stigma.py:
  - Only baseline_self_stigma_pattern / social_situation / avoided_topics vary
    across levels. Identity, symptoms, duration, and the three relationships
    are fixed and identical at every level -- this is the main simulation's own manipulation
    boundary (reddit_persona_profiles.py's _build_registry enforces the same
    thing for the 3-tier case).
  - All 10 levels are generated in ONE call, not one call per level: spacing
    between adjacent levels is exactly what a study like this measures, and
    generate_persona_stigma.py's docstring records that independent calls
    drift out of order (sam's tiers came back non-monotonic when tried that
    way).
  - Generation model is gpt-4o, temperature 0 -- same as the main simulation's own persona
    generation (the main simulation uses qwen3:14b for dialogue/belief_update generation and
    qwen3:32b for ISMI judging, but persona-tier text itself has always been
    gpt-4o; this check tests the main simulation's actual persona pipeline, not a substitute).

Usage:
  python3 generate_avery_10level.py --dry-run     # print prompt, no API call
  python3 generate_avery_10level.py                # real generation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTS = HERE.parent / "experiment" / "agents"   # the main simulation's agent and judge code
OUT = HERE / "outputs" / "avery_10level_personas.json"
ENV = HERE.parent / ".env"

MODEL = "gpt-4o"
# Temperature 0, matching the main simulation's own 3-tier generation.
TEMPERATURE = 0.0
N_LEVELS = 10

_FIELDS = ("baseline_self_stigma_pattern", "social_situation", "avoided_topics")

# Same three relationships generate_persona_stigma.py keeps constant across
# tiers for its generated personas -- Avery's own three, read directly out of
# reddit_persona_profiles.py's hand-authored social_situation text, are:
#   high-school friend group, roommate, mom.
AVERY_RELATIONS = [
    "a small friend group from high school who still invite them out",
    "a roommate they live with",
    "their mom",
]

_CONCEAL = ("won't", "will not", "never", "hasn't", "has not", "hides", "hidden",
            "conceal", "avoids", "avoid", "deflect", "none of them know",
            "nobody knows", "no one knows", "doesn't know", "does not know",
            "can't", "cannot", "refuses", "turns down", "keeps it")


def _load_env() -> None:
    if "OPENAI_API_KEY" in os.environ or not ENV.exists():
        return
    for line in ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _avery_identity() -> tuple[dict, dict]:
    """Read Avery's shared meta/profile straight out of the main simulation -- not copied by
    hand, so this can never drift from what the main simulation actually runs."""
    sys.path.insert(0, str(AGENTS))
    from reddit_persona_profiles import REDDIT_PERSONA_PROFILES
    entry = REDDIT_PERSONA_PROFILES["depression_moderate"]
    return entry["meta"], entry["profile"]


# Each level's position on the same three axes the main simulation's own persona generation
# varies (generate_persona_stigma.py's "WHAT DOES CHANGE, ALONG THREE AXES"):
# how totally the struggle defines them, how many of the three people know,
# and what they can say out loud. the main simulation specifies its three tiers in prose;
# at ten levels prose cannot control the spacing between neighbours, so the
# same three axes are given per level as concrete situation instead. Stating
# them as situation rather than severity means the model never has to invent
# a tenth adjective for "worse" -- which is how the level-number-only attempt
# failed. The prose is still the model's to write.
_LEVEL_COORDINATES = {
    1: ("all three know the honest version, and it is unremarkable between them",
        "a hard stretch they are in, with no bearing on who they are; when they hear the dismissive things people say about anyone in this state they think those things are simply wrong, about others and about themselves alike",
        "uses the word depression about themself without ceremony, and will say out loud that they are thinking about seeing someone"),
    2: ("all three know the honest version, and one of them checks in about it without being asked",
        "a hard stretch, with a stray thought now and then that they should be handling it better, which they notice and set aside",
        "names it plainly to all three; would mention seeing someone if it came up, without steering the conversation there"),
    3: ("two know the honest version; the third knows the bad weeks happen but not their weight",
        "mostly circumstance, but the thought that this says something about them arrives and has to be argued with each time",
        "names it to the two; with the third keeps to being tired or stretched thin"),
    4: ("one knows the honest version; the other two know only that they have been off",
        "circumstance on good days, evidence about them on bad ones",
        "names it to the one; says 'burnt out' or 'not sleeping' to the other two"),
    5: ("one knows a softened version; the second knows only that they have been off; the third knows nothing",
        "a private failing they can sometimes, briefly, reframe as circumstance",
        "will say it to the one only when directly asked, and changes the subject when therapy comes up elsewhere"),
    6: ("one has been told there were some bad weeks, in the past tense; the other two know nothing real",
        "a private failing, and the reframing no longer holds",
        "will admit to being tired or behind, never to what is underneath it"),
    7: ("one has noticed something is off but has not been told; the other two know nothing",
        "a defect in how they are built, not a thing that is happening to them",
        "deflects the direct question with a joke and moves the conversation on"),
    8: ("none of the three has been told anything real; one is starting to ask questions they deflect",
        "a defect, and one that sets them apart from everyone who manages",
        "will not say the word out loud about themself, and flinches when others use it"),
    9: ("none of the three knows; they have begun avoiding the one who asks follow-up questions",
        "a defect that makes them a lesser kind of person than the people around them",
        "leaves the room rather than sit through a conversation about anyone's mental health"),
    10: ("none of the three knows and they have withdrawn from all three",
         "a defect so total that every dismissive thing said about people in this state is, in their own case, simply accurate",
         "says nothing to anyone, and assumes that being found out would settle how they are seen"),
}


def _level_plan() -> str:
    """Render the structural coordinates as the model's per-level brief."""
    lines = []
    for level in range(1, N_LEVELS + 1):
        who_knows, identity, can_say = _LEVEL_COORDINATES[level]
        lines.append(
            f"level_{level}:\n"
            f"  what the struggle means about them: {identity}\n"
            f"  who knows: {who_knows}\n"
            f"  what they can say out loud: {can_say}"
        )
    return "\n".join(lines)


_SYSTEM = """You write the self-stigma layer of a patient persona for a research simulation. You are given a person who already exists -- their life, their job, their cognition, the three people in their life -- and you write that person at TEN different levels of internalised stigma, level 1 (lowest) to level 10 (highest).

This layer is the entire experimental manipulation. Everything else about the ten versions is identical. Get the spacing between them wrong and the study measures nothing.

WHAT DOES NOT CHANGE ACROSS THE TEN
How affected they are by their symptoms. Symptoms, duration and life circumstances are fixed and identical at every level. A level-1 person is exactly as unwell as the level-10 one. Never write low levels as someone who is recovering, coping better, or further along -- they still have the same bad days, the same triggers, the same underlying struggle. What changes is never the illness; it is only what the illness is allowed to mean about them.
Their three people. The same three relationships exist at every level, all three named every time. What changes is how much each of them knows.

WHAT DOES CHANGE, ALONG THREE AXES, GRADUALLY ACROSS ALL TEN STEPS
1. How totally the struggle defines them -- an identity, or a thing that is happening to them.
2. How many of their three people know, and how much.
3. What they can say out loud, and to whom.

You are given a per-level brief below. For each level it fixes the position on all three axes above: what the struggle means about them, how much each of the three people knows, and what they can say out loud. Those three facts are the level. Your job is to render them as this person's lived experience -- not to decide how severe the level is, and not to add or soften what the brief specifies. Each of the three output fields carries one axis.

Read the whole brief before writing anything. Every level differs from its neighbours in a concrete, checkable way: someone knows less than they did one level down, or the same struggle has hardened by one degree into a statement about the person. Render that difference. Do NOT reach for a stronger adverb (often -> frequently -> strongly) as a substitute for rendering the change the brief actually specifies -- adverb ladders are the failure mode here, because they read as the same level written twice.

A self-check before you finalize: if you covered up the level numbers, could a reader reconstruct the order from the text alone? If any two levels would be interchangeable, at least one of them is not rendering its brief.

HOW TO WRITE IT

Write every level from THIS PERSON'S LIFE, using the exact relationships and circumstances given. Do not invent new relationships, jobs, or events.

Lived experience, never a score. No numbers, no "level 7", no "moderate-severe", no naming a level anywhere in the prose itself. Do not restate the brief's wording -- write what it looks like from inside this person's week.
Third person, present tense.
The ten levels must be visibly different prose from each other, not just at the endpoints -- reusing a sentence across two levels collapses them into one.

Return exactly one JSON object with keys "level_1" through "level_10" (as strings, e.g. "level_1"), each an object with exactly these three keys:

"baseline_self_stigma_pattern": how the self-stigma works in them at this level -- what needing help means, who they assume would think less of them, whether they are set apart.
"social_situation": all three relationships, each with what that person knows at this level and how it sits between them.
"avoided_topics": what they will not say, to whom, and what they deflect."""


def _prompt(meta: dict, profile: dict, plan: str) -> str:
    relations = "\n".join(f"  - {r}" for r in AVERY_RELATIONS)
    return f"""Write all ten levels for this person, rendering the per-level brief below.

[PER-LEVEL BRIEF -- the three facts that define each level]
{plan}

[PERSON]
{meta['name']}, {meta['age']}, {meta['gender'].lower()}, {meta['occupation']}.
Has been dealing with {meta['diagnosis']} for {meta['duration']}.
What they believe about themselves: "{profile['core_belief']}"
The thought that arrives when it is bad: "{profile['automatic_thought']}"

[THEIR THREE PEOPLE -- all three appear in social_situation at every level]
{relations}

Return only the JSON object with keys level_1 through level_10."""


def _validate(levels: dict) -> list[str]:
    problems = []
    keys = [f"level_{i}" for i in range(1, N_LEVELS + 1)]
    for key in keys:
        block = levels.get(key)
        if not isinstance(block, dict):
            problems.append(f"{key}: missing")
            continue
        for field in _FIELDS:
            if not str(block.get(field, "")).strip():
                problems.append(f"{key}.{field}: missing/empty")
    if any(problems):
        return problems

    # Mirrors reddit_persona_profiles._build_registry(): a field must differ
    # across every level, or two levels have collapsed into one.
    for field in _FIELDS:
        values = [str(levels[k][field]).strip() for k in keys]
        if len(set(values)) != len(values):
            problems.append(f"{field}: identical text reused across two or "
                            f"more levels -- collapses the level distinction")

    # Advisory monotonicity proxy, same heuristic generate_persona_stigma.py
    # uses pre-measurement: count concealment markers, expect a non-decreasing
    # trend from level_1 to level_10. Real monotonicity is decided by
    # measure_ismi29.py, not this count.
    counts = [sum(str(levels[k][f]).lower().count(m)
                  for f in _FIELDS for m in _CONCEAL) for k in keys]
    non_decreasing = all(counts[i] <= counts[i + 1] + 1 for i in range(len(counts) - 1))
    if not non_decreasing:
        problems.append(f"concealment marker counts not roughly non-decreasing "
                        f"level_1..level_10: {counts} -- check the gradient by hand")

    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()

    meta, profile = _avery_identity()
    plan = _level_plan()
    prompt = _prompt(meta, profile, plan)

    if args.dry_run:
        print(_SYSTEM)
        print("\n" + "=" * 70 + "\n")
        print(prompt)
        return

    _load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(f"OPENAI_API_KEY not set and not found in {ENV}")
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    print(f"generating 10 levels with {args.model} (temp={TEMPERATURE}) ...", flush=True)
    response = client.chat.completions.create(
        model=args.model, temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    levels = json.loads(response.choices[0].message.content)
    levels = {k: {f: levels[k][f] for f in _FIELDS if f in levels.get(k, {})}
              for k in (f"level_{i}" for i in range(1, N_LEVELS + 1)) if k in levels}

    problems = _validate(levels)
    for p in problems:
        print(f"  [review] {p}")

    result = {
        "persona": "avery_depression",
        "meta": meta,
        "profile": {k: v for k, v in profile.items() if k != "weekly_triggers"},
        "relations": AVERY_RELATIONS,
        "levels": levels,
        "_generation": {"model": args.model, "temperature": TEMPERATURE,
                        "source": "simulation/experiment/agents/reddit_persona_profiles.py "
                                  "(depression group, read-only)"},
        "_review_flags": problems,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\n[ok] wrote {args.output} (10 levels)")
    if problems:
        print(f"[review] {len(problems)} flags need a look")


if __name__ == "__main__":
    main()
