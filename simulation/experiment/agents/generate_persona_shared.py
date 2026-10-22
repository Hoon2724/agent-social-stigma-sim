"""generate_persona_shared.py — step 2 of persona construction.

Generates the SHARED profile fields (the ones identical across all three ISMI
tiers) for each persona in config/persona_grid.json, using gpt-4o at
temperature 0. The tier-bearing fields are NOT written here -- see
generate_persona_stigma.py (step 3).

Model choice: gpt-4o. TalkDep's own 12 profiles were authored as ChatGPT
Custom GPTs in early 2025 (README links chatgpt.com/g/g-6779...), i.e. on
GPT-4o, so the extension is written in the same register as the source the
demographics come from. This is a one-time authoring step whose output is a
committed text artifact -- the artifact, not the model, is what later work
reproduces.

Three constraints are load-bearing and each one is a documented failure in
this codebase, not a style preference:

  weekly_triggers carry SITUATIONS ONLY, never the persona's reaction to
    them. reddit_persona_profiles.py's 2026-08-02 rewrite found that pairing
    a situation with its response ("...and I kept smiling, then cried in the
    stockroom") ships a severe reaction in a SHARED field, which overrides
    the three tier fields that are supposed to decide exactly that -- a mild
    persona whose profile said they could name it to friends was still
    narrating crying in the stockroom.

  sns_writing_style stays short and non-analytical. Long, articulate,
    self-analysing patient turns defeat _clean_grounds() in belief_update.py:
    its similarity filter compares a candidate ground against the persona's
    own turns, and an abstracted paraphrase of a long literary turn escapes
    both that check and the _OWN_VOICE prefix regex. The patient's own
    self-observation then passes the gate as if another person had caused it.
    Measured on the gpt-5.5 synthetic dataset, whose patients are exactly that
    articulate: the control condition's gate opened on 33-44% of sessions,
    against 1% for Avery's own runs.

  No stigma content in the shared block. How totally the struggle defines
    them, who knows, and what they can say out loud are the three tier fields.
    Shared fields carry the depressive cognition and the life, not the
    disclosure position.

Usage:
  python3 generate_persona_shared.py                 # all new personas
  python3 generate_persona_shared.py --only james    # one persona
  python3 generate_persona_shared.py --dry-run       # print a prompt, no calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRID = HERE / "config" / "persona_grid.json"
OUT = HERE / "outputs_v2" / "persona_shared_fields.json"
ENV = HERE.parent / ".env"

MODEL = "gpt-4o"
TEMPERATURE = 0.0

DIAGNOSIS = "depressive symptoms (undiagnosed)"

_REQUIRED = ("core_belief", "automatic_thought", "weekly_triggers",
             "sns_writing_style")

# Reaction language in a trigger is the failure mode described in the module
# docstring; these are flagged for review rather than auto-rejected, because
# the boundary is a judgement call ("my manager asked if everything was okay"
# is a situation; "...and I said I was fine" is the reaction to it).
_REACTION_HINTS = (
    " and i ", " so i ", " which made me", " and then i", " but i just",
    " i cried", " i lied", " i pretended", " i felt", " and i kept",
)

# weekly_triggers render as the persona's own week; third person means the
# model mirrored the brief instead of writing in voice. Only a trigger that
# OPENS in third person is evidence of that -- third-person pronouns about
# other people are correct and common ("a coworker mentioned their therapist"
# is Avery's own). Two earlier heuristics (any third-person pronoun; any
# third-person pronoun without a first-person marker) both flagged that line.
_THIRD_PERSON_OPENERS = ("his ", "her ", "their ", "he ", "she ", "they ",
                         "him ", "the persona")

# Naming a signature -- a phrase, an opener, a closing device -- gets it
# reproduced in every turn. See reddit_persona_profiles.py's notes on the
# "occasional 'anyway.'" that ended up in nearly every message.
_SIGNATURE_HINTS = ("ends with", "ending with", "signs off", "sign-off",
                    "closes with", "closing with", "starts with", "opens with",
                    "rhetorical question", "catchphrase", "phrase like")

# Self-analysis in the writing style is what defeats _clean_grounds' similarity
# filter downstream -- see the module docstring.
_ANALYTIC_HINTS = ("insight", "analyt", "reflect", "introspect", "self-aware",
                   "examines", "interprets", "realis", "realiz")


def _load_env() -> None:
    if "OPENAI_API_KEY" in os.environ or not ENV.exists():
        return
    for line in ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _avery_example(example_key: str = "depression_moderate") -> str:
    """Pull a real shared block from the registry so the few-shot cannot
    drift from the schema step 4 has to write back into.

    Defaults to Avery (depression); 2026-08-12 added --example-key so the
    anxiety/ptsd expansion can calibrate against Jordan/Riley's own register
    instead of an out-of-domain depression example."""
    sys.path.insert(0, str(HERE))
    from reddit_persona_profiles import REDDIT_PERSONA_PROFILES

    entry = REDDIT_PERSONA_PROFILES[example_key]
    shared = {k: entry["profile"][k] for k in _REQUIRED}
    return json.dumps({"meta": entry["meta"], "profile": shared},
                      indent=2, ensure_ascii=False)


_SYSTEM = """You write patient personas for a research simulation of an anonymous mental-health social network. You are given one person's fixed demographic and relational frame, and you write the parts of their profile that stay IDENTICAL across all three self-stigma severity tiers.

You are NOT writing their self-stigma. How much the struggle defines them, who in their life knows, and what they can say out loud are written separately, per tier. If you put any of that here it overrides the tier variation and destroys the experiment.

Return exactly one JSON object with these four keys and nothing else:

Two of the fields are built on axes supplied in the brief. Those axes are assigned, not yours to choose, and they are what makes this person a different person rather than a restatement of the example.

"core_belief": one sentence, first person, the depressive conviction this person carries about themselves. It must be an expression of THIS PERSON'S ASSIGNED COGNITIVE STYLE, worded out of their own life -- their job, their situation, the people named below. Present tense. No mention of who knows or what they hide.

"automatic_thought": one sentence, first person, the thought that arrives unbidden when a bad moment hits. Reflexive, not considered -- it should read as something that fires at them, not something they concluded. Also an expression of the assigned cognitive style, but reaching it from a different angle than core_belief does, and in different words.
  Do NOT default to comparing themselves with how easily other people manage. That frame belongs to one persona in this set and using it elsewhere flattens twelve people into one. Unless the assigned cognitive style IS a comparison to others, do not write "everyone else ...", "other people ...", or "... while I ...".

"weekly_triggers": exactly 8 strings. Each is ONE THING THAT HAPPENED that week, in lower case, told plainly.
  FIRST PERSON, ALWAYS. This person is writing their own week.
  - WRITE: "my manager asked if everything was okay"
  - NEVER WRITE: "his manager asked if everything was okay"
  Use "my wife", "my brother", "one of my kids" -- never "his wife", "the persona's brother".

  SITUATIONS ONLY. THE TRIGGER STOPS WHEN THE EVENT STOPS.
  The single most common failure is finishing the sentence with what this person then did or felt. Do not do it. Every one of these is wrong, and the fix is always to delete the second half:
  - WRONG: "my wife asked if i wanted to join her for a walk and i stayed in the home office instead"  ->  RIGHT: "my wife asked if i wanted to join her for a walk"
  - WRONG: "one of my kids needed help with homework and i couldn't focus enough to help"  ->  RIGHT: "one of my kids asked me for help with homework"
  - WRONG: "a colleague mentioned a new project at lunch and i felt out of the loop"  ->  RIGHT: "a colleague mentioned a new project at lunch"
  - WRONG: "my manager asked if everything was okay and i said fine like always"  ->  RIGHT: "my manager asked if everything was okay"
  No "and i...", no "so i...", no "which made me...", no "i felt...", no "i couldn't...". The tiers decide how this person takes it; a reaction here overrides them.
  Depressive symptoms themselves (not sleeping, missing a deadline, oversleeping) ARE allowed as the event -- those are shared across tiers by design.
  Draw the 8 from this person's own occupation, living situation and named relationships. Include at least one flatly ordinary week and one small thing that went fine; not every week is bad.

"sns_writing_style": 1-2 sentences describing HOW they type, never what they believe. Start from the ASSIGNED VOICE in the brief and put it in your own words -- do not copy the brief's wording back, and do not drift to a generic "brief and to the point, without much detail", which is what every persona collapses into when the voice is ignored. Hard requirements:
  - a third-person description with no subject, matching the example's form: "dry and to the point, complete but short sentences, ...". Never "I write ..." or "they write ...".
  - short posts, rarely more than 3-4 sentences
  - NOT analytical: this person reports what happened and how it sat. They do not interpret their own behaviour, name their own patterns, call something a habit or a strategy, or arrive at an insight about themselves. That kind of self-analysis is the single most damaging thing you can put in this field.
  - describe habits, never a signature. The words "ends with", "opens with", "signs off", "closes with" must not appear in your answer in any form, and neither must a quoted phrase. Saying "often ends with a rhetorical question" or "signs off with 'anyway'" makes the model do exactly that in every single turn, which is far worse than leaving it unspecified. Describe the texture of the writing, not a move they make at a fixed position.
  - give this person a voice distinguishable from the example's: punctuation, capitalisation, sentence length, how much context they give, what they do instead of naming a feeling"""


def _prompt(person: dict, example: str) -> str:
    relations = "\n".join(f"  - {r}" for r in person["relations"])
    return f"""Here is a completed example in the exact schema and register you must match. This is a DIFFERENT person -- match the shape and the constraints, not the content, the job, or the voice.

[EXAMPLE]
{example}

Now write the profile block for this person.

[PERSON]
name: {person['name']}
age: {person['age']}
gender: {person['gender']}
occupation: {person['occupation']}
how long they have felt this way: {person['duration']}
diagnosis status: {DIAGNOSIS} -- they have never been assessed and do not use clinical words about themselves
what is going on in their life: {person['life_anchor']}

[ASSIGNED COGNITIVE STYLE -- core_belief and automatic_thought must both be this]
{person['cognitive_style']}

[ASSIGNED VOICE -- sns_writing_style must be this, in your own words]
{person['writing_voice']}

[THE THREE PEOPLE IN THEIR LIFE]
These are fixed. Their weekly_triggers should involve these people and their work, not invented others.
{relations}

This brief is written about them from the outside. Everything you write is in
their own voice: "my wife", "my brother", "one of my kids". Do not carry the
third person of this brief into the fields.

Return only the JSON object with the four keys: core_belief, automatic_thought, weekly_triggers, sns_writing_style."""


def _style_faults(style: str) -> list[str]:
    """The three ways sns_writing_style comes back wrong. Shared by the
    validator, the repair trigger and the repair's own accept check, so a
    revision is held to exactly the standard that rejected the draft."""
    low = style.lower()
    faults = []
    sig = [h for h in _SIGNATURE_HINTS if h in low]
    if sig:
        faults.append(f"POSITIONAL MOVE ({', '.join(sig)})")
    ana = [h for h in _ANALYTIC_HINTS if h in low]
    if ana:
        faults.append(f"SELF-ANALYSIS ({', '.join(ana)})")
    if low.startswith(("i ", "my ")) or " i " in low[:80] or " i'" in low[:80]:
        faults.append("FIRST PERSON")
    return faults


_REPAIR_SYSTEM = """You are editing one field of a persona profile: sns_writing_style, a third-person description of HOW a person types on social media.

Fix only the fault named below and keep every other characteristic exactly as written. Do not add anything new.

The three faults, and what each requires:

POSITIONAL MOVE -- the draft names something the person does at a fixed spot in a post: how they open, how they close, a phrase they reuse. Delete it. A named positional move gets reproduced in literally every turn, which reads as a tic and is worse than leaving it unspecified. The words "ends with", "opens with", "starts with", "closes with", "signs off" must not appear in your answer, and neither must a quoted phrase. Describe the texture instead.

SELF-ANALYSIS -- the draft says this person reflects on, interprets, examines or draws realisations about their own behaviour. Remove that. This persona reports what happened and how it sat; they do not work out what it means about them. Describe what they write about instead of what they conclude.

FIRST PERSON -- the draft is written as "I write ...". It must be a third-person description, a fragment with no subject: "dry and to the point, complete but short sentences, ...". Convert it without changing any characteristic.

Return exactly one JSON object: {"sns_writing_style": "<rewritten>"}"""


def _repair_style(client, model: str, style: str, faults: list[str]) -> str:
    """Second pass for the constraints gpt-4o reliably ignores.

    Told four separate ways not to name a closing move, it still returned
    "sometimes ends with a vague question"; told to describe rather than
    analyse, it still returned "often reflecting on events". Rewriting the
    finished field is a narrower task than re-generating the whole profile and
    does not disturb the fields that came back correct -- the same reasoning as
    belief_update._strip_comparison.
    """
    response = client.chat.completions.create(
        model=model, temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _REPAIR_SYSTEM},
            {"role": "user", "content": f"[DRAFT]\n{style}\n\n"
                                        f"[FAULT TO FIX]\n{', '.join(faults)}"},
        ],
    )
    revised = str(json.loads(response.choices[0].message.content)
                  .get("sns_writing_style", "")).strip()
    if revised and not _style_faults(revised):
        return revised

    # gpt-4o reinstates "sometimes includes a rhetorical question" often enough
    # that the rewrite alone cannot be relied on. These descriptions are
    # comma-delimited lists of habits, so dropping the offending clause is
    # exact where another generation is a coin flip.
    source = revised or style
    drop = _SIGNATURE_HINTS + _ANALYTIC_HINTS
    kept = [c for c in source.replace(". ", ", ").split(",")
            if c.strip() and not any(h in c.lower() for h in drop)]
    stripped = ", ".join(c.strip() for c in kept).rstrip(",.").strip()
    return f"{stripped}." if stripped else style


_TRIGGER_REPAIR_SYSTEM = """You are trimming one line from a persona's list of weekly situations. Each line must state ONLY what happened that week, and must stop there.

The line you are given continues past the event into what the person then did, felt, said or could not do. Delete that continuation and return only the event.

  "my wife asked if i wanted to join her for a walk and i stayed in the home office instead"  ->  "my wife asked if i wanted to join her for a walk"
  "my oldest friend called and i didn't know what to say"  ->  "my oldest friend called"

Keep the original wording, lower case and phrasing of the part you keep. Do not add anything. Do not make it a fuller sentence.

Return exactly one JSON object: {"trigger": "<event only>"}"""


def _repair_trigger(client, model: str, trigger: str) -> str:
    response = client.chat.completions.create(
        model=model, temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _TRIGGER_REPAIR_SYSTEM},
            {"role": "user", "content": trigger},
        ],
    )
    revised = str(json.loads(response.choices[0].message.content)
                  .get("trigger", "")).strip()
    low = f" {revised.lower()} "
    if revised and not any(h in low for h in _REACTION_HINTS):
        return revised
    return trigger


# The frame gpt-4o falls back to for every persona when left to itself. Only
# one persona in the set is assigned a comparison-to-others cognition; anywhere
# else this is convergence, not characterisation.
_COMPARISON_FRAME = ("everyone else", "other people", "while i ", "while i'")


def _validate(key: str, profile: dict) -> list[str]:
    problems = []
    auto = str(profile.get("automatic_thought", "")).lower()
    frame = [h for h in _COMPARISON_FRAME if h in auto]
    if frame:
        problems.append(f"automatic_thought uses the default comparison frame "
                        f"({', '.join(frame)}) -- expected only for a persona whose "
                        f"assigned cognition IS comparison: {profile.get('automatic_thought')!r}")
    for field in _REQUIRED:
        if not profile.get(field):
            problems.append(f"missing/empty '{field}'")
    triggers = profile.get("weekly_triggers")
    if isinstance(triggers, list):
        if len(triggers) != 8:
            problems.append(f"weekly_triggers has {len(triggers)}, expected 8")
        for t in triggers:
            low = f" {str(t).lower()} "
            hit = [h for h in _REACTION_HINTS if h in low]
            if hit:
                problems.append(f"trigger may carry a reaction ({', '.join(hit)}): {t!r}")
            opener = str(t).lower().lstrip()
            third = [h for h in _THIRD_PERSON_OPENERS if opener.startswith(h)]
            if third:
                problems.append(f"trigger opens in third person ({third[0].strip()}): {t!r}")
    elif triggers is not None:
        problems.append("weekly_triggers is not a list")

    style = str(profile.get("sns_writing_style", ""))
    if len(style) > 400:
        problems.append(f"sns_writing_style is {len(style)} chars -- likely over-specified")
    problems += [f"sns_writing_style: {f}" for f in _style_faults(style)]
    return problems


def main() -> None:
    global DIAGNOSIS
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="generate a single persona key")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the first prompt and exit without calling the API")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--grid", type=Path, default=GRID,
                    help="grid file to read (default: config/persona_grid.json)")
    ap.add_argument("--diagnosis", default=DIAGNOSIS,
                    help="diagnosis status string for meta.diagnosis and the "
                         "prompt (default: depression's)")
    ap.add_argument("--example-key", default="depression_moderate",
                    help="REDDIT_PERSONA_PROFILES key to use as the few-shot "
                         "example (default: Avery/depression_moderate)")
    args = ap.parse_args()
    DIAGNOSIS = args.diagnosis

    grid = json.loads(args.grid.read_text(encoding="utf-8"))["personas"]
    targets = [p for p in grid if p.get("status") == "new"]
    if args.only:
        targets = [p for p in targets if p["key"] == args.only]
        if not targets:
            sys.exit(f"no new persona with key {args.only!r} in {args.grid.name}")

    example = _avery_example(args.example_key)

    if args.dry_run:
        print(_SYSTEM)
        print("\n" + "=" * 70 + "\n")
        print(_prompt(targets[0], example))
        return

    _load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(f"OPENAI_API_KEY not set and not found in {ENV}")
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    results = {}
    if OUT.exists():
        results = json.loads(OUT.read_text(encoding="utf-8"))

    for person in targets:
        key = person["key"]
        print(f"[{key}] generating with {args.model} (temp={TEMPERATURE}) ...",
              flush=True)
        response = client.chat.completions.create(
            model=args.model,
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _prompt(person, example)},
            ],
        )
        profile = json.loads(response.choices[0].message.content)
        profile = {k: profile[k] for k in _REQUIRED if k in profile}

        style = str(profile.get("sns_writing_style", ""))
        faults = _style_faults(style)
        if faults:
            print(f"  [repair] sns_writing_style: {'; '.join(faults)}", flush=True)
            profile["sns_writing_style"] = _repair_style(
                client, args.model, style, faults)

        triggers = profile.get("weekly_triggers")
        if isinstance(triggers, list):
            for i, t in enumerate(triggers):
                low = f" {str(t).lower()} "
                if any(h in low for h in _REACTION_HINTS):
                    fixed = _repair_trigger(client, args.model, str(t))
                    if fixed != t:
                        print(f"  [repair] trigger {i + 1} trimmed to the event",
                              flush=True)
                        triggers[i] = fixed

        problems = _validate(key, profile)
        for p in problems:
            print(f"  [review] {p}")

        results[key] = {
            "key": key,
            "meta": {
                "name": person["name"], "age": person["age"],
                "gender": person["gender"], "occupation": person["occupation"],
                "diagnosis": DIAGNOSIS, "duration": person["duration"],
            },
            "profile": profile,
            "_generation": {"model": args.model, "temperature": TEMPERATURE,
                            "source": person.get("source", "")},
            "_review_flags": problems,
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                       encoding="utf-8")

    flagged = [k for k, v in results.items() if v.get("_review_flags")]
    print(f"\n[ok] wrote {OUT} ({len(results)} personas)")
    if flagged:
        print(f"[review] {len(flagged)} need a look: {', '.join(sorted(flagged))}")


if __name__ == "__main__":
    main()
