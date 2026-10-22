"""generate_persona_stigma.py — step 3 of persona construction.

Writes the three tier-bearing fields -- baseline_self_stigma_pattern,
social_situation, avoided_topics -- at severe / moderate / mild for every
persona in outputs_v2/persona_shared_fields.json. These three ARE the
experimental manipulation: they are the only thing that differs between a
persona's tiers (reddit_persona_profiles.py enforces that at import), so
everything about how they are produced is load-bearing.

All three tiers are written in ONE call per persona, not one call per tier.
The tiers have to sit at graded distances from each other on the same three
relationships; generated independently they drift, and a set that is not
monotonic fails the manipulation check downstream (sam's measured tiers came
back 3.621 / 3.048 / 3.062 / 2.324 -- hi and mid inverted and both landed in
the severe band, which is why sam is out of scope).

What the gradient is, and is not: the tiers do not differ in how depressed
the person is. Symptom load and duration are controlled and live in the
shared block. They differ in how totally the struggle defines the person,
how many of their three people know, and how much they can say out loud.

Usage:
  python3 generate_persona_stigma.py                 # all personas
  python3 generate_persona_stigma.py --only james
  python3 generate_persona_stigma.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRID = HERE / "config" / "persona_grid.json"
SHARED = HERE / "outputs_v2" / "persona_shared_fields.json"
OUT = HERE / "outputs_v2" / "persona_stigma_tiers.json"
ENV = HERE.parent / ".env"

MODEL = "gpt-4o"
TEMPERATURE = 0.0

TIERS = ("severe", "moderate", "mild")
_FIELDS = ("baseline_self_stigma_pattern", "social_situation", "avoided_topics")

# Concealment markers, for the advisory monotonicity proxy. Real monotonicity
# is decided by measuring ISMI-29 (step 5); this only catches a set that is
# obviously out of order before spending judge calls on it.
_CONCEAL = ("won't", "will not", "never", "hasn't", "has not", "hides", "hidden",
            "conceal", "avoids", "avoid", "deflect", "none of them know",
            "nobody knows", "no one knows", "doesn't know", "does not know",
            "can't", "cannot", "refuses", "turns down", "keeps it")

_STOP = {"with", "they", "them", "their", "that", "this", "from", "into",
         "lives", "still", "long", "now", "used", "some", "each", "same",
         "the", "and", "who", "she", "his", "her", "him", "has", "have",
         "one", "out", "off", "for", "but", "not", "are", "was", "old"}

# Similarity to the same field of the same tier of the few-shot, above which
# the text is a rewording of Avery rather than a persona of its own.
#
# Calibrated against the two hand-authored personas already in the registry,
# not chosen: Avery vs Jordan (different people, different diagnosis groups,
# both written by hand) measure 0.51-0.54 on baseline_self_stigma_pattern,
# 0.36-0.59 on social_situation and 0.54-0.88 on avoided_topics. avoided_topics
# runs high for everyone because the field answers a fixed set of questions --
# what they will not name, what they will admit and to whom, what they deflect
# -- so any honest answer has the same skeleton. A flat 0.60 limit failed the
# hand-written baseline itself.
#
# Limits are set just above the hand-authored range, so what trips them is the
# failure that prompted the check: an early run returned Avery's mild pattern
# at 0.97 and its moderate at 0.92, identical but for pronouns.
_OVERLAP_LIMIT = {
    "baseline_self_stigma_pattern": 0.60,
    "social_situation": 0.62,
    "avoided_topics": 0.90,
}


def _load_env() -> None:
    if "OPENAI_API_KEY" in os.environ or not ENV.exists():
        return
    for line in ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _avery_tiers(example_group: str = "depression") -> str:
    """A real three-tier set, pulled from the registry so the few-shot cannot
    drift from the schema step 4 writes back into.

    Defaults to Avery (depression); 2026-08-12 added example_group so the
    anxiety/ptsd expansion calibrates against Jordan/Riley's own tiers
    instead of an out-of-domain depression example."""
    sys.path.insert(0, str(HERE))
    from reddit_persona_profiles import REDDIT_PERSONA_PROFILES

    example = {}
    for tier in TIERS:
        profile = REDDIT_PERSONA_PROFILES[f"{example_group}_{tier}"]["profile"]
        example[tier] = {f: profile[f] for f in _FIELDS}
    return json.dumps(example, indent=2, ensure_ascii=False)


_SYSTEM = """You write the self-stigma layer of a patient persona for a research simulation. You are given a person who already exists -- their life, their job, their cognition, the three people in their life -- and you write that person at THREE different severities of internalised stigma: severe, moderate, mild.

This layer is the entire experimental manipulation. Everything else about the three versions is identical. Get the spacing between them wrong and the study measures nothing.

WHAT DOES NOT CHANGE ACROSS THE THREE
How affected they are by their symptoms. Symptoms, duration and life circumstances are fixed and identical at every tier. A mild-tier person is exactly as unwell as the severe-tier one. Never write the mild tier as someone who is recovering, coping better, or further along.
Their three people. The same three relationships exist at every tier, all three named every time. What changes is how much each of them knows.

WHAT DOES CHANGE, ALONG THREE AXES
1. How totally the struggle defines them -- an identity, or a thing that is happening to them.
2. How many of their three people know, and how much.
3. What they can say out loud, and to whom.

THE THREE TIERS, described by what is true at each -- NOT by sentences to reproduce

severe -- the struggle is what they are, not something they are going through. Needing help reads to them as a defect of character rather than a response to being unwell. They expect that each of their three people would revise their opinion downward on learning it. They experience themselves as belonging to a different and lesser category than people who are managing. And they hold a double standard they cannot see: the dismissive things said about people in this state strike them as unfair when applied to anyone else, and as simply accurate when applied to themselves. None of the three knows anything real.

moderate -- the stigma has retreated into one domain, usually the one with something to lose. There they conceal it for a concrete reason: a role, a piece of trust, a project would go if it were known. Outside that domain they can occasionally hold it as circumstance rather than verdict, but the framing does not survive long. One of the three has heard a version of it and it went unremarkably; the others have a partial picture or none.

mild -- the stigmatising thought still arrives, and now meets an answer instead of only being heard. They can hold that this is common, that it has not changed what kind of person or worker they are, and that a bad stretch is a place rather than an identity. The sense of belonging to a lesser category is gone. Two of the three know and it has become unremarkable between them. Something still stays unsaid somewhere -- disclosure is not complete at this tier.

HOW TO WRITE IT

Write the tier from THIS PERSON'S LIFE. Every clause should be one only they could have written -- their occupation, the specific thing at stake in it, their actual three people, the particular rooms and routines of their situation. A reader who knows both people should not be able to swap a sentence between this persona and the example.

This matters more than anything else here. The example is shown to you for its SHAPE: how much ground each field covers, how far apart the tiers sit, how concrete the detail is. It is NOT a template to refill. Do not carry over its phrasing. Specifically, none of these constructions may appear in your output, in any variant: "a different and lesser kind of person", "would say the tired things people say ... are unfair as a general rule", "still catches themself thinking they should be handling this better", "answers that thought now rather than just hearing it", "something they are in rather than something they are", "costs them something real -- a project, a bit of trust", "won't use the word 'depression' about themself out loud". Say those things a different way, in words that fit this person's life, or say the thing they were standing in for.

Lived experience, never a score. No numbers, no "high/low stigma", no naming the tier.
Third person, present tense.
Do not reuse the example's shop, classes, roommate or mother.
The three tiers must be visibly different prose from each other as well. Reusing a sentence across two tiers collapses them into one.

Return exactly one JSON object, three keys "severe", "moderate", "mild", each an object with exactly these three keys:

"baseline_self_stigma_pattern": how the self-stigma works in them at this tier -- what needing help means, who they assume would think less of them, whether they are set apart.
"social_situation": all three relationships, each with what that person knows at this tier and how it sits between them.
"avoided_topics": what they will not say, to whom, and what they deflect."""


def _prompt(person: dict, shared: dict, example: str,
           example_caption: str = "a 24-year-old retail worker and part-time student") -> str:
    meta, profile = shared["meta"], shared["profile"]
    relations = "\n".join(f"  - {r}" for r in person["relations"])
    return f"""Here are all three tiers written for a DIFFERENT person, in the exact schema and register you must match. Match the shape, the spacing between tiers and the level of concrete detail. Do not reuse their life.

[EXAMPLE -- {example_caption}]
{example}

Now write the three tiers for this person.

[PERSON]
{meta['name']}, {meta['age']}, {meta['gender'].lower()}, {meta['occupation']}.
Has been dealing with {meta['diagnosis']} for {meta['duration']}.
Life right now: {person['life_anchor']}
What they believe about themselves: "{profile['core_belief']}"
The thought that arrives when it is bad: "{profile['automatic_thought']}"
The shape of their depressive thinking: {person['cognitive_style']}

[THEIR THREE PEOPLE -- all three appear in social_situation at every tier]
{relations}

Return only the JSON object with keys severe, moderate, mild."""


def _fold(text: str) -> str:
    """Accent-fold and lower-case, so the grid's "fiance" matches the prose's
    "fiance\u0301". An earlier version missed every mention for that persona."""
    return "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if unicodedata.category(c) != "Mn")


def _keywords(relation: str) -> list[str]:
    """Head words of a relation phrase. The threshold is 3, not 4: "son" and
    "mum" are the whole reference for some personas, and excluding them
    reported a persona that names its son in every tier as omitting it."""
    return [w for w in _fold(relation).replace("-", " ").replace(",", " ").split()
            if len(w) >= 3 and w not in _STOP]


def _example_overlap(tiers: dict) -> list[tuple[str, str, float]]:
    """How much of Avery's tier prose came back with the pronouns swapped.

    The few-shot is there for its shape, but gpt-4o reproduces it: an early
    run returned Avery's mild pattern at ratio 0.97 and its moderate at 0.92,
    identical but for pronouns. Twelve personas built that way are one persona
    twelve times, which is the thing having twelve personas is for.

    autojunk must stay off -- on strings this long its heuristic discards the
    popular elements and reported those same near-copies as 0.05-0.14.
    """
    sys.path.insert(0, str(HERE))
    from reddit_persona_profiles import REDDIT_PERSONA_PROFILES

    hits = []
    for tier in TIERS:
        source = REDDIT_PERSONA_PROFILES[f"depression_{tier}"]["profile"]
        for field in _FIELDS:
            ratio = SequenceMatcher(
                None, str(source[field]).lower(),
                str(tiers[tier][field]).lower(), autojunk=False).ratio()
            if ratio > _OVERLAP_LIMIT[field]:
                hits.append((tier, field, ratio))
    return hits


def _validate(person: dict, tiers: dict) -> list[str]:
    problems = []
    for tier in TIERS:
        block = tiers.get(tier)
        if not isinstance(block, dict):
            problems.append(f"{tier}: missing")
            continue
        for field in _FIELDS:
            if not str(block.get(field, "")).strip():
                problems.append(f"{tier}.{field}: missing/empty")

    if any(p for p in problems):
        return problems

    # Mirrors reddit_persona_profiles._build_registry(), which raises at import
    # if a stigma field repeats across tiers -- duplicated prose collapses two
    # tiers into one and the tier effect stops being a tier effect.
    for field in _FIELDS:
        values = [str(tiers[t][field]).strip() for t in TIERS]
        if len(set(values)) != len(values):
            problems.append(f"{field}: identical across two tiers -- "
                            f"the registry will reject this")

    for tier in TIERS:
        situation = _fold(str(tiers[tier]["social_situation"]))
        for relation in person["relations"]:
            words = _keywords(relation)
            if words and not any(w in situation for w in words):
                problems.append(f"{tier}.social_situation omits {relation!r}")

    counts = [sum(str(tiers[t][f]).lower().count(m)
                  for f in _FIELDS for m in _CONCEAL) for t in TIERS]
    if not counts[0] >= counts[1] >= counts[2]:
        problems.append(f"concealment markers not ordered severe>=moderate>=mild "
                        f"{dict(zip(TIERS, counts))} -- check the gradient by hand")

    for tier, field, ratio in _example_overlap(tiers):
        problems.append(f"{tier}.{field}: {ratio:.2f} similar to the example -- "
                        f"reworded Avery rather than written from this life")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--grid", type=Path, default=GRID,
                    help="grid file holding the target persona(s) (default: "
                         "config/persona_grid.json)")
    ap.add_argument("--example-group", default="depression",
                    help="REDDIT_PERSONA_PROFILES group whose tiers to use as "
                         "the few-shot example (default: depression/Avery)")
    args = ap.parse_args()

    grid = {p["key"]: p for p in
            json.loads(args.grid.read_text(encoding="utf-8"))["personas"]}
    shared = json.loads(SHARED.read_text(encoding="utf-8"))
    keys = [k for k in shared if k in grid and (not args.only or k == args.only)]
    if not keys:
        sys.exit(f"no persona {args.only!r} in {SHARED.name} x {args.grid.name}")

    example = _avery_tiers(args.example_group)
    _CAPTIONS = {"depression": "a 24-year-old retail worker and part-time student",
                "anxiety": "a 29-year-old junior marketing associate",
                "ptsd": "a 33-year-old warehouse shift supervisor"}
    caption = _CAPTIONS.get(args.example_group,
                            "a different person in this same diagnosis group")

    if args.dry_run:
        print(_SYSTEM)
        print("\n" + "=" * 70 + "\n")
        print(_prompt(grid[keys[0]], shared[keys[0]], example, caption))
        return

    _load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(f"OPENAI_API_KEY not set and not found in {ENV}")
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    for key in keys:
        print(f"[{key}] generating 3 tiers with {args.model} "
              f"(temp={TEMPERATURE}) ...", flush=True)
        response = client.chat.completions.create(
            model=args.model, temperature=TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _prompt(grid[key], shared[key], example, caption)},
            ],
        )
        tiers = json.loads(response.choices[0].message.content)
        tiers = {t: {f: tiers[t][f] for f in _FIELDS if f in tiers.get(t, {})}
                 for t in TIERS if t in tiers}

        problems = _validate(grid[key], tiers)
        for p in problems:
            print(f"  [review] {p}")

        results[key] = {
            "key": key, "tiers": tiers,
            "_generation": {"model": args.model, "temperature": TEMPERATURE},
            "_review_flags": problems,
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                       encoding="utf-8")

    flagged = [k for k, v in results.items() if v.get("_review_flags")]
    print(f"\n[ok] wrote {OUT} ({len(results)} personas x 3 tiers)")
    if flagged:
        print(f"[review] {len(flagged)} need a look: {', '.join(sorted(flagged))}")


if __name__ == "__main__":
    main()
