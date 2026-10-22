"""Write each persona's session-0 belief update, once, to config/session0_beliefs.json.

This is session 0 of the belief trajectory (belief_update.initial_selfreport
stores it verbatim) and it is what the pre measurement is administered to, so
pre and every later session are the same kind of material in the same voice --
which is the whole point: an ISMI delta between them is a difference in the
person, not a difference between two generators.

Replaces make_pre_narratives.py (2026-08-13). Two things changed:

  The name. It was called a "pre narrative", which read as a separate artifact
  produced for the measurement. It is not -- it is the first belief update,
  measured like all the others, and calling it that keeps session 0 and
  sessions 1..N from drifting apart in anyone's head.

  The frame. The old prompt was handed facts + avoided_topics and asked for
  "how you see yourself". baseline_self_stigma_pattern -- the field that
  actually carries this persona's stigma, and the one the tier manipulation is
  written into -- never reached it, so the tier had to survive into session 0
  through avoided_topics alone. It now arrives, in the shape of Corrigan's
  Application stage.

The frame, concretely. Corrigan & Rao (2012) run Awareness -> Agreement ->
Application -> Harm; what separates this study's three tiers is Application,
how completely the stereotype is taken as true OF THIS PERSON (see
persona_simple.STIGMA_TEMPLATE_BY_TIER, reused here rather than restated).
So the prompt gives the persona the stereotype content their own profile
carries and asks what of it applies to them, and then how much of that they
let anyone see (Corrigan & Rao's disclosure hierarchy, the same ladder
belief_update tracks from session 1 on).

The five ISMI axes are NOT written into this prompt (2026-08-14: tried
feeding each axis's gloss in explicitly, scaled by tier -- reverted).
Spelling out the instrument's own subscale definitions ("whether you feel
like a different and lesser kind of person...") and asking the model to
cover them is asking it to write toward the measure's own vocabulary, which
risks the eventual ISMI score reading how well the generator echoed its
prompt rather than anything about the persona. The persona's own stigma
content already lives in facts_block (core_belief, automatic_thought,
baseline_self_stigma_pattern) and in STIGMA_TEMPLATE_BY_TIER's Application
shape below; the five axes are left to emerge from that material however
they do, the same way a real person's ISMI-29 responses aren't drafted
against the subscale definitions either. belief_update.py (sessions 1..N)
follows the same rule -- no axis classification anywhere in the pipeline,
only measure_ismi29.py reads the finished text and scores it post hoc.

Generated once per persona and cached. Session 0 has no run-specific input, so
regenerating it per run would put pure sampling noise into the baseline every
condition is measured against -- observed directly in batch09, where severe
pre ranged 2.759-2.92 across runs of the same tier for no reason tied to the
condition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from belief_update import _COMPARISON, _strip_comparison
from llm_client import OpenAICompatibleModel
from persona_simple import STIGMA_TEMPLATE_BY_TIER, _tier_of, facts_block
from reddit_persona_profiles import REDDIT_PERSONA_PROFILES
from strategy_selector import complete_json

HERE = Path(__file__).resolve().parent

_PROMPT = """You are writing a private note to yourself about how you see yourself right now.

[WHO YOU ARE]
{facts}

[THE PEOPLE IN YOUR LIFE AND WHAT THEY KNOW]
{social}

[WHAT YOU KEEP AWAY FROM]
{avoided}

Nobody has said anything to you about any of this. You have not talked to
anyone about it and you are not in treatment. This is just where you are.

[HOW FAR THE THINGS SAID ABOUT PEOPLE IN THIS STATE APPLY TO YOU]
This is the part to get exactly right -- it is what separates you from
someone else in the same situation. {application}

Write 5-7 first-person sentences, in the present tense. Cover it the way it
actually sits in you: what of this you think is true of you, what you expect
people would actually do differently if they knew, how much you keep away
from people because of it, whether you can hold that it does not settle what
you are worth, and how much of it you let anyone see. Only where each of
those is genuinely part of where you are -- do not work through them as a
list, and do not invent a position on one you have no feeling about.

Ground it in the specific material above rather than restating it word for
word. Write plainly, the way a person writes privately -- unpolished, no
metaphors or literary imagery. Do not compare this to how you used to be: no
"better", "worse", "than before", "lately", "starting to", "no longer".
There is nothing to compare to yet. Do not mention scores, questionnaires,
therapy jargon, or this instruction.

Return exactly one JSON object:
{{"belief_update": "<5-7 first-person sentences>"}}"""


def build_prompt(key: str) -> str:
    detail = REDDIT_PERSONA_PROFILES[key]["profile"]

    def _text(field: str, fallback: str) -> str:
        value = detail.get(field) or fallback
        if isinstance(value, (list, tuple)):
            return "; ".join(str(x) for x in value)
        return str(value)

    return _PROMPT.format(
        # facts_block already carries core_belief, automatic_thought AND
        # baseline_self_stigma_pattern, so the stigma content is not repeated
        # as a block of its own here -- it arrives once, and the tier's
        # Application shape below says how completely it lands.
        facts=facts_block(key),
        social=_text("social_situation", "(nobody in particular)"),
        avoided=_text("avoided_topics", "(nothing in particular)"),
        # The tier's Application shape, verbatim from the registry that the
        # dialogue pipeline also reads -- so session 0 and the persona's own
        # tier cannot drift apart.
        application=STIGMA_TEMPLATE_BY_TIER[_tier_of(key)],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--personas", nargs="*",
                    default=["depression_severe", "depression_moderate",
                             "depression_mild"])
    ap.add_argument("--out", type=Path,
                    default=HERE / "config" / "session0_beliefs.json")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the first prompt and exit without calling the API")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate personas that already have an entry")
    a = ap.parse_args()

    if a.dry_run:
        print(build_prompt(a.personas[0]))
        return

    model = OpenAICompatibleModel(a.model, a.base_url, num_ctx=a.num_ctx, num_gpu=99)
    existing = json.loads(a.out.read_text(encoding="utf-8")) if a.out.exists() else {}

    for key in a.personas:
        if key in existing and not a.overwrite:
            print(f"[skip] {key} already present")
            continue
        prompt = build_prompt(key)
        text = ""
        for _ in range(3):
            try:
                parsed = complete_json(model, [{"role": "user", "content": prompt}],
                                       temperature=0.7)
                text = str(parsed.get("belief_update", "")).strip()
                if text:
                    break
            except (ValueError, KeyError, TypeError):
                continue
        if not text:
            raise SystemExit(f"failed to generate a session-0 belief for {key}")

        # Same guard sessions 1..N get, applied here for the same reason. The
        # prompt bans comparison language and 6 of 27 carried it anyway on the
        # first pass ("better", "no longer", "used to") -- session 0 has
        # nothing to compare to, so any of it reads to the instrument as
        # improvement that has already happened before the run starts. This is
        # a rewrite rather than a regeneration on purpose: measured on the
        # session 1..N path, three full regenerations still left the language
        # in 8 of 12 cases, because regenerating asks the model to solve the
        # same problem the same way again (see belief_update._strip_comparison).
        hits = [h for h in _COMPARISON.findall(text) if h]
        if hits:
            revised = _strip_comparison(model, text, hits)
            if revised != text:
                print(f"  [strip] {key}: removed {sorted(set(hits))}", flush=True)
                text = revised

        existing[key] = text
        print(f"\n=== {key} ===\n{text}\n", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
    print(f"wrote {a.out} ({len(existing)} personas)")


if __name__ == "__main__":
    main()
