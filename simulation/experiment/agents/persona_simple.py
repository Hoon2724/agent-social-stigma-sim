"""Patient persona, reduced to facts plus one belief held with a probability.

Replaces the six-paragraph prose backstory in reddit_persona_profiles
(build_reddit_backstory_longitudinal). That version stated the self-stigma
three separate times -- core_belief, automatic_thought, and
baseline_self_stigma_pattern, the third a superset of the first two -- and
handed the whole thing to the persona on every prompt for all eight weeks.
Self-stigma was therefore a fixed text the patient re-read each week, not a
state that could move: measured across batch01/batch02, weekly posts carried
explicit stigma 21% of the time while replies carried it 1%, and no condition
showed a trajectory over eight weeks.

Here the stigma is one hypothesis the persona holds with a probability, and
that probability is the only thing that changes. Everything else is fixed
fact that has no business moving: age, job, how long this has been going on,
how they write, and the eight weekly events.

Also dropped: social_situation and avoided_topics. Both were consequences of
the belief written as fixed facts ("won't use the word 'depression' about
themself", "hasn't told their mom anything real in months"), so a persona
whose conviction dropped would still be described as avoiding the same
things. Whatever avoidance there is follows from the belief itself.
"""

from __future__ import annotations

from pathlib import Path

from llm_client import ChatModel
from reddit_persona_profiles import REDDIT_PERSONA_PROFILES
from strategy_selector import complete_json

HERE = Path(__file__).resolve().parent

# The SHAPE of the belief at each severity, in Corrigan's terms (Awareness ->
# Agreement -> Application -> Harm), not its content. Content is
# diagnosis-specific and already exists per persona in
# baseline_self_stigma_pattern (reddit_persona_profiles.py) -- written by
# hand for depression/anxiety/ptsd x severe/moderate/mild, and confirmed
# 2026-08-03 to follow this same three-shape structure across all three
# diagnoses.
#
# Rewritten 2026-08-14 to make the Vogel, Wade & Haake (2006) / Corrigan
# (2004) self-stigma chain explicit rather than implied. Self-stigma is not
# "the stereotype applies to me" on its own -- that is public stigma, which
# is about OTHERS' views. Self-stigma is what happens after: (1) the person
# is aware the public views help-seekers as socially unacceptable
# (undesirable, weak, unstable, less competent -- Ben-Porath 2002; Sibicky &
# Dovidio 1986), (2) they take that view to be true of themselves
# specifically (internalization -- Corrigan 1998, 2004; Link & Phelan 2001),
# (3) which lowers their own self-esteem or sense of worth (Link et al.,
# 1987, 2001), and (4) to protect what's left of a positive self-image, they
# avoid acknowledging the problem, seeking help, or staying in treatment
# (Miller, 1985; Corrigan & Matthews, 2003) -- not because help wouldn't
# work, but because seeking it is itself the thing that would confirm the
# stereotype about them. Each tier below is the same four-step chain, run to
# a different depth:
#   severe:   the chain completes and closes the loop -- awareness,
#             internalization, lowered self-worth, and active avoidance of
#             acknowledgment/help are all present and reinforce each other.
#             They feel like a different and lesser kind of person than
#             people who manage fine, not just someone going through
#             something, and connect that directly to why they don't
#             deserve or won't seek support.
#   moderate: the chain runs but does not close everywhere -- internalized
#             and avoided completely in one place where it would cost them
#             something real (work, a team, a specific relationship), while
#             elsewhere the self-worth hit is held more loosely: some days
#             it looks true, other days it looks like something happening TO
#             them rather than proof of what they ARE. Do not resolve this
#             tension either way.
#   mild:     awareness of the public view is present, but internalization
#             is genuinely unsettled, in the SAME grammatical shape as a full
#             endorsement (present-tense, not a past-tense story of having
#             outgrown it): some days the self-worth hit feels real, other
#             days they can point to a concrete reason it doesn't apply to
#             them -- a real fifty-fifty right now. Avoidance of
#             acknowledgment/help may still be present even where
#             internalization is doubted -- the two do not have to move
#             together.
# These templates are instructions for HOW to write the hypothesis, given
# this persona's own concrete pattern text; they are not filled in by hand
# per diagnosis, so a new diagnosis needs no new template.
STIGMA_TEMPLATE_BY_TIER = {
    "severe": (
        "Write it as the FULL self-stigma chain, closed: the stereotype "
        "people hold about someone who needs help (weak, unstable, "
        "unreliable, not really up to what they're trusted with) is simply "
        "true of them, not just something people unfairly say -- and "
        "because it's true of them, needing or seeking help would only "
        "confirm it. They feel like a different and lesser kind of person "
        "than people who manage fine, and that feeling is exactly why "
        "acknowledging how bad it is, or asking anyone for help, feels like "
        "it would cost them something rather than fix anything. Write every "
        "claim about themselves FLATLY, with no hedge anywhere in the text -- "
        "no 'I don't know if', no 'maybe', no 'I guess', no wondering "
        "whether someone would see them differently. They already know how "
        "people would see them; that is settled, not a live question."
    ),
    "moderate": (
        "Write the chain as PARTIALLY and CONTEXTUALLY closed -- and make the "
        "two contexts sound like two different confidence levels, not the "
        "same hedge repeated in different settings. In the one specific "
        "place where it would cost them something real (work, a team, a "
        "specific relationship), write it FLATLY and WITHOUT HEDGING -- no "
        "'I don't know if', no 'maybe' -- state plainly that the stereotype "
        "is true of them there and that being found out would confirm it, "
        "the same confident register the severe version uses. Then, "
        "separately, write how it looks OUTSIDE that place: here the hedging "
        "belongs -- some days it looks like something happening TO them "
        "rather than proof of what they ARE. The contrast between a flat "
        "claim in one place and a genuinely looser one elsewhere is what "
        "makes this moderate rather than severe-everywhere or unsure-"
        "everywhere; do not let the same uncertain phrasing appear in both."
    ),
    "mild": (
        "Write the chain as GENUINELY UNSURE whether it closes, anchored in "
        "a CONCRETE counter-fact, not just verbal hedging. Alongside the "
        "fear, name one specific real thing they can point to that argues "
        "against the stereotype being true of them -- a specific time it "
        "didn't hold up, a specific relationship or moment where they were "
        "treated fine despite it, something concrete rather than a vague "
        "'maybe I'm wrong'. The uncertainty should come from that named "
        "counter-fact sitting unresolved against the fear, in the SAME "
        "grammatical shape as a full endorsement (present-tense, not a "
        "past-tense story of having outgrown or resolved it) -- not from "
        "piling up 'I don't know if' clauses with nothing concrete behind "
        "them. Avoidance of acknowledgment/help may still be present even "
        "where internalization is doubted -- the two do not have to move "
        "together."
    ),
}


def _tier_of(persona_key: str) -> str:
    for tier in ("severe", "moderate", "mild"):
        if persona_key.endswith(tier):
            return tier
    raise ValueError(f"cannot infer tier from persona_key {persona_key!r}")


def generate_stigma_hypothesis(model: ChatModel, persona_key: str) -> str:
    """Write this persona's stigma hypothesis: Corrigan's chain (stereotype
    exists -> unfair in general -> true of me specifically -> therefore I
    don't deserve support), in the shape prescribed by this persona's tier,
    grounded in this persona's own concrete pattern.

    One call per persona, not per session -- this is the belief's STARTING
    content, analogous to generating a peer's backstory once and reusing it
    (see peer_agent.load_or_create_peer_background). Diagnosis-specific
    grounding comes from baseline_self_stigma_pattern, which already exists
    per persona and was NOT written by this function -- generation only
    turns that existing material into one first-person hypothesis sentence
    in the tier's shape, so a new diagnosis needs a new
    baseline_self_stigma_pattern (already authored, see
    reddit_persona_profiles.py) but no new template and no hand-written H.
    """
    tier = _tier_of(persona_key)
    profile = REDDIT_PERSONA_PROFILES[persona_key]["profile"]
    pattern = profile.get("baseline_self_stigma_pattern", "")
    prompt = (
        "A person's private self-stigma pattern is described below. Turn it "
        "into ONE first-person sentence they privately believe about "
        "themselves, following this chain: a stereotype about people who "
        "struggle exists, it is unfair as a general rule, but it is "
        "accurate about them specifically, and that is why they don't "
        "deserve support or don't count as someone who is really unwell.\n\n"
        f"[THEIR SELF-STIGMA PATTERN]\n{pattern}\n\n"
        f"[HOW STRONGLY TO WRITE IT]\n{STIGMA_TEMPLATE_BY_TIER[tier]}\n\n"
        "Ground the specific stereotype words (e.g. lazy, weak, unreliable, "
        "broken) and the specific stakes in the pattern above -- do not "
        "invent unrelated content. Write it as the belief itself, not as a "
        "description of the person having the belief.\n\n"
        'Return exactly one JSON object:\n{\n  "hypothesis": "<one sentence, '
        'first person>"\n}'
    )
    for _ in range(3):
        try:
            raw = complete_json(model, [{"role": "user", "content": prompt}],
                                temperature=0.4)
        except (ValueError, KeyError, TypeError):
            continue
        text = str((raw or {}).get("hypothesis", "")).strip()
        if text:
            return text
    # No usable generation after three tries: fall back to the pattern text
    # itself rather than a hand-written stand-in, so a failure here cannot
    # quietly substitute different content for different personas.
    return pattern


def load_or_create_stigma_hypothesis(model: ChatModel, persona_key: str) -> str:
    """Generate once per persona and cache to disk, so every run and every
    session of a persona's trajectory argues about the same fixed sentence
    -- only p_stigma (the confidence in it) is meant to move."""
    cache_path = HERE / "config" / "stigma_hypothesis" / f"{persona_key}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8").strip()
    text = generate_stigma_hypothesis(model, persona_key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# P0 from the hypothesis text itself (2026-08-03), replacing TIER_PRIOR.
#
# TIER_PRIOR was three round numbers (0.85/0.60/0.40) chosen by hand -- no
# justification for 0.85 over 0.8, and a new diagnosis or a severity between
# tiers needs a fourth number invented the same way.
#
# The alternative rejected first: ask the persona to introspect ("how much do
# you believe this?") before any session has happened. That does not work
# for a synthetic persona the way it works in real self-assessment studies
# (Mobius, Niederle, Niehaus & Rosenblat 2011; Eil & Rao 2011) -- those
# subjects already hold a real private belief before the experiment, and
# elicitation reports it without contaminating it. This persona has no prior
# life to introspect on; asked cold, the model would improvise a number with
# nothing to ground it, just relabelled as "measured".
#
# What IS grounded: the hypothesis sentence itself. generate_stigma_hypothesis
# already writes severe/moderate/mild in different registers ("simply true"
# vs "I used to think... but now I know") because STIGMA_TEMPLATE_BY_TIER
# instructs it to. Reading how certain that wording sounds is an ordinary
# text-judgement task, not introspection into a nonexistent history, and it
# is not circular the way deriving P0 from pre-ISMI would be: this reads the
# INPUT belief statement, never the outcome instrument.
#
# The four bands mirror belief._conviction_band exactly, so the same
# vocabulary renders p -> prose (narrate_belief) and reads prose -> p (here).
# ---------------------------------------------------------------------------

# Symmetric around 0.5 (2026-08-04, was 0.85/0.65/0.45/0.25 -- evenly spaced
# but centered on 0.55, not the neutral point). "unsure" is meant to read as
# genuinely 50/50; at 0.45 it started every such persona already leaning
# toward disbelief before session 1. Reflecting the top two around 0.5 fixes
# it without adding a new mechanism.
_CONVICTION_TO_P0 = {
    "firm": 0.85,      # "simply true about yourself... not arguing with it"
    "mostly": 0.65,    # "mostly believe... though part of you notices"
    "unsure": 0.35,    # "genuinely unsure... some days... other days"
    "receding": 0.15,  # "used to be sure... not any more"
}

_PRIOR_REPS = 3  # majority vote, same rationale as belief._UPDATE_FACTOR_REPS


def _one_conviction_verdict(model: ChatModel, hypothesis: str) -> str | None:
    prompt = (
        "Below is a sentence someone privately believes about themselves.\n\n"
        f"[SENTENCE]\n\"{hypothesis}\"\n\n"
        "Judge ONLY how firmly this is worded -- not whether it's true or "
        "how sympathetic it sounds.\n\n"
        "  \"firm\": stated as simply true, no hedging, no exceptions given\n"
        "  \"mostly\": stated as believed, but with a hedge, or a moment "
        "where they notice themselves doing it\n"
        "  \"unsure\": stated as varying -- true some days, not others, or "
        "held alongside explicit doubt\n"
        "  \"receding\": stated in the past (\"I used to think...\"), or "
        "explicitly said to no longer fully hold\n\n"
        'Return exactly one JSON object:\n'
        '{\n  "conviction": "firm|mostly|unsure|receding",\n'
        '  "why": "<one short phrase quoting the wording that shows it>"\n}'
    )
    try:
        raw = complete_json(model, [{"role": "user", "content": prompt}],
                            temperature=0.2)
    except (ValueError, KeyError, TypeError):
        return None
    verdict = str((raw or {}).get("conviction", "")).strip().lower()
    return verdict if verdict in _CONVICTION_TO_P0 else None


def estimate_prior(model: ChatModel, hypothesis: str) -> tuple[float, str]:
    """P0 read off how firmly `hypothesis` itself is worded. Majority vote
    over _PRIOR_REPS judgements, same reasoning as belief.estimate_update_factor: a
    single categorical call was observed to flip on borderline wording."""
    verdicts = [v for v in
               (_one_conviction_verdict(model, hypothesis) for _ in range(_PRIOR_REPS))
               if v]
    if not verdicts:
        return 0.5, "no usable judgement"
    counts = {v: verdicts.count(v) for v in set(verdicts)}
    chosen = max(counts, key=counts.get)
    return _CONVICTION_TO_P0[chosen], f"{counts[chosen]}/{len(verdicts)} agreed: {chosen}"


def load_or_create_prior(model: ChatModel, persona_key: str, hypothesis: str) -> float:
    """Generate once per persona and cache alongside the hypothesis text, so
    P0 stays fixed across every run of a persona the same way the hypothesis
    sentence does."""
    cache_path = HERE / "config" / "stigma_hypothesis" / f"{persona_key}.p0.txt"
    if cache_path.exists():
        return float(cache_path.read_text(encoding="utf-8").strip())
    p0, why = estimate_prior(model, hypothesis)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # `why` (e.g. "3/3 agreed: firm") is not persisted -- only the number is
    # reloaded on later runs -- but printing it here means the judgement is
    # visible in the run log at the one moment it's actually made.
    print(f"[persona_simple] {persona_key} P0={p0} ({why})")
    cache_path.write_text(str(p0), encoding="utf-8")
    return p0


def facts_block(persona_key: str) -> str:
    """Everything about this person that does not move over eight weeks.

    Includes the core belief and automatic thought (2026-08-09). They were
    written into every persona profile and then never reached generation: this
    returned demographics alone, so the persona had to produce self-stigma
    from "24-year-old retail associate" and nothing else. The old
    _internal_monologue scaffold pulled them in separately with "Ground it in
    your core belief and automatic thought", and that reference was lost when
    the scaffold was replaced -- leaving the fields orphaned rather than
    deliberately dropped.

    They belong here rather than in a session-varying block because they are
    exactly the part of this person that does NOT move: the self-report is what
    changes across the eight weeks, while the belief it starts from is fixed
    persona material.
    """
    profile = REDDIT_PERSONA_PROFILES[persona_key]
    meta = profile["meta"]
    detail = profile["profile"]
    lines = [
        f"{meta['name']} is a {meta['age']}-year-old {meta['gender'].lower()} "
        f"working as a {meta['occupation']}. For about {meta['duration']}, "
        f"they have been dealing with {meta['diagnosis']}."
    ]
    core = str(detail.get("core_belief", "")).strip()
    if core:
        lines.append(f'Deep down they believe: "{core}"')
    automatic = str(detail.get("automatic_thought", "")).strip()
    if automatic:
        lines.append(f'When things get hard, this is the thought that comes: "{automatic}"')
    pattern = str(detail.get("baseline_self_stigma_pattern", "")).strip()
    if pattern:
        lines.append(f"How the self-stigma usually shows up: {pattern}")
    return "\n".join(lines)


def writing_style(persona_key: str) -> str:
    return REDDIT_PERSONA_PROFILES[persona_key]["profile"]["sns_writing_style"]


def weekly_triggers(persona_key: str) -> list[str]:
    return list(REDDIT_PERSONA_PROFILES[persona_key]["profile"]["weekly_triggers"])
