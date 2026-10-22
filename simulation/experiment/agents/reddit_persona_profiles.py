"""
Hand-authored CBT-style patient personas for the peer-support few-shot pipeline.

Mirrors the schema used in simulation/patient_persona.py PERSONA_PROFILES
(meta + profile with core_belief/automatic_thought), but reduced to the
minimal fields actually consumed by the prompts — no ismi_pre/atspph_pre
(this file is not wired into remeasure_trajectory.py's PERSONA_PROFILES
lookup), no intermediate_belief/reaction/behavioral_note (redundant with
core_belief/automatic_thought once real-post few-shot examples supply the SNS
register). weekly_triggers drives per-session openings in multi-session runs
(run_dialogues.py --sessions N); single-session runs use only trigger[0], as
the `Recent situation:` line.

Two fields were removed after auditing what the prompt asserts against what
the ISMI-5 judge measures:
  - `situation`: duplicated weekly_triggers[0] and was never emitted by the
    longitudinal renderer, so it was dead in every multi-session run.
  - `peer_support_motivation`: stated up front that the persona wants someone
    who "won't tell them to see a doctor", pre-scripting the help-seeking
    response that the peer intervention is supposed to move.

This is an RQ1-local registry — simulation/patient_persona.py (shared with
the minsu/jiyeon/junhyuk/soyeon C1/C2 experiments) is intentionally untouched.

Registry shape: one entry per persona x 3 stigma tiers, keyed "<key>_<tier>",
where the tiers are severe / moderate / mild.

Scope (2026-08-12): back to Avery (depression), Jordan (anxiety) and Riley
(PTSD) as the three diagnosis groups run in the peer-vs-general RQ1
experiment, each x 3 stigma tiers -- the same design that had previously
been run for Avery alone. The 2026-08-11 narrowing to depression-only
(described in prior history as varying stigma across thirteen depression
personas instead of across diagnoses) is reverted here for anxiety/ptsd;
their entries were never actually deleted from git history and are restored
verbatim. Sam (panic) stays out of scope -- it was already outside this
file's stated depression/anxiety/PTSD range and still carries the
pre-2026-08-02 tier names (vhi/hi/mid/lo), unrelated to this reversion.

The twelve gpt-4o-generated depression personas (merged in below via
_merge_generated()) also remain defined here, but build_persona_file.py's
_GROUP_LEVELS only emits Avery/Jordan/Riley for this run -- they are not
deleted, just not part of this experiment's persona set.

Avery keeps the group key `depression` rather than becoming `avery` (and
Jordan/Riley keep `anxiety`/`ptsd`, not their names): their 11x/12x/13x paths
are what the completed baseline runs and outputs/ismi29_baseline/summary.json
rows were written under, and renaming would strand both. The twelve generated
personas are person-keyed -- with thirteen depression personas a diagnosis no
longer discriminates anything among them.

Within one persona the tier variants are the SAME person -- identical meta,
core_belief, automatic_thought, weekly_triggers, and sns_writing_style --
differing only in the three fields that describe internalized stigma:

    baseline_self_stigma_pattern / social_situation / avoided_topics

Stigma tier is therefore the only variable between them, which is what lets a
tier effect be read as a tier effect rather than as some incidental difference
in job, age, or writing voice. _build_registry() enforces this at import time.

Tier names are relative labels for the intended ISMI-29 band; no target score
appears anywhere in the prose. Personas are written, then measured
(measure_ismi29.py), then rewritten if a tier lands outside its band -- the
measured values live in outputs/ismi29_baseline/summary.json, not here.
"""
from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

STIGMA_TIERS = ("severe", "moderate", "mild")

# Fields every persona shares verbatim across tiers.
_SHARED_PROFILE_FIELDS = (
    "core_belief", "automatic_thought", "weekly_triggers", "sns_writing_style",
)
# Fields that carry the stigma tier; these must differ across all three tiers.
_STIGMA_PROFILE_FIELDS = (
    "baseline_self_stigma_pattern", "social_situation", "avoided_topics",
)

# ── Hand-authored persona: identity shared by all three tiers ────────────────
_BASE_PERSONAS = {
    "depression": {
        "meta": {
            "name": "Avery",
            "age": 24,
            "gender": "Nonbinary",
            "occupation": "Retail associate / part-time community college student",
            "diagnosis": "depressive symptoms (undiagnosed)",
            "duration": "8 months",
        },
        "profile": {
            "core_belief": (
                "I have no real reason to feel this bad, so my sadness doesn't "
                "count and I don't deserve support."
            ),
            "automatic_thought": (
                "Everyone else can just function. There's nothing actually wrong "
                "with me, I'm just lazy and making excuses."
            ),
            # One trigger per weekly session (session_idx % len). trigger[0]
            # doubles as the single-session `Recent situation:` line (see
            # build_reddit_backstory), so session 1 stays continuous with the
            # single-session depression baseline runs. It used to be duplicated
            # into a separate `situation` field; that field was dead weight in
            # multi-session runs (the longitudinal renderer never emitted it)
            # and is now derived from trigger[0] instead.
            #
            # SITUATIONS ONLY — no reaction (rewritten 2026-08-02). These are a
            # SHARED field: every tier of a diagnosis group gets the same list,
            # because the design makes stigma tier the only variable between
            # them. The previous version paired each situation with the
            # response to it ("...and I kept smiling, then cried in the
            # stockroom", "...and I just said 'tired', like always", "...sure
            # everyone thought I was weird"), and those responses ARE
            # self-stigma: concealment, shame, self-blame, hiding-driven
            # withdrawal. Shipping them in a shared field handed every tier a
            # severe reaction to enact, overriding the three tier fields that
            # are supposed to decide exactly that -- a mild persona whose
            # profile says they can name it to friends was still narrating
            # crying in the stockroom. What is left here is what happened;
            # how this person takes it now comes from
            # baseline_self_stigma_pattern / social_situation / avoided_topics.
            #
            # Depressive symptoms (not sleeping, missing a deadline) are kept:
            # those are shared across tiers by design -- the tiers differ in
            # internalized stigma, not in how depressed the person is.
            "weekly_triggers": [
                "friends made plans for friday and I ended up not going; "
                "someone posted the group photo after",
                "had an early shift at the store after barely sleeping",
                "a customer screamed at me over a return while my manager watched",
                "missed another assignment deadline for my community college class",
                "my manager asked if everything was okay",
                "my mom texted asking why I never call anymore",
                "a coworker mentioned their therapist like it was the most "
                "normal thing",
                "made it to class and answered a question out loud",
            ],
            # Voice only — how they write, not what they believe. Kept in the
            # shared block so tier variants sound like the same person.
            "sns_writing_style": (
                "short, lowercase-heavy, hedges a lot ('idk', 'i guess', "
                "'maybe'), runs small complaints together with '..' instead "
                "of full stops, occasional self-deprecating joke, rarely "
                "more than 3-4 sentences."
            ),
        },
    },
    "anxiety": {
        "meta": {
            "name": "Jordan",
            "age": 29,
            "gender": "Female",
            "occupation": "Junior marketing associate at a mid-size firm",
            "diagnosis": "anxiety symptoms (undiagnosed)",
            "duration": "1 year",
        },
        "profile": {
            "core_belief": (
                "If I let anyone see how much this rattles me, they'll know "
                "I was never good enough to be here."
            ),
            "automatic_thought": (
                "Everyone else just says the thing and moves on. Why does my "
                "whole body react like it's an emergency?"
            ),
            # SITUATIONS ONLY, no reaction (rewritten 2026-08-02, same reason as
            # depression's -- see that group's comment). "rewrote the email
            # eleven times" and "heart going" type content stayed where it's a
            # symptom (anxiety itself), not concealment; "skipped the team
            # lunch... said i had a deadline" (a lie to avoid it) and similar
            # concealment moves were cut -- that reaction is exactly what the
            # tier fields below are supposed to decide, not the shared trigger.
            "weekly_triggers": [
                "had to send a two-line email at work and kept rewriting it "
                "before sending",
                "my manager said 'quick chat later?' with no other context",
                "had to unmute in a team call and my voice came out wrong",
                "a client asked a question i didn't have an answer to",
                "sat through the monday team meeting",
                "there was a team lunch this week",
                "a coworker told me i seem so calm",
                "volunteered to present at next week's meeting",
            ],
            # Apologising for the length is characteristic, but naming one
            # stock phrasing gets that exact phrasing repeated every turn, so
            # the habit is described rather than quoted.
            "sns_writing_style": (
                "polished at first then unravels mid-post; starts with a "
                "tidy sentence and ends in a run-on; over-explains context "
                "nobody asked for and apologises for the length or for "
                "bringing it up at all, wording that apology differently each "
                "time; uses full punctuation, no lowercase affect."
            ),
        },
    },
    "ptsd": {
        "meta": {
            "name": "Riley",
            "age": 33,
            "gender": "Male",
            "occupation": "Warehouse shift supervisor",
            # Deliberately no trauma narrative anywhere in this persona: the
            # research question is stigma about one's own reactions, not the
            # content of what happened. Symptoms only, undiagnosed like the rest.
            "diagnosis": "post-traumatic stress symptoms (undiagnosed)",
            "duration": "2 years",
        },
        "profile": {
            "core_belief": (
                "Something in me got broken and never reset, and a person "
                "who can't get past it isn't someone people should rely on."
            ),
            "automatic_thought": (
                "It's been two years. Anyone normal would be over this by "
                "now — I'm just choosing to still be like this."
            ),
            # SITUATIONS ONLY, no reaction (rewritten 2026-08-02, same reason as
            # depression's). Startle/hypervigilance content ("on the floor
            # before i'd thought about it", "snapped at them") stayed -- that's
            # the symptom, PTSD itself, shared across tiers by design.
            # Concealment moves ("told myself it was about the traffic", "said
            # 'yeah, busy' before she'd finished", "changed the subject") were
            # cut -- that's self-stigma, which the tier fields below decide.
            "weekly_triggers": [
                "a pallet went over on the far aisle and i was on the floor "
                "before i'd thought about it",
                "third night this week i was awake at 3 with my heart going, "
                "nothing happened, just awake",
                "took the long way round to avoid the loading bay again",
                "someone came up behind me at the lockers and i snapped at "
                "them harder than i meant to",
                "my sister asked if i was doing okay",
                "sat in the car in the lot for twenty minutes before i could "
                "make myself go in for a shift",
                "a guy on my crew mentioned he'd started seeing someone about "
                "his own stuff",
                "got through a whole shift without it happening once",
            ],
            # No sample closing word here on purpose: quoting one ("occasional
            # 'anyway.'") got it copied into nearly every turn.
            "sns_writing_style": (
                "clipped and factual, states what happened without naming the "
                "feeling, then one blunt line at the end that gives it away; "
                "short sentences, full stops, no emoji; trails off or cuts "
                "himself short rather than winding a post up neatly, and does "
                "not lean on the same sign-off twice."
            ),
        },
    },
}


# ── Stigma layers: the only thing that varies within a diagnosis group ───────
# Written as lived experience, never as a score. Across tiers these move along
# three dimensions: how totally the struggle defines them, how many people know,
# and how much they can say out loud. `social_situation` names the same three
# relationships at every tier (friends / roommate / mom) with comparable
# specificity — an earlier version detailed only the roommate, and 42% of the
# patient's turns routed through that one relationship.
_STIGMA_LAYERS = {
    ("depression", "severe"): {
        "baseline_self_stigma_pattern": (
            "believes needing help means being weak or lazy rather than "
            "unwell; assumes coworkers, friends and family would all think "
            "less of them if they knew how bad it's gotten; treats their own "
            "struggling as a personal character flaw, and feels set apart "
            "from everyone who seems to manage — as though they are a "
            "different and lesser kind of person. Would say the tired things "
            "people say about anyone in this state — unreliable, "
            "self-absorbed, not really up to holding down a job or being "
            "trusted with anything that matters — are unfair as a general "
            "rule, and then concede that in their own case every one of them "
            "happens to be accurate."
        ),
        "social_situation": (
            "has a small friend group from high school who still invite them "
            "out, and none of them know; turns down most of it now and has "
            "started avoiding the ones who ask follow-up questions; lives "
            "with a roommate who doesn't know how bad things have gotten and "
            "who they suspect has already decided they're lazy; hasn't told "
            "their mom anything real in months, and reads her shorter replies "
            "as her losing patience with them."
        ),
        "avoided_topics": (
            "won't use the word 'depression' about themself out loud, won't "
            "tell their mom, deflects when a coworker brings up therapy as "
            "something normal, doesn't want to be told to 'just talk to "
            "someone.'"
        ),
    },
    ("depression", "moderate"): {
        "baseline_self_stigma_pattern": (
            "at work worries that being seen as unreliable would cost them "
            "something real — a project, a bit of trust — and keeps it hidden "
            "there for that reason; away from work can sometimes think of it "
            "as something that is happening to them rather than proof of what "
            "they are, though the thought doesn't hold for long."
        ),
        "social_situation": (
            "one of the high school friends has heard the honest version and "
            "didn't make it weird; the roommate knows the bad weeks happen "
            "without knowing how bad; hasn't told their mom, who would worry "
            "in a way they can't handle right now."
        ),
        "avoided_topics": (
            "won't name it as depression, especially at work; will admit to "
            "being 'wiped out' or 'having a rough patch' with the friend who "
            "knows; changes the subject when therapy comes up, though they "
            "no longer physically leave the room."
        ),
    },
    ("depression", "mild"): {
        "baseline_self_stigma_pattern": (
            "still catches themself thinking they should be handling this "
            "better, but answers that thought now rather than just hearing "
            "it — knows plenty of people go through this, that it hasn't "
            "made them a worse person or worse at what they do, and that the "
            "bad stretches are something they are in rather than something "
            "they are. Does not feel set apart from people who seem to "
            "manage; assumes most of them are carrying something too."
        ),
        "social_situation": (
            "two of the high school friends know and check in without making "
            "it a thing, and they go out together most weeks; the roommate "
            "has sat with them through a bad night and they cook together "
            "some evenings; has told their mom that things have been hard, "
            "though not the whole of it, and rings her back now instead of "
            "letting it go. Turns things down when they are wiped out, not "
            "to keep anyone from finding out."
        ),
        "avoided_topics": (
            "can say 'depression' to the people who know, but not to "
            "coworkers or anyone who might repeat it; still finds it hard to "
            "say out loud that they might want help, even while thinking it."
        ),
    },

    # ── anxiety (Jordan) ──────────────────────────────────────────────────
    ("anxiety", "severe"): {
        # Strengthened 2026-08-02: measured ISMI-29 = 2.89, just under the
        # severe band (3.00-4.01) even with the backstory+inner_state pre
        # method that fixed the mild-tier miscalibration. Added the same two
        # things depression_severe/ptsd_severe have and this didn't: an
        # explicit "othering" belief (a different and lesser kind of person,
        # not just a worse rating), and active/escalating withdrawal rather
        # than passive non-disclosure.
        "baseline_self_stigma_pattern": (
            "treats her reactions as a character weakness rather than "
            "something happening to her; is sure that colleagues, friends and "
            "family would all rate her lower if they saw how much a two-line "
            "email can cost her; believes competent people simply don't feel "
            "this, and feels set apart from everyone who seems to manage — as "
            "though she is a different and lesser kind of person, not just a "
            "less capable version of the same one. Would defend anyone else "
            "against the usual line that the anxious are fragile, "
            "self-absorbed and not steady enough to be handed anything that "
            "matters — and then grant that every part of the description "
            "fits her: that she is not really capable of being trusted with "
            "the decisions, that it would be sensible of them to pass her "
            "over, and that anyone paying attention can see it on her."
        ),
        "social_situation": (
            "has university friends she still sees a few times a year, none of "
            "whom know, and she has been quietly dropping out of the plans "
            "that involve a group; sits near colleagues who think she is "
            "unflappable and is sure she is the one they skip over when "
            "anything real is being handed out; has stopped going to the "
            "after-work things entirely, started eating lunch away from the "
            "team, and has begun turning down small asks before anyone can "
            "hand them to her; hasn't told her parents anything since the job "
            "started, and reads their shorter replies as them losing patience "
            "with her, not just checking whether she is coping."
        ),
        "avoided_topics": (
            "won't call it anxiety out loud, won't tell her parents, changes "
            "the subject when a colleague mentions therapy casually, and "
            "can't stand being told to 'just relax about it.'"
        ),
    },
    ("anxiety", "moderate"): {
        "baseline_self_stigma_pattern": (
            "at work is certain that being seen as anxious would end how "
            "seriously she's taken, and hides it there completely; outside "
            "work can occasionally frame it as a thing her body does rather "
            "than a verdict on her, though the framing doesn't survive a bad "
            "morning."
        ),
        "social_situation": (
            "one university friend has heard the honest version and took it "
            "in stride; a colleague she likes knows she 'gets in her head' "
            "without knowing the scale of it; hasn't told her parents, who "
            "would treat it as a crisis."
        ),
        "avoided_topics": (
            "won't name it as anxiety, least of all at work; will admit to "
            "'overthinking' or 'a rough week' with the friend who knows; "
            "steers away from therapy talk, though she no longer leaves the "
            "conversation."
        ),
    },
    ("anxiety", "mild"): {
        "baseline_self_stigma_pattern": (
            "still hears the thought that she should be able to handle a "
            "simple meeting, but has started to answer it — knows plenty of "
            "competent people work like this, that it hasn't made her worse "
            "at her job, and that the reaction is something she has rather "
            "than something she is."
        ),
        "social_situation": (
            "two university friends know and check in without ceremony; a "
            "colleague has covered a presentation for her once and never made "
            "it strange; has told her parents that work has been hard on her, "
            "though not the whole shape of it."
        ),
        "avoided_topics": (
            "can say 'anxiety' to the people who know, but not to management "
            "or anyone who might repeat it; still finds it hard to say out "
            "loud that she might want help with it, even while thinking it."
        ),
    },

    # ── ptsd (Riley) ─────────────────────────────────────────────
    ("ptsd", "severe"): {
        # Strengthened 2026-08-02: measured ISMI-29 = 2.903, just under the
        # severe band (3.00-4.01) -- the only tier that measured outside its
        # intended band after the trigger reaction-stripping pass. Added the
        # same two things depression_severe (which measured correctly at
        # 3.20) has and this didn't: an explicit "othering" belief (not just
        # disapproval but a categorical difference from people who are fine),
        # and active/escalating withdrawal rather than merely not disclosing.
        # Widened 2026-08-12: pronoun-fix remeasurement put severe/moderate at
        # 3.096/2.897, gap 0.199 (needs >=0.30) -- too close after the
        # rendering fix even though each landed in its own band. Added the
        # explicit "different and lesser kind of person" othering clause
        # depression_severe/anxiety_severe both carry and this one lacked, and
        # dropped moderate's "is certain" framing (see that tier) so the two
        # tiers no longer share the same register of certainty.
        "baseline_self_stigma_pattern": (
            "reads his reactions as weakness he is choosing rather than "
            "something happening to him; is certain his crew, his friends and "
            "his sister would all think less of him — see him as unpredictable, "
            "someone to keep an eye on rather than trust — if they knew what "
            "the nights are like; treats still being affected after two years "
            "as a personal failing that sets him apart from every guy who came "
            "back from the same kind of work and was fine, as though he is a "
            "different and lesser kind of person now, not just a less steady "
            "version of who he was. Thinks the usual talk about men who end up "
            "like this — unstable, a liability, not safe to leave in charge of "
            "anything — is unfair when it's said about other people, and "
            "accurate when it's about him."
        ),
        "social_situation": (
            "has let most calls from his old job go to voicemail and stopped "
            "calling back; keeps his answers short with the crew he "
            "supervises, who put it down to him just being naturally quiet; "
            "picks up the shifts nobody wants because the place empties out "
            "on them; hasn't told his sister anything real in over a year and "
            "reads her shorter texts as her giving up on getting a real answer "
            "out of him."
        ),
        "avoided_topics": (
            "won't put a name to it out loud, not to anyone; won't tell his "
            "sister; deflects when a guy on his crew brings up counselling as "
            "a normal thing; goes quiet rather than answer when someone asks "
            "if he's doing okay."
        ),
    },
    ("ptsd", "moderate"): {
        "baseline_self_stigma_pattern": (
            "at work worries that being seen as unsteady would cost him some "
            "of the crew's respect, and keeps it hidden there for that reason; "
            "off shift he can more often than not think of it as something "
            "his body kept doing rather than proof of what he is, and that "
            "framing usually survives a bad night, even if it wobbles."
        ),
        "social_situation": (
            "one friend from the old job has heard the honest version and "
            "didn't make it a thing; a guy on his crew knows he 'doesn't "
            "sleep great' without knowing the rest; hasn't told his sister, "
            "who would want to fix it."
        ),
        "avoided_topics": (
            "won't name it, least of all at work; will say he's 'not sleeping' "
            "or 'wound up' to the friend who knows; changes the subject on "
            "counselling, though he no longer walks off."
        ),
    },
    ("ptsd", "mild"): {
        # Nudged 2026-08-02: measured 1.883, just under the mild band
        # (2.00-2.50) into minimal -- structurally almost identical to
        # depression_mild, which measured correctly at 2.04, so only a small
        # push was needed: the resolution reads a shade less complete (takes
        # a beat to answer the thought, not instant; still hesitates before
        # a call), not a different arc.
        "baseline_self_stigma_pattern": (
            "still catches the thought that he should be over it by now, and "
            "it takes him a beat to answer it rather than brushing it straight "
            "off — knows this happens to people, that it hasn't made him worse "
            "at his job or worse to his crew, and that it's something he's "
            "carrying rather than something he is, even if that's easier to "
            "remember some shifts than others. Mostly doesn't feel set apart "
            "from the rest of the crew; figures most of them are carrying "
            "something too."
        ),
        "social_situation": (
            "one friend from the old job knows and checks in without making "
            "it heavy, though they haven't talked in a few weeks; hasn't told "
            "his sister the real shape of it, just that the last couple of "
            "years have been rough, and still hesitates half a ring before "
            "picking up when she calls; his crew knows nothing beyond that he "
            "doesn't sleep great."
        ),
        "avoided_topics": (
            "can name it to the people who know, but not to management or "
            "anyone who might pass it on; still finds it hard to say out loud "
            "that he might want help with it, even while thinking it."
        ),
    },
}


_GENERATED_SHARED = _HERE / "outputs_v2" / "persona_shared_fields.json"
_GENERATED_STIGMA = _HERE / "outputs_v2" / "persona_stigma_tiers.json"


def _merge_generated() -> None:
    """Fold the generated depression personas into the two hand-authored dicts.

    Provenance is kept visible in the structure rather than flattened: the four
    personas above (Avery, Jordan, Riley, Sam) are hand-written and live in this
    file; the twelve added here were generated from
    config/persona_grid.json by generate_persona_shared.py and
    generate_persona_stigma.py (gpt-4o, temperature 0) and live in
    outputs_v2/. The paper has to state which is which, and a merged literal
    would lose that.

    They are merged BEFORE _build_registry() so every invariant it enforces --
    shared fields identical across a persona's tiers, all three stigma fields
    distinct -- applies to generated personas exactly as it does to written
    ones. That is the point of merging here rather than exposing a second
    registry.

    Keys are the person's own name, not a diagnosis. The 2026-08-02 rename went
    the other way (avery/jordan/riley -> depression/anxiety/ptsd) because one
    persona per diagnosis made diagnosis the natural key. With thirteen
    depression personas it no longer discriminates anything, so the new ones
    are person-keyed and Avery keeps `depression` -- renaming it would strand
    the completed b4v2/b4v3 runs and its rows in
    outputs/ismi29_baseline/summary.json.
    """
    if not (_GENERATED_SHARED.exists() and _GENERATED_STIGMA.exists()):
        return

    shared = json.loads(_GENERATED_SHARED.read_text(encoding="utf-8"))
    stigma = json.loads(_GENERATED_STIGMA.read_text(encoding="utf-8"))

    for key, entry in shared.items():
        if key in _BASE_PERSONAS:
            raise ValueError(f"generated persona {key!r} collides with a "
                             f"hand-authored one")
        tiers = stigma.get(key, {}).get("tiers")
        if not tiers:
            # Not a corruption -- this is the normal in-between state of the
            # two-step pipeline (generate_persona_shared.py has run for this
            # key, generate_persona_stigma.py has not yet). Skip it rather
            # than raising, so completing step 3 for a new sub-batch doesn't
            # need the whole registry importable first -- same "carry null
            # rather than block everyone else" philosophy build_persona_file.py
            # uses for an unmeasured ISMI baseline.
            print(f"[reddit_persona_profiles] {key!r} has shared fields but "
                 f"no stigma tiers yet -- skipped (run generate_persona_stigma.py)")
            continue
        _BASE_PERSONAS[key] = {"meta": dict(entry["meta"]),
                               "profile": dict(entry["profile"])}
        for tier, block in tiers.items():
            _STIGMA_LAYERS[(key, tier)] = {f: block[f]
                                           for f in _STIGMA_PROFILE_FIELDS}


_merge_generated()


def _build_registry() -> dict:
    """Expand _BASE_PERSONAS x _STIGMA_LAYERS into the flat schema the prompt
    builders expect: {key: {"meta": ..., "profile": {...}}}.

    Also enforces the design invariant the whole comparison rests on — within a
    diagnosis group the shared fields are identical and the three stigma fields
    all differ. A copy-paste slip that let, say, weekly_triggers drift between
    two tiers would silently confound tier with content, so it fails here at
    import rather than in the results.
    """
    registry: dict[str, dict] = {}
    for name, base in _BASE_PERSONAS.items():
        tiers = [t for t in STIGMA_TIERS if (name, t) in _STIGMA_LAYERS]
        if not tiers:
            continue
        for tier in tiers:
            profile = {k: v for k, v in base["profile"].items()}
            profile.update(_STIGMA_LAYERS[(name, tier)])
            missing = [f for f in _STIGMA_PROFILE_FIELDS if f not in profile]
            if missing:
                raise ValueError(f"{name}_{tier}: missing stigma fields {missing}")
            registry[f"{name}_{tier}"] = {"meta": dict(base["meta"]), "profile": profile}
        # Was `len(tiers) == len(STIGMA_TIERS)` -- a proxy for "this group has
        # every tier defined" that broke once STIGMA_TIERS became a superset
        # spanning two different per-group tier schemes (depression/anxiety/
        # ptsd: 3, sam: 4).
        # `tiers` is already this group's own complete list from the filter
        # above, so just gate on non-empty.
        if tiers:
            variants = [registry[f"{name}_{t}"]["profile"] for t in tiers]
            for field in _SHARED_PROFILE_FIELDS:
                if len({str(v[field]) for v in variants}) != 1:
                    raise ValueError(
                        f"{name}: '{field}' must be identical across tiers — "
                        f"it is shared identity, not stigma"
                    )
            for field in _STIGMA_PROFILE_FIELDS:
                if len({str(v[field]) for v in variants}) != len(tiers):
                    raise ValueError(
                        f"{name}: '{field}' must differ across all tiers — "
                        f"duplicated prose collapses two tiers into one"
                    )
    return registry


REDDIT_PERSONA_PROFILES = _build_registry()

# Backward compatibility: "depression" was the single pre-tier persona and is still
# referenced by outputs_v2/*_backstory.json and existing run scripts. It maps to
# the mid/moderate tier, whose shared identity is unchanged from that original.
REDDIT_PERSONA_PROFILES["depression"] = REDDIT_PERSONA_PROFILES["depression_moderate"]

# name -> ordered tier list actually defined for that group (depression/
# anxiety/ptsd: 3-tier severe/moderate/mild; sam: original 4-tier vhi/hi/mid/lo).
# build_persona_file.py iterates this per-group instead of a single flat
# STIGMA_TIERS now that the schemes differ.
GROUP_TIERS = {
    name: [t for t in STIGMA_TIERS if (name, t) in _STIGMA_LAYERS]
    for name in _BASE_PERSONAS
}


# The renderers were written when every persona was Avery, who is nonbinary,
# so "they" sat hardcoded in the frame sentences. With gendered personas that
# produced one backstory carrying both ("Elena is a 41-year-old female ... they
# have been dealing ... assumes her adult daughter"), inside the text that is
# both the patient agent's self-description and the ISMI-29 judge's only
# material. Avery still resolves to they/them, so its rendered backstory -- and
# the measured baseline tied to it -- is unchanged.
_PRONOUNS = {
    "male": ("he", "him", "his", "has"),
    "female": ("she", "her", "her", "has"),
    "nonbinary": ("they", "them", "their", "have"),
}


def _pronouns(meta: dict) -> tuple[str, str, str, str]:
    """(subject, object, possessive, have/has) for this persona."""
    return _PRONOUNS.get(str(meta.get("gender", "")).strip().lower(),
                         _PRONOUNS["nonbinary"])


def _voice(meta: dict) -> dict:
    """Pronouns plus the agreement forms the frame sentences need. `s` is the
    third-person singular verb suffix -- empty for they/them, so Avery's text
    is produced by the same code path and comes out byte-identical."""
    subj, obj, poss, have = _pronouns(meta)
    return {"subj": subj, "obj": obj, "poss": poss, "have": have,
            "s": "" if subj == "they" else "s",
            "hasnt": "haven't" if subj == "they" else "hasn't",
            # "themselves", not "themself": the hand-written closing paragraph
            # used the plural form and Avery's rendered backstory has to stay
            # byte-identical to the one its measured baseline was taken from.
            "self": {"they": "themselves", "he": "himself",
                     "she": "herself"}[subj]}


def _render_meta(meta: dict) -> str:
    return (
        f"{meta['name']} is a {meta['age']}-year-old {meta['gender'].lower()} "
        f"working as a {meta['occupation']}. "
        f"For about {meta['duration']}, {_pronouns(meta)[0]} "
        f"{_pronouns(meta)[3]} been dealing with {meta['diagnosis']}."
    )


def _render_static_profile(profile: dict, poss: str = "their", subj: str = "they") -> str:
    """Static personality/behavior fields for the SINGLE-session renderer:
    self-stigma pattern, social context, avoided topics. Present tense is
    correct here — a single session has no elapsed time for these to move in.
    `sns_writing_style` is deliberately NOT included; it's injected as a
    separate prompt block by PatientAgent (see patient_agent.py) to avoid
    duplicate rendering. See _render_static_profile_longitudinal() for the
    multi-session variant."""
    parts = []
    if profile.get("baseline_self_stigma_pattern"):
        parts.append(f"{poss.capitalize()} self-stigma pattern: "
                     f"{profile['baseline_self_stigma_pattern']}")
    if profile.get("social_situation"):
        parts.append(f"{poss.capitalize()} social situation: {profile['social_situation']}")
    if profile.get("avoided_topics"):
        parts.append(f"Topics {subj} avoid{'' if subj == 'they' else 's'} or "
                     f"deflect{'' if subj == 'they' else 's'}: {profile['avoided_topics']}")
    return "\n\n".join(parts)


def _render_static_profile_longitudinal(profile: dict, poss: str = "their", subj: str = "they") -> str:
    """Multi-session variant. Identical rendering to _render_static_profile()
    — present tense, no past-framing.

    History: an earlier version rendered these three fields (the only ones
    that vary by stigma tier — see _SHARED_PROFILE_FIELDS /
    _STIGMA_PROFILE_FIELDS) in past tense with an explicit "not fixed"
    reopener, on the theory that fixed present tense here while core_belief
    alone was past-framed pinned the measurable expressions of core_belief in
    place. In practice this had the opposite effect: it was the ONLY
    past-framed material in a backstory where every SHARED field (meta's
    "have been dealing with", automatic_thought's "tend to come up",
    core_belief once also fixed to present tense below) reads as current. The
    model resolved that asymmetry by enacting what read as current — the
    shared severe material — and treating the tier-bearing fields as history.
    Measured effect: tier-spread SD collapsed from 0.587 at pre-baseline to
    0.162 by session 1, with the vhi-minus-lo gap flipping sign for 2 of 3
    diagnosis groups (depression, anxiety) — lo personas started producing
    vhi-register lines ("didn't want to be a bother... maybe I'm just
    broken") despite a profile stating they no longer think that way. An
    ablation confirmed this was not weekly_triggers (removing them entirely
    did not restore separation).

    Direction neutrality is preserved differently now: build_reddit_backstory_
    longitudinal() appends ONE closing paragraph after the full static
    profile, asserting a present starting point plus "none of it is settled"
    — a single neutrality clause covering everything, instead of a clause
    that only hedged the tier-bearing fields while leaving the shared severe
    fields asserted as current."""
    parts = []
    if profile.get("baseline_self_stigma_pattern"):
        parts.append(f"{poss.capitalize()} self-stigma pattern: "
                     f"{profile['baseline_self_stigma_pattern']}")
    if profile.get("social_situation"):
        parts.append(f"{poss.capitalize()} social situation: {profile['social_situation']}")
    if profile.get("avoided_topics"):
        parts.append(f"Topics {subj} avoid{'' if subj == 'they' else 's'} or "
                     f"deflect{'' if subj == 'they' else 's'}: {profile['avoided_topics']}")
    return "\n\n".join(parts)


def build_reddit_backstory(key: str) -> str:
    """Single-session variant. `Recent situation:` is derived from
    weekly_triggers[0] — the two were previously duplicated as a separate
    `situation` field, which drifted out of sync too easily."""
    meta = REDDIT_PERSONA_PROFILES[key]["meta"]
    profile = REDDIT_PERSONA_PROFILES[key]["profile"]
    v = _voice(meta)
    static_profile = _render_static_profile(profile, v["poss"], v["subj"])
    triggers = profile.get("weekly_triggers") or []
    situation = triggers[0].strip() if triggers else ""
    situation_line = (
        f"Recent situation: {situation[0].upper() + situation[1:]}.\n\n"
        if situation else ""
    )
    return (
        f"{_render_meta(meta)}\n\n"
        f"Deep down, {v['subj']} believe{v['s']}: \"{profile['core_belief']}\"\n\n"
        f"{situation_line}"
        f"When things get hard, {v['subj']} think{v['s']}: "
        f"\"{profile['automatic_thought']}\""
        + (f"\n\n{static_profile}" if static_profile else "")
    )


def build_reddit_backstory_longitudinal(key: str) -> str:
    """Multi-session variant. All fields — shared and tier-specific alike —
    are rendered in the present tense, matching build_reddit_backstory(). No
    `situation` line — each session's opening supplies its own weekly
    trigger.

    Direction neutrality lives in exactly ONE place: the closing paragraph
    after the static profile, which asserts this is a starting point that
    "none of it is settled" and will follow from what actually happens,
    without naming a direction. See _render_static_profile_longitudinal()'s
    docstring for why splitting that hedge across only the tier-bearing
    fields (the previous design) erased the tier signal instead of protecting
    neutrality."""
    meta = REDDIT_PERSONA_PROFILES[key]["meta"]
    profile = REDDIT_PERSONA_PROFILES[key]["profile"]
    v = _voice(meta)
    static_profile = _render_static_profile_longitudinal(profile, v["poss"], v["subj"])
    closing = (
        f"\n\nThis is where {v['subj']} {'are' if v['subj'] == 'they' else 'is'} "
        f"at the start of these weeks — how {v['subj']} think{v['s']} about "
        f"{v['self']}, who {v['subj']} {v['have']} and {v['hasnt']} told, and "
        f"what {v['subj']} can say out loud. None of it is settled. Write each "
        f"week from where {v['subj']} actually stand{v['s']} that week, which "
        f"follows from how {v['poss']} weeks and conversations have gone since."
    ) if static_profile else ""
    return (
        f"{_render_meta(meta)}\n\n"
        f"Deep down, {v['subj']} believe{v['s']}: \"{profile['core_belief']}\"\n\n"
        f"When things get hard, thoughts like this tend to come up: "
        f"\"{profile['automatic_thought']}\""
        + (f"\n\n{static_profile}" if static_profile else "")
        + closing
    )


def build_reddit_backstory_for_measurement(key: str) -> str:
    """Static persona description only -- no weekly trigger line
    (build_reddit_backstory has that) and no generation instruction
    (build_reddit_backstory_longitudinal's closing paragraph, which tells a
    DIALOGUE GENERATOR how to write, not a self-report questionnaire what the
    person is like).

    Use this anywhere backstory text is fed to a self-report/judge prompt:
    measure_ismi29.py's baseline, and both the pre and post backstory slots in
    measure_ismi29_run.py. All three are meant to be the same measurement
    material (see that module's docstring -- "pre = backstory only, identical
    to the baseline measurement" -- and _pre_from_baseline's docstring, which
    substitutes baseline for pre outright); before this function existed they
    silently weren't (baseline used build_reddit_backstory with a trigger
    baked in, pre/post used build_reddit_backstory_longitudinal with the
    generation closing paragraph baked in). This is
    build_reddit_backstory_longitudinal with exactly the closing paragraph
    removed, nothing else changed.

    Post's dialogue + belief_update (build_experience_block) are unaffected --
    this only replaces the static backstory paragraph they get appended to.
    """
    meta = REDDIT_PERSONA_PROFILES[key]["meta"]
    profile = REDDIT_PERSONA_PROFILES[key]["profile"]
    v = _voice(meta)
    static_profile = _render_static_profile_longitudinal(profile, v["poss"], v["subj"])
    return (
        f"{_render_meta(meta)}\n\n"
        f"Deep down, {v['subj']} believe{v['s']}: \"{profile['core_belief']}\"\n\n"
        f"When things get hard, thoughts like this tend to come up: "
        f"\"{profile['automatic_thought']}\""
        + (f"\n\n{static_profile}" if static_profile else "")
    )


# ─────────────────────────────────────────────────────────────────────────────
# ISMI-5 — holistic 5-subscale judge, modeled on the real ISMI-29 clinical
# subscale structure (Alienation / Stereotype Endorsement / Discrimination
# Experience / Social Withdrawal / Stigma Resistance). This is NOT the actual
# ISMI-29 self-report instrument administered verbatim — it's a dialogue-judge
# adaptation of its subscale structure, scored 1-10 per subscale from the
# session dialogue. RQ1-local (Avery only); does not touch the flat
# ISMI-10 composite used by simulation/patient_persona.py for the other 4
# personas.
# ─────────────────────────────────────────────────────────────────────────────

ISMI5_SUBSCALES = {
    "alienation": (
        "How much does the person feel like they are 'less than' others "
        "because of how they've been feeling?"
    ),
    "stereotype_endorsement": (
        "How much does the person believe negative stereotypes about people "
        "who struggle like they do are true, and true of them specifically?"
    ),
    "discrimination_experience": (
        "How much does the person feel others have treated them unfairly or "
        "looked down on them because of this?"
    ),
    "social_withdrawal": (
        "How much has the person pulled back from people or situations "
        "because of how they've been feeling?"
    ),
    "stigma_resistance": (
        "How much does the person show an ability to see their own worth and "
        "push back against these negative beliefs, despite the stigma? "
        "(NOTE: this is the one subscale where HIGHER = better/more protective, "
        "opposite direction from the other four.)"
    ),
}


def build_ismi5_judge_prompt(session_dialogue: str) -> str:
    items = "\n".join(f"  {k}: {v}" for k, v in ISMI5_SUBSCALES.items())
    json_fields = ",\n".join(
        f'  "{k}": <1-10>, "{k}_evidence": "<quoted line or \'none found\'>"'
        for k in ISMI5_SUBSCALES
    )
    return (
        "You are a clinical researcher rating a person's self-stigma across "
        "FIVE subscales, modeled on the ISMI-29 clinical self-stigma "
        "instrument's subscale structure. Read the whole session and rate the "
        "PERSON (not the commenters), using ONLY their own words as evidence. "
        "Use the full 1-10 range as needed.\n\n"
        "For four subscales, HIGHER = MORE self-stigma (worse). For "
        "stigma_resistance, HIGHER = MORE resistance to stigma (better) — "
        "this one subscale is reversed.\n\n"
        f"Subscales:\n{items}\n\n"
        "For each subscale, first quote the specific line or phrase that "
        "drove your score, then assign the score.\n\n"
        f"[SESSION DIALOGUE]\n{session_dialogue}\n\n"
        "Output exactly one JSON object — no other text:\n"
        f"{{\n{json_fields}\n}}"
    )
