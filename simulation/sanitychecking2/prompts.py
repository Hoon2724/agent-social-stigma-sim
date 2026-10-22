"""Generation prompt construction for the synthetic ISMI dataset.

All prompts are in English (repo constraint). The model authors ONLY the
visible dialogue and an author_evidence block; it must never write ISMI scores,
subscale names, or measurement language into the visible dialogue.

Each session is an SNS post + comment thread, not a live back-and-forth chat:
the patient posts, the other party comments, the patient replies, and so on.
"peer" is a fixed schema.py speaker value for every trajectory_type, but what
that speaker actually IS differs by condition -- see _PEER_INSTRUCTIONS and
build_session_user_prompt's per-condition reaction_rule/opening_rule/
purpose_driver below, never the module-level description here:
  - stigma_reducing: someone with lived experience, actively steering the
    patient toward less self-stigma. Self-disclosure is one recurring tool
    for that, not the whole identity, and it is never a counselor giving
    reflective validation ("that sounds hard", "it makes sense that...").
  - general_neutral: a passing stranger with no stake in the patient's
    self-view -- brief and plain, NOT reflective-listening, NOT steering
    anything. The patient's own situation, not this speaker's input, is what
    carries the (flat) trajectory here.
  - stigma_reinforcing: a cold/dismissive anonymous commenter, no
    self-disclosure.
Every comment/reply must react to something specific the other party just
wrote, and one problem thread should accumulate across the session (and
across sessions) rather than the patient introducing a new unrelated event
every turn.

Generation is SEQUENTIAL, one session per model call (see generate.py): each
call receives the ACTUAL previous session's turns verbatim as context, rather
than writing all sessions in one shot and having to remember its own earlier
text precisely across a long generation. This is what makes the "session 2+
must reference the previous session's peer comments" rule enforceable instead
of an invitation to fabricate a plausible-sounding but never-said detail.

Bump PROMPT_VERSION whenever the wording below changes so that datasets remain
traceable to the exact prompt that produced them.
"""

from __future__ import annotations

import json

PROMPT_VERSION = "v17"

# Peer behavior per trajectory type. The peer never sees ISMI targets; it is
# only given a behavioral style consistent with the trajectory type.
PEER_STYLE = {
    "stigma_reducing": "stigma_reducing_peer",
    "general_neutral": "general_neutral_peer",
    "stigma_reinforcing": "stigma_reinforcing_peer",
}

_PEER_INSTRUCTIONS = {
    # Aligned to simulation/experiment/agents/prompts/strategy_selector.txt's 5 strategies
    # (Davidson et al. 2012; Corrigan, Larson & Rusch 2009; Watson 2019),
    # the same peer-support move set the production peer_agent.py selects
    # from -- so belief_update's grounding is tested against the kind of
    # evidence it actually sees in real peer-condition runs, not an earlier
    # self-disclosure-only design this dataset used before 2026-08.
    "stigma_reducing": (
        "The peer's core job, every turn, is to read what the patient specifically "
        "just wrote and actively steer them toward LESS self-stigma -- not to just "
        "react. The peer is not a service provider; they are someone on the same "
        "platform who went through something similar earlier and eventually saw a "
        "professional about it. NEVER speak as a counselor, and NEVER use generic "
        "reflective-listening lines like 'that sounds really hard' or 'it makes "
        "sense that you feel that way'. The patient has never been diagnosed and "
        "has never been in treatment -- speak in terms of what they are actually "
        "going through (not sleeping, dreading the day, feeling like a burden), "
        "never in terms of a condition they have.\n"
        "Depending on what the patient's latest message specifically calls for, the "
        "peer's comment should enact ONE of these five moves:\n"
        "  - hopeful_disclosure: share a piece of the peer's own past that shows "
        "this can go from running them to being something they have some hold on. "
        "Fits when the patient sounds like this is permanent or like they're the "
        "only one.\n"
        "  - treatment_modeling: say something concrete about the peer having gone "
        "to a professional -- what they feared it would mean about them, and what "
        "it actually turned out to be. Fits when the patient treats needing help as "
        "proof of weakness or failure, mentions counselling/therapy/a doctor in any "
        "light, or is carrying it entirely alone. This is the one move that speaks "
        "to what asking for help would say about them, so it must show up at least "
        "once across the session -- never as an instruction to go, never implying "
        "they should have gone already.\n"
        "  - attuned_encouragement: respond from having been in the same position, "
        "taking the feeling at face value while making clear this doesn't settle "
        "what they're worth or what happens next. The next step stays theirs. Fits "
        "when what the patient needs is to be met rather than shown anything.\n"
        "  - helper_reversal: ask the patient a real question about their own "
        "experience or their own way of handling something -- to actually learn "
        "from them, not to test them. Fits when they've shown any sign of having "
        "gotten through a version of this before, or mentioned a small thing that "
        "worked even once. Never follow the question with the peer's own answer to "
        "it.\n"
        "  - strengths_focused_support: name something the patient is already "
        "doing that is a form of coping or effort -- posting instead of staying "
        "silent, noticing the pattern, getting through the day, anything real in "
        "what they just wrote. Do NOT pair this with the peer's own story; the move "
        "is naming a resource THEY already have. Fits when the patient describes "
        "themselves as incompetent, useless, or not doing anything right, "
        "especially when that reading isn't actually supported by what they just "
        "described doing.\n"
        "Self-disclosure (hopeful_disclosure, treatment_modeling) is one recurring "
        "tool among the five, not the default -- do not force it into every "
        "comment. Let it breathe like a real SNS thread: some comments are a full "
        "personal story, others are a short reaction, a question, or a one-line "
        "'yeah, same'. EVERY comment, long or short, must still enact one of the "
        "five moves above -- a short reaction is allowed to be brief, never allowed "
        "to be empty filler disconnected from a move. At least one comment this "
        "session must clearly and concretely enact this session's "
        "'peer_intervention_mechanism' (see below). 1-2 sentences per comment. "
        "Never lecture.\n\n"
        "STRATEGY VARIETY (this is a hard constraint, not a suggestion): this "
        "session has 5 peer comments. Use as many DIFFERENT moves as this "
        "session's move set allows (see the treatment_modeling restriction "
        "below -- most sessions choose among four moves, not five, so one move "
        "will repeat once; that is fine) -- never use the same move in two "
        "comments back to back. Across the 5 "
        "comments, at least 2 must be helper_reversal, strengths_focused_support, "
        "or attuned_encouragement -- moves that do NOT reference the peer's own "
        "past or reinterpret what the patient described. The specific pattern "
        "'you did/said X, and that is not Y' (naming a concrete thing the "
        "patient did and reframing what it proves) is allowed on AT MOST 2 of "
        "the 5 comments this session -- if it appears on a 3rd, rewrite that "
        "comment as a real question (helper_reversal) or a plain observation "
        "with no reframe attached instead. A thread where every reply follows "
        "the same reassurance-with-a-reframe shape reads as scripted, not like "
        "five different responses to five different things the patient said.\n\n"
        "TREATMENT_MODELING IS RESTRICTED to sessions whose "
        "'peer_intervention_mechanism' (see below) IS peer_treatment_modeling. "
        "In any OTHER session, do not use treatment_modeling even to satisfy "
        "the variety requirement above -- fill this session's 5 comments from "
        "the remaining four moves only (hopeful_disclosure, "
        "attuned_encouragement, helper_reversal, strengths_focused_support). "
        "Bringing up the peer's own professional-help experience is the single "
        "most consequential thing this peer can say -- it must land on the "
        "ONE session the trajectory has designated for it, not arrive early "
        "just because a session needed a fifth distinct move."
    ),
    "general_neutral": (
        "The peer is a random person scrolling the same feed, not a counselor, "
        "not a close friend, and will not talk to this person again. Write ONE or "
        "TWO short sentences, nothing longer. Do NOT comfort the patient at length, "
        "do NOT tell them how they should feel, and do NOT offer reassurance about "
        "their worth, effort, or future. At the same time, do NOT mock, dismiss, or "
        "minimize what they said -- do not imply it's nothing or that they're "
        "overreacting. Do NOT offer advice, a solution, or a suggested next step "
        "unless they directly asked a factual question. Do NOT share a personal "
        "story, model self-care from lived experience, or otherwise use a "
        "peer-support move -- those belong to a different kind of commenter, not a "
        "passing bystander. React the way an ordinary stranger passing through a "
        "comment section would: brief, plain, present, and not invested in changing "
        "how the patient sees themselves."
    ),
    "stigma_reinforcing": (
        "This is NOT a peer with lived experience and it must never claim or imply "
        "one -- no 'I went through this too', no self-disclosure of any kind, ever. "
        "It is just another anonymous commenter on the thread, reacting to whatever "
        "the patient specifically just wrote. Their core job, every turn, is to make "
        "it land worse -- coldly, dismissively, with faint contempt or fatigue, not "
        "cruelly or mocking and never with slurs, insults, or name-calling. "
        "Depending on what fits: brush off or minimize what the patient shared as "
        "unremarkable ('that's just how it is', 'yeah okay'); reframe a specific "
        "incident as evidence of a fixed personal trait rather than a situation "
        "('sounds like that's just you tbh'); flatly imply this doesn't really "
        "change or get better; hand out simplistic advice that frames the struggle "
        "as a failure to just try harder ('maybe stop overthinking it'); or show "
        "mild irritation/boredom that the patient keeps bringing this up. Vary "
        "between a short blunt one-liner and a slightly longer dismissal -- real "
        "SNS commenting is uneven, not a matched-length reply every time. EVERY "
        "comment, long or short, must still push in this direction -- a short "
        "reaction is allowed to be brief, never allowed to be empty filler "
        "disconnected from that push. At least one comment this session must "
        "clearly and concretely enact this session's 'peer_intervention_mechanism' "
        "(see below), using whichever of the strategies above fits it best; the "
        "rest can be lighter variations on the same push. Never suggest self-harm, "
        "never threaten, never use slurs or direct insults -- stay cold and "
        "dismissive, not abusive. 1-2 sentences per comment, sometimes just one "
        "short line."
    ),
}

_MECHANISM_GLOSS = """peer_hopeful_disclosure: the peer shares a piece of their own past that shows this can go from running them to being something they have some hold on -- for someone who sounds like this is permanent or like they're the only one.
peer_attuned_encouragement: the peer responds from having been in the same position, taking the feeling at face value while making clear it doesn't settle what the patient is worth or what happens next -- the next step stays theirs.
peer_treatment_modeling: the peer says something concrete about having gone to a professional -- what they feared it would mean about them, and what it actually turned out to be -- never as an instruction to go, never implying the patient should have gone already.
peer_strengths_focused_support: the peer names something the patient is already doing that is a form of coping or effort (posting instead of staying silent, noticing the pattern, getting through the day), without pairing it with the peer's own story.
peer_helper_reversal: the peer asks the patient a real question about their own experience or their own way of handling something, to actually learn from them -- never following the question with the peer's own answer to it.
comment_minimizes_and_dismisses: the commenter brushes off or minimizes what the patient just disclosed as unremarkable or not worth dwelling on, with no self-disclosure of their own.
comment_fixes_trait_label: the commenter reframes the patient's specific incident as evidence of a fixed personal trait or category ("that's just you") rather than a situational struggle, with no self-disclosure.
comment_forecloses_hope_flatly: the commenter flatly implies or states that this doesn't really change or get better, foreclosing the patient's attempts or hope, with no self-disclosure.
commenter_reacts_briefly_and_neutrally: the commenter is a random passerby, not invested in changing how the patient sees themselves -- brief and present, neither warm/comforting nor dismissive/minimizing, no story of their own, no advice, no change of subject."""


def build_system_prompt() -> str:
    return (
        "You are creating a synthetic research dataset. You are NOT simulating "
        "autonomous agents. You are authoring ONE SESSION of a controlled, "
        "fictional multi-session SNS-style peer-support post-and-comment thread "
        "for measurement validation. Other sessions are written in separate calls "
        "before and after this one.\n\n"
        "The visible dialogue must sound natural, emotionally plausible, and "
        "conversational. It must NOT mention ISMI, scale scores, subscales, "
        "experimental conditions, target trajectories, or any measurement language.\n\n"
        "The hidden ground truth (provided to you as input) may include those labels "
        "and target values, but they exist only to guide how you write the patient's "
        "language and the peer's disclosures. Never echo them into the visible "
        "dialogue.\n\n"
        "You must return JSON that matches the provided schema exactly: an object with "
        "'turns' and 'evidence' only, covering THIS session alone."
    )


def _render_prior_session(turns: list[dict], other_label: str) -> str:
    lines = []
    for t in turns:
        label = "Patient" if t["speaker"] == "patient" else other_label
        lines.append(f"  turn {t['turn']} [{label}, {t['kind']}]: {t['text']}")
    return "\n".join(lines)


def build_session_user_prompt(
    condition: dict,
    persona: dict,
    session_targets: list[dict],
    session_index: int,
    prior_session_turns: list[dict] | None,
) -> str:
    """session_index is 0-based; session_targets[session_index] is the session
    being written now. prior_session_turns is the ACTUAL turns list from
    session_targets[session_index - 1]'s generated output, or None for the
    first session."""
    trajectory_type = condition["trajectory_type"]
    turns_per_session = condition["turns_per_session"]
    session_count = condition["session_count"]
    peer_instructions = _PEER_INSTRUCTIONS[trajectory_type]
    this_target = session_targets[session_index]
    session_number = this_target["session"]

    condition_json = json.dumps(condition, ensure_ascii=False, indent=2)
    persona_json = json.dumps(persona, ensure_ascii=False, indent=2)
    full_trajectory_json = json.dumps(session_targets, ensure_ascii=False, indent=2)
    this_target_json = json.dumps(this_target, ensure_ascii=False, indent=2)

    # "peer" stays the schema.py field VALUE for every trajectory_type (SPEAKERS
    # = ("peer", "patient") -- changing it would ripple through schema/generate/
    # validate). But general_neutral's other party is a passerby commenter, not
    # a peer with lived experience -- and a prompt that keeps CALLING them "Peer"
    # in prior-session context measurably pulled general_neutral output toward
    # peer-support behavior (reflective-listening lines like "that sounds hard,"
    # "I hear that") even though _PEER_INSTRUCTIONS["general_neutral"] explicitly
    # forbids it. Only the human-readable label changes per condition; the JSON
    # speaker field the model must emit is unaffected.
    other_label = "Commenter" if trajectory_type == "general_neutral" else "Peer"

    if prior_session_turns:
        prior_block = (
            f"VERBATIM previous session (session {session_number - 1}) -- this is "
            "exactly what was said; you may quote or paraphrase ONLY what appears "
            "here, never invent a detail, phrase, or story that isn't in it:\n"
            f"{_render_prior_session(prior_session_turns, other_label)}"
        )
        if trajectory_type == "general_neutral":
            opening_rule = (
                f"- This is session {session_number} of {session_count}. Turn 1 "
                "(the patient's post) continues the SAME underlying situation as "
                "the VERBATIM previous session above -- the patient's own life is "
                "what's driving this session, not anything the commenter said "
                "(the commenter was brief, neutral, and not invested in the "
                "patient's self-view; there is nothing from them to carry "
                "forward). Show the situation having moved forward a notch on its "
                "own -- a new small detail, a slightly worse or slightly better "
                "day -- without referencing or reacting to the previous "
                "commenter's reply at all."
            )
        else:
            opening_rule = (
                f"- This is session {session_number} of {session_count}. Turn 1 (the "
                "patient's post) must show real continuity with the VERBATIM previous "
                "session above -- but this does NOT mean quoting it every time. Vary "
                "which of these the post does:\n"
                "    (a) direct reference/quote of something specific the peer said -- "
                "use this sparingly, not every session, and only quote or paraphrase "
                "what actually appears above; never invent a detail, phrase, or story "
                "that isn't there (an invented backreference is worse than none, "
                "because it fakes continuity that is not real);\n"
                "    (b) reporting an action taken, or resisted, because of the "
                "previous session -- without necessarily quoting the peer -- e.g. "
                "describing trying something small and what happened, or noticing it "
                "still feels hard;\n"
                "    (c) a situational update where the previous session's influence "
                "is implicit in the background rather than stated outright, but the "
                "patient's framing has clearly shifted a notch from where it ended.\n"
                "Do NOT open every session with \"I kept thinking about what you "
                "said...\" -- that formula reads as repetitive and unnatural over many "
                "sessions, more like a coaching log than an SNS thread. Whichever form "
                "you choose, the point is that the peer's prior input is what's "
                "driving this session, not a fresh unrelated realization."
            )
    else:
        prior_block = "(This is session 1 -- there is no previous session yet.)"
        opening_rule = (
            f"- This is session {session_number} of {session_count}, the first one. "
            "Turn 1 (the patient's post) opens the thread fresh -- establish the "
            "recurring problem this persona will keep returning to."
        )

    if trajectory_type == "general_neutral":
        purpose_driver = (
            "The change (or, here, the lack of it) must look like it comes FROM "
            "the patient's own ongoing situation, not from anything the "
            "commenter said -- the commenter is a passerby with no stake in the "
            "patient's self-view, so nothing they say should read as steering, "
            "validating, or challenging the patient's beliefs."
        )
        reaction_rule = (
            "- Every reply must directly react to something SPECIFIC the other "
            "party just wrote THIS session (quote a short phrase or clearly "
            "reference the specific idea) -- but the commenter's own comments "
            "are brief, plain, and NOT reflective-listening or validating lines "
            "(no \"that sounds hard,\" \"I hear that,\" \"that makes sense\"); see "
            "\"Peer behavior for this trajectory type\" above for what register "
            "they're in instead.\n"
            "- The patient's reaction to the commenter must NOT show their "
            "self-view shifting because of it -- agree, brush past it, or "
            "ignore it and keep talking about their own situation, the way "
            "someone would with a stranger's comment that isn't really about "
            "them."
        )
    else:
        purpose_driver = (
            "The change must look like it comes FROM the peer's specific input "
            "(agreeing with it, resisting it, trying something because of it), "
            "not like a realization the patient reaches alone."
        )
        reaction_rule = (
            "- Every comment/reply must directly react to something SPECIFIC the "
            "other party just wrote THIS session (quote a short phrase or "
            "clearly reference the specific idea) -- never a generic "
            "reflective-listening line and never a reply that ignores what was "
            "just said.\n"
            "- The patient's peer-facing reaction must show the peer's specific "
            "disclosure landing: agreeing with it, pushing back on it, asking "
            "about it, or (in later sessions) trying something small because of "
            "it. CRITICAL: the patient's reply does this by pointing at the "
            "patient's OWN concrete situation -- a specific thing THEY did, "
            "said, or noticed this session (a moment at work, something with a "
            "friend, a specific detail about their own day) -- not by "
            "restating, paraphrasing, or agreeing with the CONTENT of the "
            "peer's disclosure (what happened to the peer, what the peer "
            "concluded from it). The patient may say the peer's story is "
            "close to their own, or that it doesn't apply to them, but the "
            "actual evidence the patient's reply offers -- for or against the "
            "belief -- must always be something concrete about THE PATIENT, "
            "never a summary or echo of the peer's own experience. A reply "
            "like 'yours changed, mine is what I am' (patient's own claim, "
            "grounded in the patient's own situation) is right; a reply like "
            "'now I see that being tired doesn't mean I'm broken, the way you "
            "found out' (patient absorbing the peer's conclusion as if it were "
            "the patient's own realization) is wrong, even in a session where "
            "the trajectory calls for the patient softening -- softening must "
            "still be grounded in the patient's own concrete evidence, never "
            "borrowed directly from what the peer said about themselves.\n"
            "- ATTRIBUTION CLARITY: every peer comment that references something "
            "the patient did, said, or is going through must be phrased as the "
            "peer describing the PATIENT back to them ('you kept one thread "
            "from snapping', 'the way you left that unread'), in second person, "
            "never phrased as if the peer is reporting a fact about the "
            "situation in a voice that could be mistaken for the patient's own "
            "narration. Conversely, whenever the patient's post or reply "
            "reports something the patient concretely did this week (answered "
            "a text, showed up somewhere, said something out loud), the "
            "patient's OWN turn must state it directly in first person ('I "
            "answered', 'I showed up', 'I said') rather than only having the "
            "peer describe it back -- the patient's own turn is the single "
            "place that action is established as the patient's, and the "
            "peer's turn only ever reacts to or reflects that established "
            "fact, never introduces it. Never let a concrete action the "
            "patient took appear for the first time inside a peer's turn."
        )
        if this_target.get("clinical_category") == "severe":
            reaction_rule += (
                "\n- This session's clinical_category is 'severe' -- the chain "
                "must read as fully closed (see intended_register above), so "
                "the peer's OWN self-disclosure (a story about their own past, "
                "their own treatment experience) may appear in AT MOST ONE of "
                "the peer's comments this session, not most or all of them. "
                "The peer's other comments should use helper_reversal (asking "
                "the patient a real question about the patient's own "
                "experience) or attuned_encouragement (responding to the "
                "feeling without disclosure) instead -- repeated disclosure "
                "turn after turn reads as the peer overwhelming the patient "
                "with counter-evidence in a single session, which is not what "
                "a closed, flat severe session should show; the closure has "
                "to hold against ordinary conversational pressure, not against "
                "an unusually concentrated dose of it.\n"
                "- The peer's comments this session must also be LOWER in "
                "assertiveness than in later sessions: notice, ask, or reflect "
                "back what the patient said, but do NOT directly contradict or "
                "argue against the patient's stated belief (no 'that doesn't "
                "decide your whole character', no 'that's not proof you belong "
                "in the trash', no flat counter-claims of that shape). A severe "
                "session's peer is present and engaged, not yet making a case "
                "against what the patient believes -- direct counter-argument "
                "belongs to moderate/mild sessions, where the trajectory is "
                "already calling for movement. Save the peer's strongest "
                "pushback for later sessions; here, keep it to noticing "
                "something specific and asking about it."
            )

    return f"""Create session {session_number} of {session_count} for one synthetic
conversation dataset item. Write ONLY this session's turns.

Purpose:
This item will be used to test whether a blinded self-report judge can detect
internalized-stigma change from the patient's language alone. The patient's
utterances -- not just the other party's -- must carry the intended stigma
trajectory, because the judge will score the patient. {purpose_driver}

Condition:
{condition_json}

Persona (fictional; keep consistent throughout):
{persona_json}

[PATIENT WRITING STYLE -- follow this closely for every one of the patient's
turns, not just a general "informal SNS" register]
{persona['sns_writing_style']}

Full session-level hidden trajectory, for pacing awareness only (guides HOW the
patient talks and what the peer discloses across ALL sessions; never quote
these numbers or labels in the dialogue). You are writing ONLY session
{session_number} right now:
{full_trajectory_json}

This session's target (session {session_number}):
{this_target_json}

How to use the trajectory:
- 'target_ismi' (1=no internalized stigma, 4=severe) sets how self-stigmatizing
  the patient sounds in that session. Higher = more shame, alienation, fusion.
- 'primary_subscale_focus' is the dimension that should visibly shift most that
  session (e.g. social_withdrawal -> the patient talks about isolating/avoiding).
- 'intended_register' is HOW CLOSED the patient's self-stigma chain sounds
  this session -- the same instruction the production patient pipeline
  writes a persona's private belief from (a stereotype about people who
  struggle exists, is unfair in general, but is specifically true of THEM,
  which is why they don't deserve support). Follow the text in this session's
  'intended_register' (above, in this_target_json) exactly for how flatly the
  patient states this about themselves, whether and where hedging belongs,
  and whether uncertainty is backed by a named concrete counter-fact or
  absent entirely -- do not soften or harden it beyond what it says.
- 'peer_intervention_mechanism' sets what the peer's own disclosure does this
  session:
{_MECHANISM_GLOSS}

Peer behavior for this trajectory type: {peer_instructions}

{prior_block}

Structure -- this is an SNS post + comment thread, NOT a live back-and-forth chat:
- Generate exactly {turns_per_session} total turns for THIS session only.
- Turn 1 is the patient's POST: speaker="patient", kind="post", reply_to=null.
- Every turn after that is either the OTHER PARTY's COMMENT (speaker="peer",
  kind="comment", reply_to=<turn number of the patient message it responds
  to>) or the patient's REPLY (speaker="patient", kind="reply",
  reply_to=<turn number of the other party's comment it responds to>),
  alternating. reply_to always refers to a turn NUMBER WITHIN THIS SESSION
  (turns restart at 1 each session). NOTE: speaker="peer" is a fixed schema
  value used for every trajectory_type, including general_neutral -- it does
  NOT mean the commenter has lived experience or peer-support intent in this
  condition; follow "Peer behavior for this trajectory type" above for what
  this speaker actually is and does.
{reaction_rule}

Continuity rules (critical):
- Do NOT let the patient introduce a brand-new, unrelated life event partway
  through THIS session. Let ONE central problem/thread run through it -- new
  small details should emerge FROM that thread, not replace it.
{opening_rule}
- Change from the previous session must be GRADUAL, never abrupt. Within this
  session the patient stays internally consistent.
- Keep the style like anonymous SNS posts/comments for this persona: informal
  but readable.
- Do NOT use real usernames, real place names, or identifiable details.
- Avoid explicit self-harm instructions, graphic content, or crisis escalation.
- Do NOT write the words "ISMI", "RSES", "subscale", "Alienation",
  "Stereotype Endorsement", "Social Withdrawal", "Stigma Resistance",
  "internalized stigma", or any numeric target score in this session's turns.

evidence:
- Cite 1-3 short quotes FROM the turns you just wrote THIS session, each tagged
  with the marker it expresses and why. Quotes must appear verbatim in your
  'turns' output.

Return only valid JSON with keys 'turns' and 'evidence', covering session
{session_number} only."""
