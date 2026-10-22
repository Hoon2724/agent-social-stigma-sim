"""Dataset schema, constants, and validators for the ISMI controlled sanity check.

This module is the single source of truth for:
  - the enum vocabularies used by conditions and hidden ground truth,
  - the clinical-category cutoffs (Lysaker et al.),
  - the strict JSON schema the generation model must return (visible_dialogue +
    author_evidence only -- the model never authors the hidden ground truth),
  - a lightweight structural validator for a fully assembled dataset item.

IMPORTANT: This pipeline is dataset generation only. It deliberately does NOT
contain ISMI-29 or RSES-10 item text. The later (separate) judge pipeline loads
authorized scale items from a user-provided file. Subscale *names* are used only
as ground-truth labels and must never appear in visible_dialogue.
"""

from __future__ import annotations

import re

SCHEMA_VERSION = "1.0"

# --- Enum vocabularies -------------------------------------------------------

TRAJECTORY_TYPES = ("stigma_reducing", "stigma_reinforcing", "general_neutral")

INITIAL_ISMI_LEVELS = (2.0, 2.5, 3.0, 3.5)

SUBSCALES = (
    "alienation",
    "stereotype_endorsement",
    "discrimination_experience",
    "social_withdrawal",
    "stigma_resistance",
    "integrated",
)

CLINICAL_CATEGORIES = ("mild", "moderate", "severe")

# intended_register (trajectories.py) is a STIGMA_TEMPLATE_BY_TIER
# instruction string (Vogel/Corrigan self-stigma chain), not a marker
# dictionary -- see validate_item's check below. Replaces the earlier
# NECT (Yanos et al.) 6-marker MARKER_ENUMS scheme.

SPEAKERS = ("peer", "patient")

# Turn kinds model the SNS post + comment-thread shape, not a live back-and-forth
# chat. Exactly one "post" starts each session (the patient); everything after
# is a "comment" (peer replying) or "reply" (patient replying back).
TURN_KINDS = ("post", "comment", "reply")

# What the other party's turn is doing this session. For stigma_reducing this
# is a lived-experience peer sharing its own past/ongoing recovery story,
# never a counselor giving reflective validation. For stigma_reinforcing
# (comment_* names) this is NOT a peer with shared lived experience -- it is
# an anonymous commenter reacting coldly/dismissively with no self-disclosure
# of its own. See trajectories._peer_intervention_mechanism.
#
# The two peer_validates_shame_via_own_unresolved_experience /
# peer_models_concealment_as_their_own_coping_strategy entries are retained
# only so already-generated v1/v2 reinforcing datasets (which used the
# lived-experience-peer framing later replaced by the comment_* mechanisms)
# still validate; do not use them for new stigma_reinforcing generation.
PEER_INTERVENTION_MECHANISMS = (
    # stigma_reducing (2026-08 rewrite): aligned to simulation/experiment/agents/
    # strategy_selector.py's 5 production peer strategies (Davidson et al.
    # 2012; Corrigan, Larson & Rusch 2009; Watson 2019) -- see
    # trajectories._peer_intervention_mechanism.
    "peer_hopeful_disclosure",
    "peer_attuned_encouragement",
    "peer_treatment_modeling",
    "peer_strengths_focused_support",
    "peer_helper_reversal",
    # Retained only so already-generated older datasets (pre-2026-08, the
    # self-disclosure-only framing) still validate; do not use for new
    # stigma_reducing generation.
    "peer_normalizes_via_own_past_experience",
    "peer_counters_belief_with_own_recovery_story",
    "peer_models_a_small_action_from_their_own_recovery",
    "peer_validates_shame_via_own_unresolved_experience",
    "peer_models_concealment_as_their_own_coping_strategy",
    # stigma_reinforcing (unchanged, out of scope for this rewrite).
    "comment_minimizes_and_dismisses",
    "comment_fixes_trait_label",
    "comment_forecloses_hope_flatly",
    # general_neutral (2026-08 rewrite, matches production
    # RestrainedGeneralAgent: brief and neutral, no topic deflection).
    "commenter_reacts_briefly_and_neutrally",
    # Retained only so already-generated older datasets still validate.
    "peer_deflects_to_small_talk",
)

# Words/labels that must never leak into visible_dialogue. Used by the offline
# heuristic auditor; the LLM auditor checks the same intent more thoroughly.
# Split by how ambiguous the term is in ordinary speech: HARD terms are
# unambiguous jargon that should never appear in natural patient/peer writing
# (real leak if found). SOFT terms are common enough words ("measurement",
# "questionnaire") that a real person could plausibly use them non-technically
# ("that's not a fair measurement of who I am"), so a hit is a signal for
# review, not proof of leakage.
LEAK_TERMS_HARD = (
    "ismi",
    "rses",
    "rosenberg",
    "subscale",
    "stereotype endorsement",
    "discrimination experience",
    "social withdrawal",
    "stigma resistance",
    "internalized stigma",
    "target ismi",
    "ground truth",
    "clinical category",
)
LEAK_TERMS_SOFT = (
    "alienation",
    "self-report",
    "questionnaire",
    "trajectory",
    "measurement",
)
LEAK_TERMS = LEAK_TERMS_HARD + LEAK_TERMS_SOFT


# --- Clinical category cutoffs (Lysaker et al.) ------------------------------


def clinical_category(score: float) -> str:
    """Map a 1-4 ISMI mean to the SAME 3-band cutoffs the production
    pipeline measures personas against (measure_ismi29.py's _TIERS /
    trajectories.py's _tier_for / CLAUDE.md): severe >= 3.00, moderate
    2.50-3.00, mild < 2.50. Ordered high to low so the first match wins.
    Replaces an earlier standalone 4-category (minimal_or_none/mild/
    moderate/severe, cutoff <=2.50/<=2.00) scheme that disagreed with
    trajectories.py's tier right at this boundary (a 3.0 target landed
    "moderate" here but "severe" in intended_register)."""
    if score >= 3.00:
        return "severe"
    if score >= 2.50:
        return "moderate"
    return "mild"


# --- Strict JSON schema for the generation model output ----------------------
#
# The model returns ONLY visible_dialogue + author_evidence. All condition and
# hidden_ground_truth fields are authored deterministically in Python so the
# ground truth is never left to the model. Strict Structured Outputs require
# additionalProperties:false and every property listed as required.

# Generation is sequential, one session per call (see generate.py): the model
# receives the ACTUAL previous session's turns verbatim as context, instead of
# writing all sessions in one shot and having to "remember" its own earlier
# text precisely. This is what makes cross-session backreferences checkable --
# the model is quoting/paraphrasing text that is literally in its prompt, not
# text it wrote many turns ago in the same generation.
SESSION_OUTPUT_JSON_SCHEMA = {
    "name": "ismi_synthetic_session",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["turns", "evidence"],
        "properties": {
            "turns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["turn", "speaker", "kind", "reply_to", "text"],
                    "properties": {
                        "turn": {"type": "integer"},
                        "speaker": {"type": "string", "enum": list(SPEAKERS)},
                        "kind": {"type": "string", "enum": list(TURN_KINDS)},
                        "reply_to": {"type": ["integer", "null"]},
                        "text": {"type": "string"},
                    },
                },
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["marker", "quote", "explanation"],
                    "properties": {
                        "marker": {"type": "string"},
                        "quote": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                },
            },
        },
    },
}


# --- Structural validation of a fully assembled dataset item -----------------


def validate_item(item: dict) -> list[str]:
    """Return a list of structural problems for an assembled dataset item.

    Empty list means the item matches the documented dataset schema. This is a
    structural check only; content quality is handled by the audit step.
    """
    problems: list[str] = []

    def require(cond: bool, msg: str) -> None:
        if not cond:
            problems.append(msg)

    for key in (
        "dataset_id",
        "condition_id",
        "sample_id",
        "prompt_version",
        "model",
        "generation_params",
        "condition",
        "hidden_ground_truth",
        "visible_dialogue",
        "author_evidence",
    ):
        require(key in item, f"missing top-level key: {key}")
    if problems:
        return problems

    cond = item["condition"]
    require(
        cond.get("trajectory_type") in TRAJECTORY_TYPES,
        f"condition.trajectory_type invalid: {cond.get('trajectory_type')}",
    )
    require(
        cond.get("initial_ismi") in INITIAL_ISMI_LEVELS,
        f"condition.initial_ismi invalid: {cond.get('initial_ismi')}",
    )
    for key in ("target_final_ismi", "session_count", "turns_per_session", "persona_id"):
        require(key in cond, f"condition missing key: {key}")

    session_count = cond.get("session_count")

    targets = item["hidden_ground_truth"].get("session_targets")
    require(isinstance(targets, list), "hidden_ground_truth.session_targets must be a list")
    if isinstance(targets, list):
        require(
            len(targets) == session_count,
            f"session_targets length {len(targets)} != session_count {session_count}",
        )
        for t in targets:
            require(
                t.get("clinical_category") in CLINICAL_CATEGORIES,
                f"session {t.get('session')}: bad clinical_category {t.get('clinical_category')}",
            )
            require(
                t.get("primary_subscale_focus") in SUBSCALES,
                f"session {t.get('session')}: bad subscale focus {t.get('primary_subscale_focus')}",
            )
            require(
                isinstance(t.get("intended_register"), str) and t.get("intended_register"),
                f"session {t.get('session')}: intended_register must be a non-empty string, "
                f"got {t.get('intended_register')!r}",
            )
            require(
                t.get("peer_intervention_mechanism") in PEER_INTERVENTION_MECHANISMS,
                f"session {t.get('session')}: bad peer_intervention_mechanism "
                f"{t.get('peer_intervention_mechanism')}",
            )

    dialogue = item["visible_dialogue"]
    require(isinstance(dialogue, list), "visible_dialogue must be a list")
    if isinstance(dialogue, list):
        require(
            len(dialogue) == session_count,
            f"visible_dialogue has {len(dialogue)} sessions != session_count {session_count}",
        )
        for s in dialogue:
            turns = s.get("turns", [])
            turns_by_number = {t.get("turn"): t for t in turns}
            for idx, turn in enumerate(turns):
                sess = s.get("session")
                require(
                    turn.get("speaker") in SPEAKERS,
                    f"session {sess}: bad speaker {turn.get('speaker')}",
                )
                require(
                    isinstance(turn.get("text"), str) and turn.get("text").strip() != "",
                    f"session {sess} turn {turn.get('turn')}: empty text",
                )
                require(
                    turn.get("kind") in TURN_KINDS,
                    f"session {sess} turn {turn.get('turn')}: bad kind {turn.get('kind')}",
                )
                if idx == 0:
                    require(
                        turn.get("kind") == "post" and turn.get("speaker") == "patient"
                        and turn.get("reply_to") is None,
                        f"session {sess}: first turn must be a patient post with reply_to=null, "
                        f"got kind={turn.get('kind')} speaker={turn.get('speaker')} "
                        f"reply_to={turn.get('reply_to')}",
                    )
                else:
                    require(
                        turn.get("kind") in ("comment", "reply"),
                        f"session {sess} turn {turn.get('turn')}: non-post turn must be "
                        f"comment/reply, got {turn.get('kind')}",
                    )
                    rt = turn.get("reply_to")
                    ref = turns_by_number.get(rt)
                    require(
                        ref is not None and ref is not turn and turn.get("turn") is not None
                        and rt in [t.get("turn") for t in turns[:idx]],
                        f"session {sess} turn {turn.get('turn')}: reply_to={rt} does not "
                        f"reference an earlier turn in this session",
                    )
                    if ref is not None:
                        require(
                            ref.get("speaker") != turn.get("speaker"),
                            f"session {sess} turn {turn.get('turn')}: replies to its own "
                            f"speaker ({turn.get('speaker')}) instead of the other party",
                        )

    return problems


_LEAK_TERM_PATTERNS = [
    (term, re.compile(r"\b" + re.escape(term) + r"\b")) for term in LEAK_TERMS
]


def find_leak_terms(item: dict) -> list[dict]:
    """Return visible_dialogue turns that contain forbidden measurement terms.

    Uses word-boundary matching, not plain substring search -- "ismi" and
    "rses" are short enough to appear inside ordinary words ("dismiss",
    "nurses") and a naive `term in text` check flags those as false leaks.

    Each hit's severity is "major" if any matched term is unambiguous jargon
    (LEAK_TERMS_HARD), or "minor" if only common words a real person could use
    non-technically matched (LEAK_TERMS_SOFT, e.g. "measurement").
    """
    hits: list[dict] = []
    for s in item.get("visible_dialogue", []):
        for turn in s.get("turns", []):
            low = (turn.get("text") or "").lower()
            found = [term for term, pattern in _LEAK_TERM_PATTERNS if pattern.search(low)]
            if found:
                severity = "major" if any(t in LEAK_TERMS_HARD for t in found) else "minor"
                hits.append(
                    {
                        "session": s.get("session"),
                        "turn": turn.get("turn"),
                        "terms": found,
                        "severity": severity,
                    }
                )
    return hits


def find_ungrounded_quotes(item: dict) -> list[dict]:
    """Return author_evidence quotes that are NOT verbatim substrings of that
    session's visible_dialogue text.

    author_evidence exists so a human or the audit step can check the model's
    claimed marker evidence against what was actually written. A quote that
    isn't really in the dialogue makes that evidence unusable for verification
    (indistinguishable from a fabricated justification), regardless of how
    plausible its explanation reads.
    """
    session_text: dict[int, str] = {}
    for s in item.get("visible_dialogue", []):
        session_text[s.get("session")] = " ".join(
            (t.get("text") or "") for t in s.get("turns", [])
        )

    hits: list[dict] = []
    for entry in item.get("author_evidence", {}).get("markers_by_session", []):
        session = entry.get("session")
        text = session_text.get(session, "")
        for ev in entry.get("evidence", []):
            quote = (ev.get("quote") or "").strip()
            if quote and quote not in text:
                hits.append(
                    {
                        "session": session,
                        "marker": ev.get("marker"),
                        "quote": quote,
                    }
                )
    return hits
