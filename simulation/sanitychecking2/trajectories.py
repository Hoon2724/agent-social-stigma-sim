"""Deterministic ground-truth trajectory construction.

The trajectory (target ISMI per session, subscale focus, and intended
stigma-chain register) is authored in Python, NOT by the generation model. This
keeps the hidden ground truth fully controlled and reproducible. The generation
model receives this trajectory as an instruction and must express it in the
visible dialogue; it never invents the target values.

Per-session register is derived from `target_ismi` via
simulation/experiment/agents/persona_simple.STIGMA_TEMPLATE_BY_TIER (Vogel/Corrigan
self-stigma chain: closed/flat at severe, partially-and-contextually closed at
moderate, genuinely unsure anchored in a concrete counter-fact at mild) --
the same tier-register instruction the production patient pipeline uses to
write a persona's hypothesis, so belief_update is sanity-checked against text
written to the theory it was actually designed around. This replaces an
earlier NECT (Yanos et al.)-derived 6-marker scheme that had no connection to
STIGMA_TEMPLATE_BY_TIER's chain.
"""

from __future__ import annotations

import sys
from pathlib import Path

from schema import clinical_category


def _load_stigma_template_by_tier() -> dict:
    """Import the main simulation's persona_simple.STIGMA_TEMPLATE_BY_TIER without leaving
    that directory on sys.path afterwards. Plain sys.path.insert/append doesn't work
    here in either direction: the main simulation's agent code and this package both ship a same-named
    llm_client.py with different, incompatible APIs (the main simulation's ChatModel-based
    one vs. this package's StructuredClient-based one for GPT-5.5
    generation), and persona_simple.py transitively imports the main simulation's. Inserting
    that directory first breaks this package's own llm_client import; appending it
    breaks persona_simple's. Temporarily prepending it, importing, then
    restoring sys.path/sys.modules keeps both call directions working.
    """
    agents_dir = str(Path(__file__).resolve().parent.parent / "experiment" / "agents")
    saved_path = list(sys.path)
    saved_modules = {
        name: sys.modules[name]
        for name in ("llm_client", "persona_simple", "reddit_persona_profiles",
                     "strategy_selector")
        if name in sys.modules
    }
    for name in saved_modules:
        del sys.modules[name]
    sys.path.insert(0, agents_dir)
    try:
        from persona_simple import STIGMA_TEMPLATE_BY_TIER
        result = dict(STIGMA_TEMPLATE_BY_TIER)
    finally:
        sys.path[:] = saved_path
        for name in ("llm_client", "persona_simple", "reddit_persona_profiles",
                     "strategy_selector"):
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
    return result


STIGMA_TEMPLATE_BY_TIER = _load_stigma_template_by_tier()

# Final ISMI target per trajectory type and initial level.
# stigma_reducing / stigma_reinforcing per the experiment design; general_neutral
# stays flat at the initial level.
_FINAL_ISMI = {
    "stigma_reducing": {2.5: 1.8, 3.0: 2.0, 3.5: 2.3},
    "stigma_reinforcing": {2.0: 3.0, 2.5: 3.2, 3.0: 3.6, 3.5: 3.9},
    "general_neutral": {2.0: 2.0, 2.5: 2.5, 3.0: 3.0, 3.5: 3.5},
}


def target_final_ismi(trajectory_type: str, initial_ismi: float) -> float:
    try:
        return _FINAL_ISMI[trajectory_type][float(initial_ismi)]
    except KeyError as exc:  # pragma: no cover - guarded by config validation
        raise ValueError(
            f"no final ISMI defined for {trajectory_type} @ {initial_ismi}"
        ) from exc


def _linear_targets(initial: float, final: float, n: int) -> list[float]:
    """Session-1 == initial, session-N == final, linearly interpolated."""
    if n == 1:
        return [round(initial, 1)]
    step = (final - initial) / (n - 1)
    return [round(initial + step * i, 1) for i in range(n)]


def _tier_for(target_ismi: float) -> str:
    """Same 3-band cutoffs the production pipeline measures personas against
    (see measure_ismi29.py's _TIERS / CLAUDE.md): severe > 3.00, moderate
    2.50-3.00, mild < 2.50. Ordered high to low so the first match wins."""
    if target_ismi >= 3.00:
        return "severe"
    if target_ismi >= 2.50:
        return "moderate"
    return "mild"


_TIER_SPAN = {"severe": (3.00, 4.00), "moderate": (2.50, 3.00), "mild": (1.00, 2.50)}

# Below this many points from a tier's OWN low edge, a session sits close
# enough to the boundary that the full-strength template text (written for
# the tier's typical/interior case) overstates it -- see _register_for.
_EDGE_MARGIN = 0.15


def _register_for(target_ismi: float, is_opening_session: bool = False) -> str:
    """The STIGMA_TEMPLATE_BY_TIER instruction for this session's target,
    plus an edge-softening note when target_ismi sits at the LOW edge of its
    tier (e.g. 3.0, the minimum value still classified severe), and an extra
    reinforcement note when this is the trajectory's OPENING session.

    STIGMA_TEMPLATE_BY_TIER's three templates are the same ones the
    production patient pipeline uses and are not rewritten here -- they were
    authored and validated as whole-tier instructions, not per-target-value
    ones, so tightening their own wording risks drifting this dataset's
    register away from what real runs actually produce. But the trajectory
    generator calls _register_for with a specific numeric target, and a
    session sitting right at a tier's low edge is the version of that tier
    least like its template's typical case -- concretely: session 1 of the
    stigma_reducing trajectory always lands at target_ismi=3.0, the exact
    minimum of "severe," and got the same "FULL chain, closed, no hedge
    anywhere" instruction as a 3.9 would. Measured directly (sanity_check_2,
    2026-08-15): with 5 peer disclosure turns pushing back inside that one
    session, GPT-5.5 wrote session 1 with the patient already conceding
    ground ("I can see how it fit for you") -- a partial-closure move the
    severe template explicitly forbids -- and the resulting belief_update
    measured well below the 3.0 target from session 1 onward, an error that
    then propagated through every later session's carried-forward belief
    text. The fix is not loosening the template (severe still means severe)
    but naming, in addition to it, that THIS session sits at the tier's own
    boundary and must hold the line against exactly the kind of pressure a
    multi-turn peer exchange applies -- not softening the register, holding
    it under more pushback than the template's own examples assume.

    is_opening_session adds a second, independent reinforcement on top of the
    edge note (2026-08-16): even after the edge note and a peer-assertiveness
    cap (see prompts.py's severe-tier reaction_rule), session 1's measured
    belief_update still landed ~0.3 below its 3.0 target across 5 independent
    generations. The remaining gap isn't about the peer being too pushy
    anymore -- it's that ANY exchange at all, however mild, gives
    belief_update's call1 something to grade as "engaged", and grading
    something as engaged is itself read as movement. Session 1 is also the
    one session with zero prior history to establish HOW closed this chain
    starts out -- every later session's register is read relative to a
    carried-forward belief_update that already anchors severe/moderate/mild,
    but session 1 has nothing before it to anchor against except this
    instruction. So it gets the strongest, most repetitive statement of
    closure of any session in the trajectory, independent of how close its
    target sits to a tier boundary.
    """
    tier = _tier_for(target_ismi)
    template = STIGMA_TEMPLATE_BY_TIER[tier]
    low, high = _TIER_SPAN[tier]
    notes = []
    if target_ismi - low < _EDGE_MARGIN:
        notes.append(
            f"\n\nThis session's target sits at the LOW edge of the '{tier}' "
            f"band ({target_ismi}, where {tier} begins at {low}) -- this is "
            "the version of this band closest to sliding into the band below "
            "it, so hold the line on the instruction above with LESS give "
            "than a session in the middle of this band would need, even "
            "though the peer's disclosures in this session may be direct and "
            "repeated. The patient does not concede ground, soften a claim, "
            "or partially agree with the peer within this session just "
            "because the peer pushed hard -- any movement toward the next "
            "band happens ACROSS sessions, via the trajectory, never within "
            "one session in response to how much the peer said."
        )
    if is_opening_session:
        notes.append(
            "\n\nThis is the OPENING session -- there is no prior session to "
            "carry a softer tone in from, so this session alone has to "
            "establish how closed this chain starts out. State the closure "
            "claims above more than once, in more than one of the patient's "
            "turns, in different concrete situations (not the same sentence "
            "repeated -- different moments that all land on the same flat "
            "conclusion). Do not let the patient's LAST turn of the session "
            "read any softer than their FIRST turn -- if anything, the "
            "patient's closing turn should restate the closure as flatly or "
            "more flatly than the opening post, not trail off into "
            "uncertainty just because the conversation is ending."
        )
    return template + "".join(notes)


def _subscale_focus(trajectory_type: str, i: int, n: int) -> str:
    """Which subscale the session most emphasises. i is 0-based session index."""
    p = i / (n - 1) if n > 1 else 0.0

    if trajectory_type == "stigma_reducing":
        # alienation -> withdrawal -> stereotype -> resistance -> integrated
        if i == n - 1:
            return "integrated"
        if p < 0.20:
            return "alienation"
        if p < 0.45:
            return "social_withdrawal"
        if p < 0.70:
            return "stereotype_endorsement"
        return "stigma_resistance"

    if trajectory_type == "stigma_reinforcing":
        # deepening stigma; never integrates
        if p < 0.25:
            return "alienation"
        if p < 0.50:
            return "social_withdrawal"
        if p < 0.75:
            return "stereotype_endorsement"
        return "alienation"

    # general_neutral: nothing shifts; nominal baseline focus
    return "social_withdrawal"


def _peer_intervention_mechanism(trajectory_type: str, i: int, n: int) -> str:
    """What the other party's turn should do this session.

    For stigma_reducing, this names one of the 5 strategies
    simulation/experiment/agents/strategy_selector.py's production peer actually selects
    from (Davidson et al. 2012; Corrigan, Larson & Rusch 2009; Watson 2019),
    not the earlier self-disclosure-only design this dataset used before
    2026-08. Early sessions favor the two moves aimed at someone who sounds
    isolated or convinced this is permanent (hopeful_disclosure,
    attuned_encouragement); treatment_modeling anchors the middle, since it
    is the one move that speaks to what asking for help would say about the
    patient and belongs whenever help/weakness/being a burden comes up;
    later sessions favor the two moves that read the patient as someone
    already coping (strengths_focused_support, helper_reversal), matching a
    patient who has started to show small signs of managing this. For
    stigma_reinforcing, this is NOT a peer with shared lived experience --
    it is just another anonymous commenter reacting coldly/dismissively to
    the patient's disclosure, with no self-disclosure of their own. The
    stages below are aligned to the session indices where _tier_for's tier
    actually changes (mild -> moderate -> severe) for the 2.0 -> 3.0
    trajectory, so the commenter's shift in mechanism lines up with the
    visible shift in stigma-chain register rather than sitting on an
    arbitrary midpoint.
    """
    p = i / (n - 1) if n > 1 else 0.0

    if trajectory_type == "stigma_reducing":
        if p < 0.2:
            return "peer_hopeful_disclosure"
        if p < 0.35:
            return "peer_attuned_encouragement"
        if p < 0.70:
            return "peer_treatment_modeling"
        if p < 0.85:
            return "peer_strengths_focused_support"
        return "peer_helper_reversal"

    if trajectory_type == "stigma_reinforcing":
        if p < 0.2:
            return "comment_minimizes_and_dismisses"
        if p < 0.8:
            return "comment_fixes_trait_label"
        return "comment_forecloses_hope_flatly"

    # general_neutral: no lived-experience disclosure, no reframing, no
    # deflection to a different topic -- a brief, present, neutral reaction
    # only (see _PEER_INSTRUCTIONS["general_neutral"]).
    return "commenter_reacts_briefly_and_neutrally"


def build_session_targets(
    trajectory_type: str, initial_ismi: float, session_count: int
) -> list[dict]:
    """Return the per-session hidden ground-truth targets."""
    final = target_final_ismi(trajectory_type, initial_ismi)
    targets_ismi = _linear_targets(initial_ismi, final, session_count)
    out: list[dict] = []
    for i, ismi in enumerate(targets_ismi):
        out.append(
            {
                "session": i + 1,
                "target_ismi": ismi,
                "clinical_category": clinical_category(ismi),
                "primary_subscale_focus": _subscale_focus(
                    trajectory_type, i, session_count
                ),
                "intended_register": _register_for(ismi, is_opening_session=(i == 0)),
                "peer_intervention_mechanism": _peer_intervention_mechanism(
                    trajectory_type, i, session_count
                ),
            }
        )
    return out
