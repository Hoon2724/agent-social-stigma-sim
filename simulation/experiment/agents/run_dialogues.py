"""Generate patient opening plus N peer-patient rounds for every persona."""

from __future__ import annotations

import argparse
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

from io_utils import read_json, sha256_file, write_json
from llm_client import OpenAICompatibleModel
from memory import ConversationMemory
from patient_agent import PatientAgent
from peer_agent import PeerSupportAgent, load_or_create_peer_background
from general_agent import GeneralAgent, RestrainedGeneralAgent
from reddit_persona_profiles import REDDIT_PERSONA_PROFILES
from persona_simple import facts_block, load_or_create_stigma_hypothesis, load_or_create_prior
from strategy_selector import complete_json
from belief_update import belief_update_avg, initial_selfreport, narrative_of

HERE = Path(__file__).resolve().parent

# RestrainedGeneralAgent is the only bystander variant in active use -- see
# its docstring for why the earlier real-reply-pool variant was replaced.
_GENERAL_VARIANTS = {"restrained": RestrainedGeneralAgent}


def _build_commenters(agent_type: str, model, peer_persona_path: Path, n_general: int,
                      rng: random.Random, dynamic_backstory: str | None = None,
                      general_variant: str = "restrained",
                      ) -> list[tuple[str, object]]:
    """The people who reply to the patient's post, as (commenter_id, agent) pairs.

    Every condition gets the same n_general bystanders; C2 (`peer`) adds the
    peer on top. Matching the counts matters — if C2 simply had one more
    commenter than C1, a difference between conditions could be the extra reply
    rather than the peer.

    Bystanders are RestrainedGeneralAgent -- see its docstring for what it
    controls for and why the earlier real-reply-pool variant was replaced.
    """
    general_cls = _GENERAL_VARIANTS[general_variant]
    commenters: list[tuple[str, object]] = []
    for index in range(n_general):
        commenter_id = f"general_{index}"
        commenters.append((commenter_id,
                           general_cls(model, rng=rng, commenter_id=commenter_id)))

    if agent_type == "peer":
        commenters.append(("peer", PeerSupportAgent(
            model, persona_path=peer_persona_path, dynamic_backstory=dynamic_backstory)))
    elif agent_type != "general":
        raise ValueError(f"Unknown agent_type: {agent_type}")

    # Zero commenters (--agent-type general --n-general 0) is now a deliberate
    # condition -- the "silence" null control (2026-08-02): the patient posts
    # and nothing ever replies, to check whether belief_update's own narrative
    # generation still drifts toward "things are getting a little better"
    # with zero social input. Combine with --rounds 0 so the round loop (built
    # to have someone reply each round) never runs; only the session-opening
    # post happens each week. Left as an error for any other accidentally-empty
    # configuration would be nice, but there is no such configuration -- peer
    # always adds exactly one commenter above, so an empty list only happens
    # via general/n_general=0, which is now intentional.
    return commenters


def _peer_of(commenters: list[tuple[str, object]]):
    """The peer-support agent among the commenters, if this condition has one."""
    for _, agent in commenters:
        if isinstance(agent, PeerSupportAgent):
            return agent
    return None


def _peer_state(peer) -> dict:
    if isinstance(peer, PeerSupportAgent):
        return {
            "primary_strategy_history": peer.primary_strategy_history,
            "used_disclosures": peer.used_disclosures,
            "reflection_notes": peer.reflection_notes,
            "engagement_history": peer.engagement_history,
        }
    elif isinstance(peer, GeneralAgent):
        return {"primary_strategy_history": peer.stance_history, "used_disclosures": []}
    return {}


def _session_dialogue_log(turns: list[dict], session: int) -> list[dict]:
    """This session's turns as [{"speaker","text"}] for belief_update."""
    role = {"patient": "me", "peer": "peer", "general": "someone", "narrator": "narrator"}
    return [
        {"speaker": role.get(t["speaker"], t["speaker"]), "text": t["content"]}
        for t in turns
        if int(t.get("session", 1)) == session and t["speaker"] != "narrator"
    ]


def _belief_text(belief: dict) -> str:
    """The text-only view of a belief checkpoint injected back into the patient."""
    return narrative_of(belief)


def _measure_pre_subscales(judge, pre_text: str, persona_key: str,
                           reps: int = 3) -> dict[str, float]:
    """ISMI-29 subscale profile of the persona's own pre text.

    Averaged over `reps` administrations for the same reason the downstream
    measurement averages: a single pass is noisy enough to move a subscale
    across a band boundary, and a band boundary is a whole level here.
    Returns {} if every administration fails, which leaves session 0 seeding
    every domain at `moderate` -- flagged loudly rather than silently, since
    an uncalibrated start invalidates the pre/post comparison.
    """
    from ismi29_items import ISMI_SUBSCALES
    from measure_ismi29 import build_pre_innerstate_block, measure_once

    block = build_pre_innerstate_block(pre_text)
    runs = []
    for _ in range(reps):
        try:
            runs.append(measure_once(judge, "", 0.3, experience=block))
        except (ValueError, KeyError, TypeError):
            continue
    if not runs:
        print(f"[Warn] pre ISMI-29 failed for persona_key={persona_key!r}; "
              f"session-0 levels fall back to moderate everywhere and this "
              f"run's pre/post comparison is uncalibrated")
        return {}
    means = {s: round(statistics.fmean(r["subscale_means"][s] for r in runs), 3)
             for s in ISMI_SUBSCALES}
    total = round(statistics.fmean(r["total_mean"] for r in runs), 3)
    print(f"[pre] {persona_key}: ISMI-29 total={total} subscales={means}")
    return means


def _apply_belief(patient, belief: dict) -> None:
    """Carry the self-report forward into the persona for the next session."""
    patient.belief_text = _belief_text(belief) or None
    patient.disclosure = str(belief.get("disclosure") or "secrecy")


def continue_trajectory(model, backstory: str, rounds: int, recent_window: int | None, top_k: int,
                        agent_type: str = "peer", peer_persona_path: Path | None = None,
                        initial_turns: list[dict] | None = None, peer_state: dict | None = None,
                        on_progress=None, seed_post: str | None = None,
                        style_examples: list[str] | None = None,
                        style_description: str | None = None,
                        sessions: int = 1,
                        weekly_triggers: list[str] | None = None,
                        track_belief: bool = False, belief_reps: int = 3,
                        persona_name: str = "the person",
                        initial_belief_history: list[dict] | None = None,
                        belief_box: list[dict] | None = None,
                        persona_key: str = "",
                        peer_cache_key: str = "",
                        dynamic_peer_backstory: bool = False,
                        n_general: int = 3,
                        comment_order: str = "shuffled",
                        seed: int = 42,
                        pre_belief: dict | None = None,
                        recent_sessions: int = 1,
                        belief_prior: float = 0.85,
                        belief_hypothesis: str = "",
                        general_variant: str = "restrained",
                        ) -> tuple[list[dict], dict, list[dict]]:
    memory = ConversationMemory(recent_sessions=recent_sessions,
                                recent_window=recent_window,
                                turns=list(initial_turns or []))
    patient = PatientAgent(model, backstory, style_examples=style_examples or [],
                           style_description=style_description)

    dynamic_backstory = None
    if dynamic_peer_backstory and agent_type == "peer":
        opening_text = next(
            (t["content"] for t in memory.turns
             if t.get("speaker") == "patient" and int(t.get("session", 1)) == 1
             and int(t.get("round", 0)) == 0),
            None,
        )
        if opening_text is None:
            # Fresh run: the peer's backstory must be seeded from something
            # the patient has actually posted, so generate session 1's
            # opening now — before the peer exists — and reuse the exact
            # same opening once the session loop reaches session 1 below.
            memory.current_session = 1
            if seed_post:
                opening = patient.opening_from_seed(seed_post)
            elif sessions == 1:
                opening = patient.opening()
            else:
                trigger = weekly_triggers[0] if weekly_triggers else None
                opening = patient.session_opening(trigger, 1, prev_tail=[], previous_openings=[])
            memory.add("patient", opening["patient_response"], 0, {})
            opening_text = opening["patient_response"]
        # Keyed per run (not just per persona): each run's patient opening can
        # differ (e.g. --runs 3 draws a fresh session-1 post each time), and a
        # peer backstory generated from run 1's opening has no reason to fit
        # run 2 or 3's. Caching only by persona_key made every run of a persona
        # share one peer identity, which quietly removed a source of run-to-run
        # variance that --runs is there to capture.
        cache_key = peer_cache_key or persona_key or persona_name
        cache_path = HERE / "config" / "dynamic_peer" / f"peer_backstory_{cache_key}.txt"
        dynamic_backstory = load_or_create_peer_background(cache_path, opening_text, model)

    # One rng drives both the bystanders' stances and the per-round ordering, so
    # a given seed reproduces the whole social configuration of a run.
    rng = random.Random(seed)
    commenters = _build_commenters(
        agent_type, model, peer_persona_path or HERE / "config" / "peer_persona.json",
        n_general, rng, dynamic_backstory=dynamic_backstory, general_variant=general_variant)
    peer = _peer_of(commenters)
    # Set after the fact rather than at construction: the dynamic-backstory
    # branch above needs `patient` to generate session 1's opening before the
    # peer (and therefore the commenter list) can exist. Taken from the built
    # list so it can't drift from the arithmetic in _build_commenters.
    patient.n_commenters = len(commenters)

    if peer_state and peer is not None:
        peer.primary_strategy_history = list(peer_state.get("primary_strategy_history", []))
        peer.used_disclosures = list(peer_state.get("used_disclosures", []))
        peer.reflection_notes = list(peer_state.get("reflection_notes", []))
        peer.engagement_history = list(peer_state.get("engagement_history", []))

    # belief_box (if passed) is the same list the caller's save_progress closes
    # over, so appends here surface in mid-run saves. Fall back to a local list.
    belief_history: list[dict] = belief_box if belief_box is not None else list(initial_belief_history or [])
    if track_belief and belief_history:
        # Resume: seed the patient with the latest carried-forward belief update.
        _apply_belief(patient, belief_history[-1])

    start_session = max((int(t.get("session", 1)) for t in memory.turns), default=1)
    for session in range(start_session, sessions + 1):
        memory.current_session = session
        session_turns = [t for t in memory.turns if int(t.get("session", 1)) == session]
        if not session_turns:
            if session == 1 and seed_post:
                opening = patient.opening_from_seed(seed_post)
            elif sessions == 1:
                opening = patient.opening()
            else:
                trigger = (weekly_triggers[(session - 1) % len(weekly_triggers)]
                           if weekly_triggers else None)
                previous_openings = [t["content"] for t in memory.turns
                                     if t["speaker"] == "patient" and int(t["round"]) == 0]
                # memory.current_session is already this session and it has no
                # turns yet, so recent() is exactly the short-term window the
                # rest of the session will see -- the previous session(s). Was
                # a hardcoded turns[-6:], a third independent notion of "recent"
                # alongside recent_window and the 6 itself; it also passed raw
                # turns, whose commenter_id sits under metadata, so bystanders
                # lost their A/B/C labels in the opening prompt only.
                opening = patient.session_opening(
                    trigger, session, prev_tail=memory.recent(),
                    previous_openings=previous_openings)
            memory.add("patient", opening["patient_response"], 0, {})
            if on_progress:
                on_progress(memory, peer, False)

        completed = max((int(t["round"]) for t in memory.turns
                         if int(t.get("session", 1)) == session), default=0)
        for round_number in range(completed + 1, rounds + 1):
            query = memory.turns[-1]["content"]
            # Reply order is redrawn each round. If the peer always spoke last
            # it would permanently hold two advantages — seeing every other
            # reply first, and sitting directly before the patient's turn — and
            # that position effect would be inseparable from the intervention
            # itself. Commenters see each other: memory.add lands immediately,
            # so whoever speaks second reads the first reply.
            order = (rng.sample(commenters, len(commenters))
                     if comment_order == "shuffled" else list(commenters))
            for commenter_id, agent in order:
                is_peer = agent is peer
                retrieved = memory.retrieve(query, "peer", top_k) if is_peer else []
                result = agent.respond(memory.recent(), retrieved)
                memory.add(
                    "peer" if is_peer else "general",
                    result["peer_response"], round_number,
                    {
                        "strategy": result["strategy"],
                        "disclosure_id": result.get("disclosure_id"),
                        # speaker stays "general"/"peer" because the patient's
                        # prompt distinguishes only those two; this is how a
                        # particular bystander is identified after the fact.
                        "commenter_id": commenter_id,
                    },
                )

            # The patient answers the round's replies as a group, so the
            # retrieval query is all of them rather than the peer's alone.
            round_replies = "\n".join(
                t["content"] for t in memory.turns
                if int(t.get("session", 1)) == session
                and int(t["round"]) == round_number and t["speaker"] != "patient"
            )
            patient_result = patient.respond(
                memory.recent(), memory.retrieve(round_replies, "patient", top_k)
            )
            memory.add("patient", patient_result["patient_response"], round_number,
                       {})
            if on_progress:
                on_progress(memory, peer, session == sessions and round_number == rounds)

        # Reflexion step: the peer reflects on which of its own moves this
        # session preceded the patient staying engaged vs. going quiet, and
        # keeps a short private note for future sessions. Runs every
        # session unconditionally (no success/fail gate) — judged only on
        # openness in what they actually wrote, never on stigma content, so this
        # cannot learn to chase the measured outcome directly.
        if isinstance(peer, PeerSupportAgent):
            session_turns = [t for t in memory.turns if int(t.get("session", 1)) == session]
            peer.reflect_on_session(session_turns)

        # End of session: update the carried-forward belief update from THIS
        # session's whole dialogue plus the previous checkpoint, then inject
        # the text view into the patient for the next session's opening.
        if track_belief:
            dlog = _session_dialogue_log(memory.public_turns(), session)
            if dlog:
                # Session 1 has no prior checkpoint of its own (belief_history
                # starts empty), so without this it is generated with
                # prev_beliefs=None -- disconnected from the pre (session-0)
                # baseline even though session 2 onward always chains from the
                # immediately preceding session. That gap let a pre->session-1
                # delta partly reflect "session 1 generated in isolation"
                # rather than a real change from the pre baseline. Seeding
                # session 1's prev_beliefs with pre_belief closes it; pre_belief
                # itself is never appended to belief_history, so the saved
                # checkpoints still start at session=1 as before.
                prev_beliefs = belief_history[-1:] or ([pre_belief] if pre_belief else None)
                belief = belief_update_avg(
                    model, backstory, persona_name, dlog,
                    prev_beliefs=prev_beliefs,
                    n_reps=belief_reps,
                    prior=belief_prior,
                    hypothesis=belief_hypothesis,
                )
                belief["session"] = session
                belief_history.append(belief)
                _apply_belief(patient, belief)
                # The round-loop's complete=True save fired BEFORE this
                # session's belief existed, so the final session's checkpoint
                # would be lost. Re-save once the last session's belief lands.
                if on_progress and session == sessions:
                    on_progress(memory, peer, True)

    return memory.public_turns(), _peer_state(peer), belief_history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas", type=Path, default=HERE / "outputs" / "backstories.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "dialogues")
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--base-url")
    parser.add_argument("--num-ctx", type=int, default=8192,
                        help="forced context size, so the model stays fully GPU-resident "
                             "instead of ollama's automatic (and observed-flaky) split "
                             "leaving most of it on CPU (see llm_client.py, "
                             "measure_ismi29.py's --num-ctx).")
    parser.add_argument("--num-gpu", type=int, default=99,
                        help="force every layer onto GPU, bypassing ollama's automatic "
                             "split heuristic entirely (see llm_client.OpenAICompatibleModel).")
    parser.add_argument("--ismi-judge-model", default="qwen3:32b",
                        help="model that administers the pre ISMI-29 used to SEED the "
                             "session-0 latent levels. The persona's starting severity is "
                             "measured rather than asserted, so this should be the same "
                             "judge measure_ismi29_run.py uses downstream -- otherwise pre "
                             "is seeded by one instrument and scored by another.")
    parser.add_argument("--ismi-judge-base-url",
                        help="endpoint for --ismi-judge-model (defaults to --base-url).")
    parser.add_argument("--rounds", type=int, default=2,
                        help="Reply-and-respond cycles per session (default 2). Each round "
                             "is: every commenter replies once, then the patient answers. "
                             "Two rounds give the peer a real exchange with the patient "
                             "while keeping a session close to a plausible thread depth.")
    parser.add_argument("--sessions", type=int, default=1,
                        help="Number of weekly sessions per run (1 = original single-session behavior). "
                             "Each session opens with a fresh patient post grounded in that week's "
                             "trigger; dialogue memory and peer state carry across sessions.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--levels", nargs="*", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-triggers", action="store_true",
                        help="Ablation: run without weekly_triggers, so each session's "
                             "opening comes from the persona's own backstory instead of a "
                             "fixed scene. The trigger list is shared verbatim across all "
                             "four stigma tiers and is written at the severe end, so this "
                             "isolates how much of the tier collapse observed at session 1 "
                             "(tier spread SD 0.587 at pre -> 0.162 at s1) comes from it.")
    parser.add_argument("--track-belief", action="store_true",
                        help="After each session, update a carried-forward ISMI-domain belief state "
                             "and inject its text view into the "
                             "patient's next session. Adds belief_history to the run JSON.")
    parser.add_argument("--belief-reps", type=int, default=3,
                        help="Unused by the current belief update (Bayesian, "
                             "belief.estimate_update_factor votes internally via its own "
                             "_UPDATE_FACTOR_REPS). Kept only so old shell scripts "
                             "invoking this flag don't fail to parse.")
    parser.add_argument("--agent-type", choices=["peer", "general"], default="peer",
                        help="Who replies to the patient's post. All conditions include "
                             "--n-general bystanders; peer (C2) adds the peer agent, "
                             "general (C1) adds no one else.")
    parser.add_argument("--n-general", type=int, default=3,
                        help="Bystanders replying to each post (default 3, the median "
                             "comment count on mental-health subreddit posts). Same in "
                             "every condition, so C1 and C2 differ only by the peer.")
    parser.add_argument("--general-variant", choices=["restrained"], default="restrained",
                        help="Which bystander class fills the --n-general slots. "
                             "'restrained': no exemplar, a directly-stated target register "
                             "(short, not over-comforting, not dismissive, no pushed "
                             "solutions) -- a mechanism control (ordinary contact minus "
                             "peer's specific lived-experience/reframing moves).")
    parser.add_argument("--comment-order", choices=["shuffled", "fixed"], default="shuffled",
                        help="shuffled (default): redraw reply order every round, so the "
                             "peer's position cannot be confounded with its effect. "
                             "fixed: bystanders first, peer last.")
    parser.add_argument("--recent-sessions", type=int, default=None,
                        help="Previous sessions kept in short-term memory alongside the "
                             "current one (default 1, from config/experiment.json). "
                             "Short-term is scoped by session, not turn count, so it stays "
                             "the same size when --rounds or --n-general change; retrieve() "
                             "searches only what falls outside it.")
    parser.add_argument("--recent-window", type=int,
                        help="LEGACY: pin short-term to a fixed turn count instead, ignoring "
                             "session boundaries (pre-2026-08-02 behavior; the old default was "
                             "11, one C2 session). Only for reproducing an older run — a fixed "
                             "count silently stops meaning 'one session' when the thread shape "
                             "changes, and differs between C1 (9 turns/session) and C2 (11).")
    parser.add_argument("--peer-persona", type=Path, default=HERE / "config" / "peer_persona.json",
                        help="Path to peer persona JSON (peer condition only)")
    parser.add_argument("--dynamic-peer-backstory", action="store_true",
                        help="Generate the peer's backstory once from the patient's own "
                             "session-1 opening post (cached under config/dynamic_peer/), "
                             "instead of the fixed backstory in config/peer_persona.json.")
    parser.add_argument("--pre-narratives", "--session0-beliefs", type=Path,
                        dest="pre_narratives",
                        default=HERE / "config" / "session0_beliefs.json",
                        help="JSON {persona_key: belief_update} of session-0 belief updates "
                             "(config/session0_beliefs.json, written by "
                             "make_session0_belief.py). When --track-belief is set, session 1's "
                             "belief checkpoint chains from this instead of starting from "
                             "nothing -- see continue_trajectory's pre_belief handling. A "
                             "persona missing from this file falls back to the old behavior "
                             "(prev_beliefs=None at session 1), with a warning. The "
                             "--pre-narratives spelling is kept so existing scripts keep "
                             "working; the file it points at is the session-0 belief update, "
                             "not a separate 'pre narrative' artifact.")
    args = parser.parse_args()

    data = read_json(args.personas)
    config = read_json(HERE / "config" / "experiment.json")
    selected = [p for p in data["personas"] if not args.levels or p["level"] in args.levels]
    model = OpenAICompatibleModel(args.model, args.base_url,
                                  num_ctx=args.num_ctx, num_gpu=args.num_gpu)
    random.seed(config["random_seed"])

    pre_narratives: dict[str, str] = {}
    if args.pre_narratives:
        pre_narratives = read_json(args.pre_narratives)

    for persona in selected:
        persona_key = persona.get("persona_key", "")
        profile = REDDIT_PERSONA_PROFILES.get(persona_key, {}).get("profile", {})
        weekly_triggers = persona.get("weekly_triggers") or profile.get("weekly_triggers")
        if args.no_triggers:
            # Ablation: the same weekly_triggers list is shared by all four
            # stigma tiers of a diagnosis (_build_registry only lets the tiers
            # differ on baseline_self_stigma_pattern / social_situation /
            # avoided_topics), and every trigger is written at the severe end
            # ("cried in the stockroom", "feeling like a fraud"). Dropping them
            # tests whether that shared material is what erases the tier
            # difference by session 1. Without triggers session_opening falls
            # back to "write about how this past week has actually been".
            weekly_triggers = None
        style_description = profile.get("sns_writing_style")
        # Facts only -- age, job, how long this has been going on. Self-stigma
        # is not part of the description of the person here, it IS the belief
        # state (belief.belief_update_bayes), rendered fresh each week from
        # its current probability (see persona_simple). A fixed prose
        # backstory stating the stigma every week would compete with a
        # probability that is supposed to be free to move, and the fixed text
        # would win -- that competition is what made the six-paragraph
        # backstory design unable to show any change across eight weeks.
        backstory = facts_block(persona_key)
        # Generated once per persona and cached (config/stigma_hypothesis/),
        # not written by hand per diagnosis x tier -- see
        # persona_simple.generate_stigma_hypothesis for why: the tier
        # templates encode the SHAPE (Corrigan's chain at that severity), and
        # this persona's own baseline_self_stigma_pattern supplies the
        # diagnosis-specific content, so a new diagnosis needs no new prose.
        belief_hypothesis = load_or_create_stigma_hypothesis(model, persona_key)
        # P0 read off how firmly THIS sentence is worded (persona_simple.
        # estimate_prior), replacing the hand-picked TIER_PRIOR numbers.
        # Must run after the hypothesis above -- there is nothing to read the
        # firmness of before that text exists.
        belief_prior = load_or_create_prior(model, persona_key, belief_hypothesis)

        pre_belief = None
        if args.track_belief and args.pre_narratives:
            pre_text = pre_narratives.get(persona_key)
            if pre_text:
                pre_belief = {"belief_update": pre_text}
            else:
                print(f"[Warn] no pre-narrative for persona_key={persona_key!r} in "
                      f"{args.pre_narratives}; session 1 will start with no prior checkpoint")

        for run in range(1, args.runs + 1):
            path = args.output_dir / f"level_{persona['level']:02d}" / f"run_{run:02d}.json"
            existing = read_json(path) if path.exists() and not args.overwrite else None
            if (existing and existing.get("status") == "complete"
                    and existing.get("rounds") == args.rounds
                    and existing.get("sessions", 1) == args.sessions):
                print(f"[Skip] {path} already complete")
                continue
            initial_turns = existing.get("turns", []) if existing else []
            initial_state = existing.get("private_generation_metadata", {}) if existing else {}
            initial_belief = existing.get("belief_history", []) if existing else []
            persona_name = persona.get("name") or profile.get("meta", {}).get("name", "the person")

            if args.track_belief and not initial_belief:
                # Seed session 0 so session 1's opening post sees the initial
                # ISMI-domain self-stigma posture. Without this,
                # belief_text stays None through all of session 1 --
                # belief_update only fires at the END of a session, so
                # nothing carries the persona's tier into session 1's
                # disclosure decision.
                #
                # Cached per persona_key (2026-08-08), unlike every later
                # session's narration which stays a fresh sample each run.
                # Session 0 has no run-specific input to be a fresh sample OF
                # -- backstory/hypothesis/P0 are already identical and cached
                # across every run of a tier, and dialogue_log is empty here,
                # so the only thing that varied run-to-run was narrate_belief's
                # own sampling noise on IDENTICAL inputs. That noise then
                # became the "pre" measurement (measure_ismi29_run.py
                # --pre-from-session0), so peer and general runs of the same
                # tier were being compared against different, noise-only
                # starting points -- observed directly in batch09: severe
                # pre ranged 2.759-2.92 across runs of the same tier/condition
                # for no reason tied to the condition itself. Caching removes
                # that noise from the baseline without touching sessions 1+,
                # where run-to-run variation is real (each run's own dialogue).
                session0_cache = (HERE / "config" / "stigma_hypothesis"
                                  / f"{persona_key}.session0.ismi_anchor.json")
                if session0_cache.exists():
                    initial_belief = [read_json(session0_cache)]
                else:
                    # The pre text must be the persona's OWN, supplied via
                    # --pre-narratives. It used to fall back to generating one
                    # with narrate_belief, which quietly changed what the pre
                    # measurement is measuring: the latent levels are now
                    # seeded from the ISMI-29 profile of this text, so a
                    # generated stand-in would make the run's starting
                    # severity a property of narrate_belief rather than of the
                    # persona. Raise instead of substituting.
                    seed_text = (
                        pre_narratives.get(persona_key, "")
                        if args.pre_narratives else ""
                    )
                    if not seed_text:
                        raise SystemExit(
                            f"--track-belief needs a pre narrative for "
                            f"persona_key={persona_key!r}. Pass --pre-narratives "
                            f"with a JSON {{persona_key: text}} containing it; the "
                            f"session-0 latent levels are seeded from that text's "
                            f"measured ISMI-29 profile, so there is nothing safe to "
                            f"substitute for it."
                        )
                    if seed_text:
                        # The pre narrative IS the session-0 self-report -- the
                        # persona's own account of themselves before anything
                        # happens, in the same voice and of the same kind as
                        # every later session's. Nothing is generated and
                        # nothing is seeded from a constant, so pre and post
                        # are two measurements of the same sort of material.
                        session0_cache.parent.mkdir(parents=True, exist_ok=True)
                        initial_belief = [
                            initial_selfreport(model, backstory,
                                               persona_name, seed_text)
                        ]
                        write_json(session0_cache, initial_belief[0])

            # Mutable holder so the mid-run save_progress callback can serialize
            # whatever belief checkpoints exist so far (belief is updated at each
            # session boundary, inside continue_trajectory, which mutates this).
            belief_box: list[dict] = list(initial_belief)

            def save_progress(memory, peer, complete, _path=path, _persona=persona, _run=run,
                              _backstory=backstory, _belief=belief_box):
                state = _peer_state(peer)
                peer_persona_hash = sha256_file(args.peer_persona) if args.agent_type == "peer" else None
                write_json(_path, {
                    "schema_version": 1,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "complete" if complete else "in_progress",
                    "level": _persona["level"],
                    "run": _run,
                    "rounds": args.rounds,
                    "sessions": args.sessions,
                    "model": args.model,
                    "agent_type": args.agent_type,
                    "n_general": args.n_general,
                    "general_variant": args.general_variant,
                    "recent_sessions": (args.recent_sessions
                                        if args.recent_sessions is not None
                                        else config.get("recent_sessions", 1)),
                    "recent_window": args.recent_window,
                    "comment_order": args.comment_order,
                    "seed": config["random_seed"] + _run,
                    "seed_mode": "reddit" if _persona.get("seed_post") else "fabricated",
                    "seed_meta": _persona.get("seed_meta"),
                    "backstory": _backstory,
                    "turns": memory.public_turns(),
                    "belief_history": list(_belief),
                    "private_generation_metadata": state,
                    "prompt_hashes": {
                        "patient": sha256_file(HERE / "prompts" / "patient_agent.txt"),
                        "selector": sha256_file(HERE / "prompts" / "strategy_selector.txt") if args.agent_type == "peer" else None,
                        "peer": sha256_file(HERE / "prompts" / "response_generator.txt") if args.agent_type == "peer" else None,
                        "peer_persona": peer_persona_hash,
                    },
                })

            print(f"[Run] level={persona['level']} run={run} agent={args.agent_type} "
                  f"sessions={args.sessions} rounds={args.rounds} track_belief={args.track_belief}")
            _turns, _state, belief_history = continue_trajectory(
                model, backstory, args.rounds,
                # None (the default) selects session-scoped short-term memory;
                # only an explicit --recent-window pins it to a turn count.
                args.recent_window,
                config["retrieved_memories"],
                args.agent_type, args.peer_persona,
                initial_turns, initial_state, save_progress,
                seed_post=persona.get("seed_post"),
                style_examples=persona.get("few_shot_examples"),
                style_description=style_description,
                sessions=args.sessions,
                weekly_triggers=weekly_triggers,
                track_belief=args.track_belief,
                belief_reps=args.belief_reps,
                persona_name=persona_name,
                initial_belief_history=initial_belief,
                belief_box=belief_box,
                persona_key=persona_key,
                peer_cache_key=f"{persona_key or persona_name}_run{run:02d}",
                dynamic_peer_backstory=args.dynamic_peer_backstory,
                n_general=args.n_general,
                general_variant=args.general_variant,
                recent_sessions=(args.recent_sessions
                                 if args.recent_sessions is not None
                                 else config.get("recent_sessions", 1)),
                comment_order=args.comment_order,
                # Offset by run so the five runs of a persona get different
                # stances and reply orders, while any single run stays
                # reproducible from the config seed.
                seed=config["random_seed"] + run,
                pre_belief=pre_belief,
                belief_prior=belief_prior,
                belief_hypothesis=belief_hypothesis,
            )


if __name__ == "__main__":
    main()
