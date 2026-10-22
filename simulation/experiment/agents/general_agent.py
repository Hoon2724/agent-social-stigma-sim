"""
general_agent.py — SNS bystander agent for the RQ1 peer-support pipeline.

Only RestrainedGeneralAgent is active: a mechanism control that states its
register directly (short, not over-comforting, not dismissive) rather than
drawing from a real-reply pool. See RestrainedGeneralAgent's docstring for
why the pool-based variant it replaced was dropped.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from quality import MAX_REGENERATION_ATTEMPTS, is_near_duplicate
from strategy_selector import format_dialogue

HERE = Path(__file__).resolve().parent


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {"peer_response": text.strip()}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {"peer_response": text.strip()}


class GeneralAgent:
    def __init__(self, model, stance: str | None = None,
                 rng: random.Random | None = None,
                 commenter_id: str | None = None) -> None:
        self.model = model
        self.stance = stance
        # Which commenter this instance is in the thread, so its own earlier
        # replies can be labelled as its own rather than as somebody else's.
        # In a 1:1 run the only other comment under the post IS this agent's
        # previous one, and showing it as "another commenter" invites the
        # agent to answer itself.
        self.commenter_id = commenter_id
        self.stance_history: list[str] = []
        # Everything this instance has said, so a stock reaction ("huh") is
        # caught even when it fell out of the recent-turns window. recent_turns
        # only holds the sliding window; this is the full trail.
        self.own_texts: list[str] = []
        # Per-turn stance draws come from the run's own generator when one is
        # given, so a run replays identically from its seed. Falling back to the
        # `random` module keeps older callers working -- run_dialogues.py seeds
        # it globally, so those runs stay reproducible too, just not per-run.
        self.rng = rng or random

    # How the thread reads to a bystander. It has no way to know one of the
    # other commenters is a designated peer, so every non-poster is just
    # another commenter; format_dialogue still appends (A)/(B) per
    # commenter_id so they read as distinct people.
    _GENERAL_SPEAKER_LABELS = {
        "patient": "The person who posted",
        "peer": "Another commenter",
        "general": "Another commenter",
        "_self": "You replied earlier",
        "narrator": "",
    }

    def _drive_by_block(self, recent_turns: list[dict]) -> str:
        """The session's opening post, and nothing else.

        A bystander scrolling a feed reacts to a post. It does not see the
        replies that came after, its own earlier comment, or what the poster
        said back -- because in a real feed it has already scrolled on.

        Showing the running thread instead (what _thread_block does, and what
        this agent used until 2026-08-09) turns the bystander into a
        conversational partner over a session's rounds, which is the peer's
        role. Three pressures do it, and they compound: the same commenter_id
        persists for 8 sessions x 5 rounds, so it meets this person 40 times;
        its own earlier replies come back labelled "You replied earlier", so it
        has to stay consistent with the warmth it already showed; and from
        round 2 the "poster's most recent message" it is told to answer is a
        reply addressed to IT, so it is holding a conversation rather than
        commenting on a post. Nothing in the exemplar pool covers that -- every
        row is one reply to one stranger's post -- so with no example to match,
        the model falls back on ordinary conversational behaviour, which is
        mirroring.

        Measured across the 9 general runs: similarity between a bystander
        reply and the patient turn just before it climbed monotonically with
        round, 0.282 -> 0.312 -> 0.346 -> 0.361 -> 0.367. The control condition
        was drifting into the treatment as each session went on.

        This also puts the prompt back in agreement with itself: it already
        tells the agent "you will not talk to this person again after this".
        """
        turns = self.current_session_turns(recent_turns)
        opening = next((t for t in turns if t.get("speaker") == "patient"), None)
        if opening is None:
            return "(no messages yet)"
        return format_dialogue([opening],
                               speaker_labels=self._GENERAL_SPEAKER_LABELS)

    @staticmethod
    def current_session_turns(recent_turns: list[dict]) -> list[dict]:
        """This session's turns only, out of the short-term window.

        A bystander scrolling a feed sees the thread it is replying into, but
        has no memory of last week — so its horizon is one session, narrower
        than the patient's and the peer's. memory.recent() carries `session`
        for exactly this. Turns without one (the synthetic narrator line at a
        session boundary) are dropped, which is what we want here anyway.
        """
        sessions = [t["session"] for t in recent_turns if t.get("session") is not None]
        if not sessions:
            return list(recent_turns)
        latest = max(sessions)
        return [t for t in recent_turns if t.get("session") == latest]


class RestrainedGeneralAgent(GeneralAgent):
    """general-v2 (2026-08-08): a mechanism control, not an ecological one.

    RealisticGeneralAgent (v1) targets "what a real random bystander
    actually writes" -- register drawn from the real EPITOME pool. That
    objective turned out to be entangled with a separate failure: when the
    drawn exemplar's own post/reply pair is topically unrelated to the
    current thread, the model cannot transplant just the TONE without the
    exemplar's content, and falls back to its own default (warm/coaching)
    voice -- observed directly, e.g. an exemplar about an unrelated TV show
    produced "Happens to the best of us... take a breath", a register the
    exemplar itself never had. That confounds "does peer's mechanism work"
    with "did tone-transfer from an unrelated exemplar succeed."

    This agent removes the exemplar/pool step entirely and states the target
    register directly: short, not over-comforting, not emotionally
    over-invested, not dismissive of the concern, no pushed solutions. It is
    deliberately NOT trying to reproduce the pool's true level-0/1/2 mix --
    it is a clean "ordinary contact minus peer's specific mechanisms" arm,
    to run alongside v1 and separate two questions that were previously
    tangled together: (1) does removing peer's lived-experience/reframing
    mechanism change the patient's response, independent of (2) whether
    register-transfer from a real-data exemplar succeeds. RealisticGeneralAgent
    (the v1 exemplar-pool variant this replaced) is no longer in the codebase --
    this is the only bystander variant now in active use.
    """

    _SYSTEM_PROMPT = (
        "You are a random person scrolling a social media platform called "
        "'minibook'. You see a stranger's post and leave ONE quick reply. "
        "You are not a therapist, counselor, or close friend, and you will "
        "not talk to this person again.\n\n"
        "Write ONE or TWO short sentences, nothing longer. Do not comfort "
        "them at length, do not tell them how they should feel about "
        "themselves, and do not offer reassurance about their worth, their "
        "effort, or their future. At the same time, do not mock, dismiss, "
        "or minimize what they said -- do not imply it is nothing or "
        "that they are overreacting. Do not offer advice, a solution, or "
        "a suggested next step unless they directly asked a factual "
        "question. Do not share your own similar experience, present "
        "yourself as someone who has been there, model self-care from lived "
        "experience, or combine empathy with recovery-oriented expectation -- "
        "those are peer-support mechanisms, not a "
        "passing bystander's role. React the way an ordinary stranger passing through a "
        "comment section would: brief, plain, present, and not invested in "
        "changing how they see themselves.\n\n"
        "Below is the thread as it stands. Reply to the poster's most "
        "recent message.\n\n"
        'Output ONLY a JSON object: {"peer_response": "<your reply>"}'
    )

    def respond(self, recent_turns: list[dict], _retrieved: list[dict] | None = None) -> dict:
        # Drive-by view (2026-08-09), matching RealisticGeneralAgent's fix:
        # this class has its own respond() and was still calling
        # _thread_block (the full running thread), so switching the general
        # condition to this variant would have silently reintroduced the
        # mirroring bug _drive_by_block was written to remove -- this class's
        # own docstring already says it is "not talk to this person again",
        # which only holds if it cannot see what they said back.
        thread_block = self._drive_by_block(recent_turns)
        recent_own_texts = self.own_texts
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": thread_block},
        ]

        response_text = ""
        for _ in range(MAX_REGENERATION_ATTEMPTS):
            # Lowered from 1.0 (2026-08-09): a control condition intended to
            # read as low-effort/near-null should not also be the noisiest
            # generator in the pipeline. High temperature was adding
            # session-to-session variance in HOW a given register got
            # rendered, on top of the (intentional, kept) variance in WHICH
            # register this turn draws.
            raw = self.model.complete(messages, temperature=0.4)
            parsed = _extract_json(raw)
            response_text = str(parsed.get("peer_response", raw.strip())).strip()
            norm = response_text.strip().lower().rstrip(".!?")
            seen_stock = norm in {t.strip().lower().rstrip(".!?") for t in self.own_texts}
            if not (is_near_duplicate(response_text, recent_own_texts) or seen_stock):
                break

        self.own_texts.append(response_text)
        self.stance_history.append("general_restrained")
        return {
            "peer_response": response_text,
            "strategy": {"primary": "general_restrained", "secondary": None,
                         "evidence": "", "rationale": ""},
            "disclosure_id": None,
        }
