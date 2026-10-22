"""Two-stage peer-support agent grounded in HOP/COP disclosure support."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from llm_client import ChatModel
from quality import (
    MAX_REGENERATION_ATTEMPTS,
    classify_figurative_voted,
    classify_template_repeat,
    enforce_plain,
    is_near_duplicate,
    revise_off_template,
)
from strategy_selector import StrategyDecision, StrategySelector, complete_json, format_dialogue


HERE = Path(__file__).resolve().parent


def generate_peer_background(patient_opening_text: str, model: ChatModel) -> str:
    """Generate a peer backstory seeded only from what the patient has
    actually posted publicly (never from a hidden diagnosis/persona field),
    so the peer's knowledge stays limited to what a real SNS commenter could
    see."""
    system = (
        "You are writing the personal background of a peer support figure "
        "on a social media platform."
    )
    user = f"""[THE PATIENT'S PUBLIC POST]
{patient_opening_text}

Before writing, silently identify what this post is really about
emotionally — is it a body reacting faster than thought can catch up (a
startle/freeze response), a fear of being seen, hiding an internal state
from others? If the post describes an involuntary reaction to a sudden
trigger (flinching, dropping, freezing, hitting the floor, going numb),
treat that as a nervous-system/emotional reaction — do NOT invent a
physical illness, disability, or mobility explanation for it unless the
post itself names one.

Write 4-6 sentences, first person, describing this peer's own past
experience. Requirements:
- Do NOT use diagnostic labels or medical terminology, and do not name a
  condition. This peer talks about what happened to them, not what they had.
- Write in behaviors and sensations, not symptom categories — but keep those
  behaviors and sensations grounded in the emotional reading above, not in a
  literal physical or medical explanation.
- The experience must feel genuinely parallel to the struggle shown in the
  post above — not generic.
- Include, in this order: a period of handling it alone because asking would
  have meant something bad about them; what they were afraid seeing a
  professional would say about them; the fact that they eventually went; and
  what it actually turned out to be instead.
- The person they went to MUST be a professional — say "a counsellor", "a
  therapist", or "a doctor" in those words. A friend, a family member, or "just
  someone who listened" is NOT this and does not count: the whole point is what
  it turned out to mean about them to have gone to someone whose job it was.
  Describe the appointment plainly, with no clinical detail.
- End on a current stance that is steadier but unfinished. The peer is not
  "healed" — just no longer ruled by the idea that needing help meant
  something was wrong with who they are.
- Tone: honest, grounded, slightly understated — not inspirational, and not a
  recommendation. Nothing here should read as advice to the reader.

Output only the background text. No labels, no headers, no extra explanation."""
    return model.complete([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], temperature=0.85).strip()


def load_or_create_peer_background(cache_path: Path, patient_opening_text: str, model: ChatModel) -> str:
    """Generate the peer's backstory once per patient and reuse it across
    every session of that patient's trajectory, the same way a real peer's
    identity stays stable across a whole relationship rather than being
    reinvented each time."""
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8").strip()
    text = generate_peer_background(patient_opening_text, model)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text


class PeerSupportAgent:
    def __init__(
        self,
        model: ChatModel,
        persona_path: Path = HERE / "config" / "peer_persona.json",
        response_prompt_path: Path = HERE / "prompts" / "response_generator.txt",
        dynamic_backstory: str | None = None,
    ) -> None:
        self.model = model
        self.selector = StrategySelector(model)
        self.persona = json.loads(persona_path.read_text(encoding="utf-8"))
        if dynamic_backstory:
            self.persona["backstory"] = dynamic_backstory
        self.response_prompt = response_prompt_path.read_text(encoding="utf-8")
        self.primary_strategy_history: list[str] = []
        self.used_disclosures: list[str] = []
        self.used_figurative_phrases: list[str] = []
        # Reflexion-style episodic memory: short notes-to-self written at the
        # end of each session, conditioned only on the patient's engagement
        # level (never on self-stigma content) so this loop cannot learn to
        # chase the measured outcome directly.
        self.reflection_notes: list[str] = []
        # Evaluator signal: fraction of "engaged" patient turns per session.
        # Used only to tell the reflection whether this session trended up or
        # down versus the last one — never gates or stops anything, since
        # every session still runs regardless of this score.
        self.engagement_history: list[float] = []

    # Turns that must pass before treatment_modeling can fire again.
    #
    # Enforced here rather than in the selector prompt, which states the same
    # rule and does not follow it: measured over an 8-turn conversation
    # (2026-08-09) it fired 4 times including turns 5 and 6 back to back. Two
    # things go wrong at that rate. A peer who brings up having seen someone
    # every other turn stops reading as a person and starts reading as a
    # referral; and because used_disclosures blocks reusing an episode, the
    # later firings kept the label while drifting off it -- turns 6 and 8 came
    # back talking about opening up to friends, which is not the move and does
    # nothing for what asking a professional would say about you. Spacing it
    # out means each firing can be the real thing.
    _TREATMENT_COOLDOWN = 3

    def _respect_cooldown(self, decision: StrategyDecision) -> StrategyDecision:
        if decision.primary_strategy != "treatment_modeling":
            return decision
        recent = self.primary_strategy_history[-self._TREATMENT_COOLDOWN:]
        if "treatment_modeling" not in recent:
            return decision
        return replace(decision, primary_strategy="attuned_encouragement",
                       secondary_strategy=None)

    def respond(self, dialogue: list[dict], relevant_memory: str = "") -> dict:
        decision = self._respect_cooldown(self.selector.select(
            dialogue,
            prior_primary_strategies=self.primary_strategy_history,
            used_disclosures=self.used_disclosures,
        ))
        recent_own_texts = [t["content"] for t in dialogue if t.get("speaker") == "peer"]
        parsed = self._generate(dialogue, decision, relevant_memory, recent_own_texts)
        response_text = str(parsed.get("peer_response", "")).strip()

        disclosure_id = parsed.get("disclosure_id")
        if disclosure_id and disclosure_id not in self.used_disclosures:
            self.used_disclosures.append(str(disclosure_id))
        self.primary_strategy_history.append(decision.primary_strategy)

        # Dedicated classify -> revise pass, separate from generation, so the
        # revision isn't anchored to the draft's own figurative framing.
        # enforce_plain re-checks its own output instead of trusting one
        # revision to have worked (see patient_agent.py's respond for the
        # observed failure: a single pass sometimes swaps one figurative
        # image for another, e.g. "weight" -> "pressure"). The initial gate
        # uses the voted classifier too -- classify_figurative alone is not
        # deterministic on borderline phrasing (re-checking a stored,
        # already-generated line flipped its own verdict on replay).
        is_figurative, image = classify_figurative_voted(self.model, response_text)
        if is_figurative:
            response_text = enforce_plain(self.model, response_text)
        if image:
            self.used_figurative_phrases.append(image)

        # Catches the reused FRAME that string-similarity (is_near_duplicate)
        # misses -- the same disclosure/support frame reworded fresh almost
        # every turn. Same mechanism patient_agent.py already uses for its own
        # repeated-frame problem.
        if recent_own_texts:
            is_repeat, what = classify_template_repeat(
                self.model, response_text, recent_own_texts)
            if is_repeat and what:
                response_text = revise_off_template(
                    self.model, response_text, what, recent_own_texts)
        # revise_off_template can put a figurative image back. It is asked
        # for "a genuinely DIFFERENT point", with nothing in its prompt about
        # plain language, so reaching for an image is one of the easiest ways
        # for it to sound different -- and it runs AFTER the figurative pass,
        # which is the last thing that would have caught it. Observed
        # 2026-08-09: metaphor density in completed runs concentrated in the
        # later rounds of a session, exactly where enough prior turns exist
        # for the repeat classifier to fire ("a quiet unraveling", "silence
        # can feel like a shield", "a mountain to climb"). Re-checking here
        # closes that hole; enforce_plain is cheap when the text is already
        # clean, since the first voted classification returns immediately.
                response_text = enforce_plain(self.model, response_text)

        return {
            "peer_response": response_text,
            "strategy": {
                "primary": decision.primary_strategy,
                "secondary": decision.secondary_strategy,
                "evidence": decision.evidence,
                "rationale": decision.rationale,
                "support_need": decision.support_need,
                # What this turn was TRYING to move (strategy_selector.
                # STRATEGY_TARGET_AXIS). Recorded per turn so "did the axis a
                # turn aimed at actually move" is answerable from the run file
                # alone, rather than by joining strategy names to a table by
                # hand afterwards. None for `answer`, which is not an
                # intervention move.
                "target_axis": decision.target_axis,
                "target_axes": list(decision.target_axes),
            },
            "disclosure_id": disclosure_id,
        }

    # Restated immediately before generation, not only as one bullet among
    # many in prompts/response_generator.txt. Measured 2026-08-09: with the
    # guidance living only in the system prompt, a turn selected as
    # treatment_modeling came back talking about friends who kept showing up
    # -- the strategy was chosen correctly and then not executed, because
    # "keep the focus on them" and the persona boundaries sit closer to the
    # point of generation and win. Same failure and same fix as the patient
    # agent's anchoring reminder.
    _STRATEGY_RESTATEMENT = {
        "hopeful_disclosure":
            "Share one specific thing from your own past where this was "
            "running you, and one concrete way it is different now. Small and "
            "factual. Do not promise them the same outcome.",
        "treatment_modeling":
            "This turn must say, in your own words, that you went and saw a "
            "professional about this -- a counsellor, a therapist, a doctor. "
            "Name what you were afraid it would say about you before you went, "
            "and what it actually turned out to be instead. Friends, family "
            "and people showing up for you are NOT this; the point is what "
            "happened when you got help from someone whose job it was. Do not "
            "tell them to go and do not imply they are overdue.",
        "attuned_encouragement":
            "Meet what they said from having felt it yourself. You can say you "
            "do not think it settles what they are worth -- but do not argue "
            "them out of the feeling, and leave the next step theirs.",
        "helper_reversal":
            "This turn must be a real question asking THEM something about "
            "their own experience or their own way of getting through a piece "
            "of this -- genuinely wanting to know, not rhetorical. Do not "
            "answer your own question. Do not follow it with reassurance that "
            "makes the question decorative -- the question itself is the "
            "whole move.",
        "strengths_focused_support":
            "This turn must name something SPECIFIC and REAL from what they "
            "just wrote as a form of effort or coping -- posting instead of "
            "sitting on it, noticing the pattern themselves, having gotten "
            "through the day, whatever is actually there. Do not bring in "
            "your own story or experience for this turn -- it is about them, "
            "not you. Do not soften it into generic praise like 'you're doing "
            "great' -- point at the specific thing they described doing.",
        "answer":
            "Answer the question they actually asked, briefly and personally.",
    }

    def _generate(
        self,
        dialogue: list[dict],
        decision: StrategyDecision,
        relevant_memory: str,
        recent_own_texts: list[str] | None = None,
    ) -> dict:
        reflection_block = "\n".join(f"- {n}" for n in self.reflection_notes[-3:]) or "(none yet)"
        user_prompt = f"""[PEER PERSONA]
{json.dumps(self.persona, ensure_ascii=False, indent=2)}

[DIALOGUE FROM PEER PERSPECTIVE]
{format_dialogue(dialogue)}

[RELEVANT PAST MEMORY]
{relevant_memory or "(none)"}

[SELECTED STRATEGY]
primary: {decision.primary_strategy}
secondary: {decision.secondary_strategy}
evidence: {decision.evidence}
{f"support need in their last message: {decision.support_need} -- {decision.evidence!r}" if decision.support_need else ""}

[ALREADY USED DISCLOSURE IDS]
{self.used_disclosures}

[FIGURATIVE LANGUAGE / METAPHORS ALREADY USED THIS CONVERSATION — do not reuse
any of these images or close variants (e.g. if "ghost" is listed, also avoid
"fading", "invisible", "not real" as stand-ins for the same idea)]
{self.used_figurative_phrases or "(none yet)"}

[YOUR OWN NOTES FROM PAST SESSIONS — what tended to keep this person engaged
vs. make them go quiet, based on your own past turns with them. Weigh these,
but do not follow them mechanically or mention them.]
{reflection_block}

{self._STRATEGY_RESTATEMENT.get(decision.primary_strategy, "")}

Generate the next peer response."""
        parsed: dict = {}
        response_text = ""
        for _ in range(MAX_REGENERATION_ATTEMPTS):
            # A malformed/empty generation is a reason to REGENERATE, not to
            # crash the run — over a long multi-session trajectory the model
            # will occasionally emit non-JSON or an empty response.
            try:
                parsed = complete_json(self.model, [
                    {"role": "system", "content": self.response_prompt},
                    {"role": "user", "content": user_prompt},
                ], temperature=1.0)
            except ValueError:
                continue
            response_text = str(parsed.get("peer_response", "")).strip()
            if not response_text:
                continue
            # Two independent duplicate signals: the free-text response
            # itself, and the model re-declaring a disclosure_id it already
            # used earlier (a same-story-different-wording reuse that text
            # similarity alone won't catch, since the retelling is paraphrased).
            reused_disclosure = (
                parsed.get("disclosure_id")
                and str(parsed["disclosure_id"]) in self.used_disclosures
            )
            text_duplicate = recent_own_texts and is_near_duplicate(response_text, recent_own_texts)
            # HOP/COP moves do not always require self-disclosure. If the peer
            # chooses to share a personal story, disclosure_id prevents reuse;
            # otherwise null is fine.
            missing_disclosure_id = False
            if not text_duplicate and not reused_disclosure and not missing_disclosure_id:
                return parsed
        # Retries exhausted (near-duplicate, or never got a usable response).
        # Return the last draft, coercing an empty response to a safe filler
        # rather than raising.
        if not response_text:
            parsed = {"peer_response": "i hear you. that sounds really hard.", "disclosure_id": None}
        return parsed

    def reflect_on_session(self, session_turns: list[dict]) -> None:
        """Reflexion step (Shinn et al. 2023): after a session ends, read back
        what you said and what the person said next, and write a short private
        note-to-self.

        Reads the actual reply text, not a declared stance. The persona used
        to attach a response_mode label (engaged/guarded/deflecting/boundary)
        to every turn, and this method paired that label with the strategy
        that preceded it. The label was undefined in the persona's own prompt
        and came back "guarded" 479 times in 480, so the pairing was the same
        sentence for every strategy. Asked on top of that to say "what to do
        less of", the model invented an attribution out of the constant: notes
        began prescribing "reduce use of lived-experience moves,
        which seem to trigger closed replies" -- useful peer moves -- and
        peer/severe collapsed to pure attunement by
        session 5 while ISMI climbed 2.31 -> 2.82. The label was removed
        2026-08-03; whether someone opened up is a judgement about what they
        wrote, so the note is now written from the exchange itself.

        Still judged ONLY on openness, never on stigma content, so this loop
        cannot learn to chase the measured outcome. And still no prescribing
        away an approach: a quiet week is information about the person, not
        proof that a move was wrong.

        Reflection always runs, win or lose. Nothing here gates or stops a
        trajectory."""
        exchanges: list[str] = []
        last_strategy: str | None = None
        last_peer_text: str = ""
        for turn in session_turns:
            if turn.get("speaker") == "peer":
                last_strategy = (turn.get("metadata") or {}).get("strategy", {}).get("primary")
                last_peer_text = str(turn.get("content", "")).strip()
            elif turn.get("speaker") == "patient" and last_strategy:
                exchanges.append(
                    f"  you ({last_strategy}): {last_peer_text}\n"
                    f"  them: {str(turn.get('content', '')).strip()}"
                )
                last_strategy = None
        if not exchanges:
            return

        system = (
            "You are a peer support agent's private note-to-self after a "
            "session. Below is what you said and what the person said back, "
            "in order. Judge ONLY how open or closed they seemed after each "
            "of your moves -- whether they picked up what you said, added "
            "something of their own, changed the subject, or kept it short "
            "-- never whether their underlying beliefs changed. Write 2-3 "
            "plain sentences on how this person seemed to be doing and what "
            "you noticed about how they responded to you. Do NOT recommend "
            "abandoning or using less of any approach: a quiet session is "
            "information about the person, not proof that something you did "
            "was wrong, and you have too little evidence to rule any "
            "approach out. No diagnosis, no scoring, no mention of "
            "self-stigma."
        )
        user = "\n\n".join(exchanges)
        try:
            note = self.model.complete([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ], temperature=0.3).strip()
        except Exception:
            return
        if note:
            self.reflection_notes.append(note)
