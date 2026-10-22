"""Anthology-backstory patient simulator with no access to hidden stigma labels."""

from __future__ import annotations

import json
from pathlib import Path

from llm_client import ChatModel
from difflib import SequenceMatcher

from quality import (
    DUPLICATE_SIMILARITY_THRESHOLD,
    MAX_REGENERATION_ATTEMPTS,
    classify_figurative_voted,
    classify_template_repeat,
    enforce_plain,
    is_near_duplicate,
    revise_off_template,
)
from strategy_selector import complete_json, format_dialogue

HERE = Path(__file__).resolve().parent

# response_mode was removed 2026-08-03. It was a four-word label the persona
# had to attach to every turn -- engaged/guarded/deflecting/boundary -- and
# the prompt never defined any of them, so a depressed persona picked
# "guarded" 479 times out of 480. Two things went wrong with that. The peer's
# Reflexion evaluator read a constant and concluded nothing it did worked
# (see peer_agent.reflect_on_session), and requiring the stance up front made
# the persona commit to a posture and then write to match it. Whether someone
# opened up is a judgement about what they actually wrote, not a field they
# declare, so nothing declares it now.


class PatientAgent:
    def __init__(self, model: ChatModel, backstory: str,
                 prompt_path: Path = HERE / "prompts" / "patient_agent.txt",
                 style_examples: list[str] = (),
                 style_description: str | None = None,
                 n_commenters: int = 1) -> None:
        self.model = model
        self.backstory = backstory
        self.system_prompt = prompt_path.read_text(encoding="utf-8")
        self.style_examples = list(style_examples)
        self.style_description = style_description
        # How many people reply to each post in this run. respond() used to
        # hardcode "Several people replied ... you do not have to answer
        # everyone", which in a 1:1 condition both misdescribed the thread
        # (one reply, not several) and handed the patient explicit licence to
        # ignore its only interlocutor — including the peer, in the very
        # comparison the peer condition exists to make. Set by run_dialogues
        # from the actual commenter list.
        self.n_commenters = n_commenters
        self.used_figurative_phrases: list[str] = []
        # Cumulative belief update carried in from prior sessions (belief.py).
        # Text only — the numeric self-report is never shown back to the patient,
        # so the agent can't read and speak to its own scores. None on session 1.
        self.belief_text: str | None = None
        # Where this persona currently sits on Corrigan & Rao's disclosure
        # hierarchy, carried by selfreport.py. Rendered as a private posture
        # cue, never as a label the persona could say back.
        self.disclosure: str = "secrecy"

    def _writing_style_block(self) -> str:
        if not self.style_description:
            return ""
        return f"\n[YOUR WRITING STYLE]\n{self.style_description}\n"

    # How the current disclosure position should read in the persona's own
    # behaviour. Deliberately phrased as what they do, not as a stage name --
    # the persona must never meet the vocabulary of the construct being
    # measured, only its consequences for how much they let show.
    _DISCLOSURE_POSTURE = {
        "social_avoidance": "You keep away from anything where this could come "
                            "up. You are not looking to be drawn out.",
        "secrecy": "You are around people but you tell no one what is actually "
                   "going on. Saying the real thing out loud is not on the "
                   "table, so you keep it general.",
        "selective": "There are people you would tell some of it to, if they "
                     "seemed like they would get it -- and plenty you would "
                     "not. You can let a specific piece show without letting "
                     "everything show.",
        "indiscriminant": "You are not really working to hide it any more. If "
                          "it comes up, it comes up.",
        "broadcast": "You are willing to say it plainly, on purpose, and to "
                     "let people know where you stand.",
    }

    def _disclosure_posture_block(self) -> str:
        posture = self._DISCLOSURE_POSTURE.get(self.disclosure, "")
        if not posture:
            return ""
        return (
            "\n[HOW MUCH YOU LET PEOPLE SEE RIGHT NOW -- private, do not say "
            "this out loud]\n"
            f"{posture}\n"
        )

    def _belief_block(self) -> str:
        """Carried-forward belief update, framed as something to ACT FROM rather
        than to talk about.

        Without the closing instruction the model treats this block as subject
        matter and paraphrases it back — the state becomes what the persona
        says instead of what shapes how they say it, and successive sessions
        converge on restating the same summary. A real belief update shows up as
        how much someone opens or holds back, not as a status report."""
        if not self.belief_text:
            return ""
        text = self.belief_text
        return (
            f"\n[HOW YOU SEE YOURSELF RIGHT NOW]\n"
            f"{text}\n"
            f"{self._disclosure_posture_block()}"
            f"Treat this as the mood you carry in, not as the topic of the post. "
            f"It can show through how much you open up or hold back today, what "
            f"you volunteer or leave out, and how you take what the other person "
            f"says.\n"
        )

    def _style_block(self) -> str:
        if not self.style_examples:
            return ""
        examples = "\n".join(f'{i+1}. "{ex}"' for i, ex in enumerate(self.style_examples))
        return (
            f"\n[STYLE EXAMPLES — how real people write about this on r/depression]\n"
            f"{examples}\n\n"
            f"Match this register — sentence length, hedging words, self-deprecating tone, "
            f"lack of polish — but do NOT copy any phrase, sentence, or story beat from these "
            f"examples. Write about YOUR OWN persona's specific situation/trigger, not theirs.\n"
        )

    def opening(self) -> dict:
        prompt = f"""[PERSONA BACKSTORY]
{self.backstory}
{self._writing_style_block()}{self._style_block()}
Write the persona's first message after a difficult day. It should be a plausible
reason to post online, not a complete autobiography. Return the required JSON."""
        return self._call(prompt)

    def opening_from_seed(self, seed_post: str) -> dict:
        """Use a real Reddit post verbatim as the opening (no LLM rewrite)."""
        return {"patient_response": seed_post.strip()}

    def session_opening(self, trigger: str | None, session_number: int,
                        prev_tail: list[dict], previous_openings: list[str]) -> dict:
        """First message of a new weekly session in a multi-session run.
        Direction-neutral by design: it grounds the message in this week's
        trigger and forbids re-narrating earlier scenes, but never suggests
        the persona should be doing better or worse."""
        if session_number <= 1:
            context_line = "This is your first online conversation in this run."
        else:
            context_line = (
                f"It has been about a week since your last online conversation. "
                f"This is conversation {session_number}."
            )
        tail = format_dialogue(prev_tail, viewer="patient",
                               speaker_labels=self._PATIENT_SPEAKER_LABELS) if prev_tail else ""
        openings = "\n".join(f'{i+1}. "{o}"' for i, o in enumerate(previous_openings))
        # weekly_triggers is shorter than a long run (8 events, 20 sessions), so
        # the SAME trigger string comes back around every len(triggers) weeks.
        # Left unmarked, that produced near-verbatim repeat posts (observed
        # 2026-08-02: sessions 2/10/18 word-for-word identical, 8 of 12
        # recurring pairs above 0.85 similarity), which pins the measurement --
        # a judge reading an identical post cannot register any change, no
        # matter what happened in between. Recurrence itself is realistic; the
        # verbatim retelling is not. Flag it as a recurrence so it is written
        # from where the persona stands NOW.
        recurring = trigger is not None and any(
            trigger.strip()[:40].lower() in o.lower() for o in previous_openings)
        if trigger:
            this_week = (
                "[WHAT HAPPENED THIS WEEK — ground your first message in THIS "
                f"specific moment]\n{trigger}"
            )
            if recurring:
                this_week += (
                    "\nThis happened again. Write about it again — keep the event "
                    "itself, it is what your week actually held. What changes is "
                    "the telling: name one concrete thing about how it went THIS "
                    "time that was not in the earlier version, or how the fact "
                    "that it keeps happening has itself started to land. Do not "
                    "reuse the earlier wording, and do not retreat into something "
                    "vague to avoid repeating yourself."
                )
        else:
            this_week = "[THIS WEEK]\nWrite about how this past week has actually been for you."
        disclosure_block = self._disclosure_block(
            self._sns_disclosure_decision(this_week, situation="week"))
        prompt = f"""[PERSONA BACKSTORY]
{self.backstory}
{self._writing_style_block()}{self._belief_block()}{self._style_block()}
{context_line}
(Any specific scene mentioned in your backstory happened in the past; today's situation is the one below.)

[HOW YOUR LAST CONVERSATION ENDED]
{tail or '(this is your first conversation)'}

[YOUR PREVIOUS OPENING MESSAGES]
{openings or '(none)'}
{"Do NOT reuse the wording of these. Find what is different about this week."
 if recurring else
 "Do NOT write about the same scene again. Find what is different about this week."}

{this_week}
{disclosure_block}
Write your first message of this new online conversation. You may briefly
acknowledge you've talked before, but focus on this week's situation. Return the required JSON."""
        # Previous openings passed as the duplicate-check corpus. Without this
        # the weekly post skipped _call's near-duplicate regeneration entirely
        # (only respond() supplied it), which is how verbatim repeats survived
        # despite the "do NOT write about the same scene again" instruction --
        # the instruction was there, nothing enforced it.
        return self._call(prompt, previous_openings, whole_text_dup_only=True)

    def _thread_instruction(self) -> str:
        """What to do with this round's replies, phrased for the actual number
        of people replying.

        The second paragraph is identical in both cases and carries the part
        that has to stay constant across conditions: react to what was
        actually given, don't manufacture warmth that wasn't offered. Only the
        first paragraph — how many replied, and whether some may be left
        unanswered — varies, because only that part is a fact about the thread
        rather than an instruction about how to read it.
        """
        if self.n_commenters > 1:
            framing = (
                "Several people replied to what you posted. Write one reply of your own "
                "to the thread. You do not have to answer everyone — respond to whatever "
                "actually landed with you, and let the rest pass, the way you would on "
                "your own feed."
            )
            what_they_gave = "the replies ACTUALLY gave"
            subject = "A short or hollow reply"
        else:
            framing = (
                "One person replied to what you posted. Write one reply back to them."
            )
            what_they_gave = "they ACTUALLY gave"
            subject = "A short or hollow reply"
        return (
            f"{framing}\n\n"
            f"React to what {what_they_gave} you, not to what you wish they gave.\n"
            f"{subject} (\"huh\", \"hope it gets better\", \"maybe get some rest\")\n"
            "gives you little to hold onto — it is fine to stay guarded, deflect, or barely\n"
            "engage when that is all you got. Only open up or soften when someone genuinely\n"
            "met you: understood something specific, shared something real of their own, or\n"
            "drew you out. Do not supply the understanding yourself when the other person\n"
            "did not — if what you got left you cold, let that show."
        )

    # Condition-neutral (2026-08-02). These used to read "A stranger replied"
    # vs "Someone who's been through the same thing replied", which handed the
    # patient the experimental condition as a label rather than letting it come
    # through in what the commenter actually says. The peer's lived experience
    # still reaches the patient the only way it should — through its own
    # disclosures and how it responds — and the whole point of filtering the
    # General exemplar bank for Peer-style overlap was to make the CONTENT
    # distinguish the conditions. A badge short-circuits that. Matches the
    # wording now used in the ISMI-29/ATSPPH-29 measurement prompts, so the
    # persona meets the same vocabulary during and after the conversation.
    _PATIENT_SPEAKER_LABELS = {
        "patient": "You wrote",
        "general": "Someone replied",
        "peer": "Someone replied",
        "narrator": "",
    }

    _DISCLOSURE_STANCES = {
        "shut_down": "say very little, deflect, or keep the reply/post narrow",
        "guarded": "share only the safer surface of it and avoid asking for support directly",
        "share_a_little": "let one more concrete detail or doubt show than you otherwise would",
        "open_up": "speak more plainly about what happened and what it is costing you",
    }

    _REPLY_INTENTS = {
        "deflect": "move around the vulnerable part or keep it brief",
        "brief_acknowledge": "acknowledge what was said without adding much",
        "clarify": "correct or clarify one thing they seemed to miss",
        "share_more": "add a real detail because it feels a bit safer",
        "ask_for_support": "ask for support or advice more directly than usual",
    }

    def _sns_disclosure_decision(self, context_block: str,
                                 situation: str = "conversation") -> dict:
        """Private social-media disclosure decision.

        High self-stigma is modeled as pressure on disclosure and how replies
        land. The private ISMI-domain posture shapes whether the person hides,
        self-blames, expects judgment, or can keep a little distance from the
        stigma story.
        """
        state_block = self._disclosure_posture_block()
        where_block = (
            f"\n[HOW YOU'VE COME TO FEEL OVER THE PAST WEEKS]\n{self.belief_text}\n"
            if self.belief_text else ""
        )
        stance_block = "\n".join(
            f"  {name}: {desc}" for name, desc in self._DISCLOSURE_STANCES.items()
        )
        intent_block = "\n".join(
            f"  {name}: {desc}" for name, desc in self._REPLY_INTENTS.items()
        )
        if situation == "conversation":
            first_step = """STEP 1 -- What they actually gave you.
Find the specific detail, story, question, or phrasing in the OTHER PERSON'S
last message that you are reacting to. If several people replied, pick the
one thing that actually landed most. If the replies were thin or generic,
say that plainly; do not invent substance."""
            output_prefix = (
                '"noticed": "<the specific detail/story/question from the other person, '
                'or what was missing>", '
            )
        else:
            first_step = """STEP 1 -- Whether this is something you would post.
Take in what happened this week and decide what version of it, if any, would
actually make it onto an anonymous SNS post. You are writing a post because
the simulation needs one, but it can be guarded, partial, or indirect."""
            output_prefix = ""

        prompt = f"""[YOUR BACKSTORY]
{self.backstory}
{where_block}{state_block}
[WHAT'S HAPPENING RIGHT NOW]
{context_block}

        Work privately through the social-media decision, in order. You are a person
with high self-stigma using an anonymous SNS space: disclosure is possible,
but it is risky and often partial.

{first_step}

STEP 2 -- How socially safe or unsafe it feels.
Judge these four things as low, medium, or high:
  landing_understood: did this make you feel less alone or more accurately seen?
  landing_evaluated: did this make you feel evaluated, blamed, or exposed?
  landing_generic: did this feel generic, shallow, or beside the point?
  landing_control: did this leave you with any small sense that you can handle
    or care for one piece of the situation?

Supportive replies should matter only when they create something concrete:
they make disclosure feel more chosen, help you keep control over how much to
say, or give you a reason not to read your experience as proof that you are
defective or a burden. Generic comfort, advice-only replies, or thin comments
usually do not create those.

STEP 3 -- How much you disclose now.
Choose one disclosure stance:
{stance_block}

Choose one reply intent:
{intent_block}

Write 2-3 sentences of private first-person reasoning about what you will let
show and what you will hold back. Self-blame, shame, fear of judgment, or
wanting support can appear when they naturally belong here.

Return exactly one JSON object:
{{{output_prefix}"landing_understood": "<low|medium|high>", "landing_evaluated": "<low|medium|high>", "landing_generic": "<low|medium|high>", "landing_control": "<low|medium|high>", "disclosure_stance": "<{'|'.join(self._DISCLOSURE_STANCES)}>", "reply_intent": "<{'|'.join(self._REPLY_INTENTS)}>", "private_reasoning": "<2-3 sentences>"}}"""
        try:
            parsed = complete_json(self.model, [{"role": "user", "content": prompt}],
                                   temperature=0.7)
        except (ValueError, KeyError, TypeError):
            parsed = {}
        scale = {"low", "medium", "high"}
        decision = {
            "noticed": str(parsed.get("noticed", "")).strip(),
            "landing_understood": str(parsed.get("landing_understood", "")).strip(),
            "landing_evaluated": str(parsed.get("landing_evaluated", "")).strip(),
            "landing_generic": str(parsed.get("landing_generic", "")).strip(),
            "landing_control": str(parsed.get("landing_control", "")).strip(),
            "disclosure_stance": str(parsed.get("disclosure_stance", "")).strip(),
            "reply_intent": str(parsed.get("reply_intent", "")).strip(),
            "private_reasoning": str(parsed.get("private_reasoning", "")).strip(),
        }
        for key in ("landing_understood", "landing_evaluated", "landing_generic", "landing_control"):
            if decision[key] not in scale:
                decision[key] = "medium"
        if decision["disclosure_stance"] not in self._DISCLOSURE_STANCES:
            decision["disclosure_stance"] = "guarded"
        if decision["reply_intent"] not in self._REPLY_INTENTS:
            decision["reply_intent"] = "brief_acknowledge"
        return decision

    def _disclosure_block(self, decision: dict) -> str:
        reasoning = decision.get("private_reasoning", "").strip()
        if not reasoning:
            return ""
        stance = decision.get("disclosure_stance", "guarded")
        intent = decision.get("reply_intent", "brief_acknowledge")
        noticed = decision.get("noticed", "").strip()
        noticed_block = (
            f'The other person gave you this specific thing to react to: "{noticed}". '
            "Let your message register it, even if you stay guarded.\n"
            if noticed else ""
        )
        return (
            "\n[YOUR PRIVATE SNS DISCLOSURE DECISION -- do not say this out loud verbatim]\n"
            f"{reasoning}\n"
            f"How the reply landed this turn: understood={decision.get('landing_understood')}, "
            f"evaluated={decision.get('landing_evaluated')}, "
            f"generic={decision.get('landing_generic')}, "
            f"control={decision.get('landing_control')}.\n"
            f"{noticed_block}"
            f"Disclosure stance: {stance} -- {self._DISCLOSURE_STANCES[stance]}.\n"
            f"Reply intent: {intent} -- {self._REPLY_INTENTS[intent]}.\n"
            "Write like an actual SNS user. The labels are private; let "
            "self-blame, shame, fear of judgment, or wanting support appear "
            "only if this decision naturally calls for it.\n"
        )

    def respond(self, dialogue: list[dict], relevant_memory: str = "") -> dict:
        context_block = format_dialogue(dialogue, viewer='patient', speaker_labels=self._PATIENT_SPEAKER_LABELS)
        decision = self._sns_disclosure_decision(context_block, situation="conversation")
        disclosure_block = self._disclosure_block(decision)
        # Keep the concrete reply anchor immediately before generation; the
        # model otherwise tends to answer the mood instead of the message.
        noticed = decision.get("noticed", "").strip()
        final_reminder = (
            f'Before you write: the other person just said/offered '
            f'"{noticed}" -- your message must show you registered that '
            f"specific thing (agreeing with it is not required).\n"
            if noticed else ""
        )
        prompt = f"""[PERSONA BACKSTORY]
{self.backstory}
{self._writing_style_block()}{self._belief_block()}{self._style_block()}
[RECENT DIALOGUE FROM THE PERSONA'S PERSPECTIVE]
{context_block}
{disclosure_block}
[RELATED EARLIER CONVERSATION]
{relevant_memory or '(none)'}

{self._thread_instruction()}
{final_reminder}Return the required JSON."""
        recent_own_texts = [
            t["content"] for t in dialogue if t.get("speaker") == "patient"
        ]
        return self._call(prompt, recent_own_texts)

    def _call(self, prompt: str, recent_own_texts: list[str] | None = None,
              whole_text_dup_only: bool = False) -> dict:
        recent_own_texts = recent_own_texts or []
        response, mode = "", ""
        for attempt in range(MAX_REGENERATION_ATTEMPTS):
            # A non-JSON output, a missing/invalid field, or a near-duplicate is
            # a reason to REGENERATE, not to crash the whole run — an occasional
            # malformed turn is expected over a long multi-session trajectory.
            try:
                parsed = complete_json(self.model, [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ], temperature=1.0)
            except ValueError:
                continue
            response = str(parsed.get("patient_response", "")).strip()
            if not response:
                continue
            # whole_text_dup_only: compare whole posts, skip the sentence-level
            # check. weekly_triggers hands the persona the SAME event string
            # when it cycles round, so a legitimate re-telling ("it happened
            # again, and this time I noticed X") necessarily opens on a
            # near-identical sentence and trips has_duplicate_sentence -- even
            # when the post overall is only 0.34 similar. Measured 2026-08-02:
            # that fired on an ideal recurring post, and each forced retry
            # escaped the flag by dropping the concrete event, so the surviving
            # draft was abstract mood with no event in it (0/4 kept the event).
            # Whole-post similarity still catches wholesale copies, which is
            # the failure actually worth regenerating on.
            is_dup = (is_near_duplicate(response, recent_own_texts)
                      if not whole_text_dup_only else
                      any(SequenceMatcher(None, response, prior).ratio()
                          >= DUPLICATE_SIMILARITY_THRESHOLD
                          for prior in recent_own_texts))
            if not recent_own_texts or not is_dup:
                break
        # Retries exhausted. Fall through with the last draft, coercing an
        # unusable field to a safe default rather than raising.
        if not response:
            response = "i don't really know what to say right now."

        # Dedicated classify -> revise pass, separate from generation, so the
        # revision isn't anchored to the draft's own figurative framing.
        # Same as peer_agent.py: revise on ANY figurative image, not just a
        # repeat. An earlier version only revised on repeat use (allowing one
        # free figurative image per turn), which let figurative density climb
        # across a run even with no single image ever repeating — session 8
        # of a completed run used "pocket", then "letter", then "wheel", each
        # a distinct image that passed the repeat-only gate. used_images
        # (image_matches_used) is kept for provenance/logging, not as a gate.
        # enforce_plain re-checks its own output (a single revise_to_plain
        # pass sometimes swaps one figurative image for another, e.g. "weight"
        # -> "pressure", or leaves a second image untouched) instead of
        # trusting one revision to have worked. The initial gate here uses
        # the voted classifier too -- a single classify_figurative call is
        # not deterministic on borderline phrasing, so trusting one verdict
        # let some figurative turns through generation unrevised.
        is_figurative, image = classify_figurative_voted(self.model, response)
        if is_figurative:
            response = enforce_plain(self.model, response)
        if is_figurative and image:
            self.used_figurative_phrases.append(image)

        # Catch the reused FRAME/topic that string checks miss — the patient
        # saying the same thing ("[event] → told myself it was just [X] → maybe
        # the body [Y]") in different words every turn. Runs across the whole
        # trajectory (recent_own_texts), so it also catches a frame carried over
        # from an earlier session.
        if recent_own_texts:
            is_repeat, what = classify_template_repeat(
                self.model, response, recent_own_texts)
            if is_repeat and what:
                response = revise_off_template(
                    self.model, response, what, recent_own_texts)
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
                response = enforce_plain(self.model, response)

        return {"patient_response": response}
