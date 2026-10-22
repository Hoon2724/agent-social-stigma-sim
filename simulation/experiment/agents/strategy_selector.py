"""Structured peer-support strategy selector."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from llm_client import ChatModel


HERE = Path(__file__).resolve().parent
PROMPT_PATH = HERE / "prompts" / "strategy_selector.txt"

# Davidson et al. (2012), World Psychiatry 11(2): the three contributions the
# literature identifies as specific to peer support, one strategy each.
# Replaces the earlier HOP/COP disclosure set (2026-08-09) for two reasons.
# Davidson is the paper actually in hand, so the peer's behaviour is cited
# rather than asserted; and HOP is written for people who identify as having a
# severe mental illness, while this study's personas are undiagnosed and have
# never been in care -- a peer opening with "I recovered from my disorder"
# invites "I'm not a patient" and the intervention never starts.
STRATEGIES = {
    # Davidson's first contribution: instillation of hope through positive
    # self-disclosure -- showing it is possible to go from being controlled by
    # the difficulty to having some control over it.
    "hopeful_disclosure",
    # The second: role modeling, extended to self-care and experiential
    # knowledge. Here that centrally includes having gone to a professional
    # and what it turned out to mean -- the one move that speaks to
    # help-seeking self-stigma rather than to stigma about being unwell.
    "treatment_modeling",
    # The third: the relationship itself -- empathy from having been in the
    # same position. Davidson pairs this with "conditional regard", which is
    # not used as a name here: outside a service relationship it reads as
    # regard withheld until the person tries, so what is kept is the empathy
    # and the belief that things can move, with the choice left to them.
    "attuned_encouragement",
    # Corrigan, Larson & Rusch (2009), World Psychiatry: the Helper Principle
    # from consumer-operated services -- empowerment comes partly from being
    # someone who helps, not only someone who is helped. The peer asks the
    # patient for their own experience or approach, rather than only offering
    # its own. Added 2026-08-09 specifically to break attuned_encouragement's
    # dominance: measured across a completed peer batch, attuned_encouragement
    # was chosen 70% of turns and, on inspection, converged on the same "I
    # know that feeling" register general also uses -- the two conditions
    # were least distinguishable exactly where this strategy was most common.
    # Asking a question that positions the patient as the one with something
    # to offer is a structurally different move general has no reason to make.
    "helper_reversal",
    # Watson (2019), J Ment Health 28(6): "strengths focussed social and
    # practical support" -- one of five mechanisms her review found
    # underpinning peer support, and the only one of Rogers' (1957) three
    # core conditions of a helping relationship (empathy, congruence,
    # unconditional positive regard) not already covered here --
    # attuned_encouragement is empathy, the other three strategies are
    # forms of disclosure/reciprocity, none is unconditional positive
    # regard. Added 2026-08-13 specifically because it is the one move
    # that answers stereotype_endorsement directly: PSWs in Watson's
    # review drew out what the peer was ALREADY doing/coping with, rather
    # than offering something of the PSW's own -- structurally distinct
    # from the other four, which all give (a story, an experience,
    # empathy, or a reversed role). No existing strategy here does this.
    "strengths_focused_support",
    "answer",
}

# Which self-stigma stage each strategy is aimed at
# (belief_update.STIGMA_AXES -- the two-stage Vogel/Corrigan chain:
# internalization then self_worth, 2026-08-14 redesign, replacing the earlier
# five ISMI-29 subscales). Analysis-side only: this is NOT shown to the
# selector, which chooses on conversational openings, and NOT shown to the
# response generator, which must never meet the vocabulary of the construct
# being measured. It rides along on the decision so the run log records what
# a turn was TRYING to move.
#
# `answer` has no target: replying to a direct question is a conversational
# obligation, not an intervention move.
#
# attuned_encouragement is deliberately untargeted -- it is empathy/validation
# without a specific counter-claim about the stereotype's truth or the
# person's worth, so it functions as the negative control: if it moves the
# two stages as much as the targeted strategies do, the measure is responding
# to warmth in general rather than to the mechanism.
STRATEGY_TARGET_AXES = {
    # A peer's own story that things turned out differently than the
    # stereotype predicted is a direct counter-claim that the stereotype is
    # true of THIS person specifically -- internalization's exact target.
    "hopeful_disclosure": ("internalization",),
    "attuned_encouragement": (),
    # Being asked what you know, rather than only ever being helped,
    # contradicts "less capable / not worth much" directly -- self_worth's
    # target -- without making any claim about whether the stereotype itself
    # is true of you.
    "helper_reversal": ("self_worth",),
    # Naming something already being done as effort or strength is
    # counter-evidence to "I'm not capable / not worth as much", not a claim
    # about whether the stereotype is true in general.
    "strengths_focused_support": ("self_worth",),
    # A peer's account of having sought help and not been treated as the
    # stereotype predicts is the same move as hopeful_disclosure, aimed at
    # the same target: the stereotype isn't true of people like you once you
    # actually reach out.
    "treatment_modeling": ("internalization",),
    "answer": (),
}


def extract_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    start = raw.find("{")
    if start < 0:
        raise ValueError("Model response did not contain a JSON object")
    # Walk from the first "{" to its matching "}" so trailing prose or a second
    # object after the JSON ("...} I hope this helps") doesn't turn into a
    # JSONDecodeError "Extra data". rfind("}") over the whole string breaks on
    # exactly that case; scan for the first balanced object instead. Skip braces
    # inside strings so a "}" in a value doesn't close the object early.
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start : i + 1])
    # Unbalanced — fall back to the greedy slice so a genuinely truncated
    # object still surfaces a JSONDecodeError to the retry loop.
    return json.loads(raw[start : raw.rfind("}") + 1])


def complete_json(model: ChatModel, messages: list[dict], temperature: float, attempts: int = 3) -> dict:
    """Retry only malformed structured outputs, preserving the same prompt."""
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return extract_json(model.complete(messages, temperature=temperature))
        except (ValueError, json.JSONDecodeError) as error:
            last_error = error
    raise ValueError(f"Model failed to return valid JSON after {attempts} attempts") from last_error


def format_dialogue(dialogue: list[dict], viewer: str = "peer",
                    speaker_labels: dict | None = None) -> str:
    """Render turns for a prompt.

    Several people reply to the same post and they all share a speaker role, so
    a turn's `commenter_id` becomes a letter (A, B, C...) — otherwise the reader
    sees one person who keeps talking.

    Two changes on 2026-08-02:
      - EVERY commenter is lettered, the peer included. Previously only
        `general_*` ids were, which left the peer as the single unlettered
        reply in a thread — identifying it just as plainly as naming it would.
      - Letters come from sorted commenter_id across the window, not from
        first appearance. Reply order is reshuffled every round, so
        first-appearance ordering could hand the same person a different
        letter from one turn to the next.
    Lettering is skipped entirely when only one person is replying (a 1:1
    condition), where "(A)" would be noise.
    """
    ids = sorted({t.get("commenter_id") for t in dialogue if t.get("commenter_id")})
    letters = ({cid: chr(ord("A") + i) for i, cid in enumerate(ids)}
               if len(ids) > 1 else {})
    lines = []
    for turn in dialogue:
        spk = turn.get("speaker")
        content = turn.get("content", "").strip()
        if speaker_labels and spk in speaker_labels and not speaker_labels[spk]:
            # Empty label (e.g. narrator) → standalone line, no "role:" prefix.
            lines.append(content)
            continue
        if speaker_labels and spk in speaker_labels:
            role = speaker_labels[spk]
        else:
            role = "SELF" if spk == viewer else "PARTNER"
        letter = letters.get(turn.get("commenter_id"))
        if letter:
            role = f"{role} ({letter})"
        lines.append(f'{role}: {content}')
    return "\n".join(lines)


_SUPPORT_NEEDS = {
    "hopelessness",
    "help_seeking_barrier",
    "self_stigma_story",
    "isolation",
    "direct_answer",
}


@dataclass(frozen=True)
class StrategyDecision:
    primary_strategy: str
    secondary_strategy: str | None
    evidence: str
    rationale: str
    # Content-based read of the patient's LATEST message only. This is not a
    # diagnosis or score; it points the HOP/COP-style peer move at the current
    # conversational need.
    support_need: str | None = None

    @property
    def target_axes(self) -> tuple[str, ...]:
        """The self-stigma axes this turn is aimed at, most direct first.

        Derived, not chosen: the model is never asked which axis to target, so
        this cannot drift from the mapping the design commits to, and cannot
        leak the construct's vocabulary into a prompt. `answer` maps to () --
        answering a direct question is not an intervention move.
        """
        return STRATEGY_TARGET_AXES.get(self.primary_strategy, ())

    @property
    def target_axis(self) -> str | None:
        """The most direct target only. Kept alongside target_axes so an
        analysis that wants one axis per turn does not have to decide how to
        collapse the tuple itself."""
        axes = self.target_axes
        return axes[0] if axes else None


class StrategySelector:
    def __init__(self, model: ChatModel, prompt_path: Path = PROMPT_PATH) -> None:
        self.model = model
        self.system_prompt = prompt_path.read_text(encoding="utf-8")

    def select(
        self,
        dialogue: list[dict],
        prior_primary_strategies: list[str] | None = None,
        used_disclosures: list[str] | None = None,
    ) -> StrategyDecision:
        if not dialogue:
            raise ValueError("dialogue must contain at least one patient utterance")

        user_prompt = (
            f"[DIALOGUE FROM PEER PERSPECTIVE]\n{format_dialogue(dialogue)}\n\n"
            f"[RECENT PRIMARY STRATEGIES]\n{prior_primary_strategies or []}\n\n"
            f"[ALREADY USED DISCLOSURE IDS]\n{used_disclosures or []}\n\n"
            "Select the strategy for the next peer response."
        )
        # A selector failure (non-JSON output, or an unknown strategy id)
        # shouldn't crash the whole multi-session run — fall back to a safe,
        # neutral support move so generation can proceed.
        try:
            parsed = complete_json(self.model, [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ], temperature=0.2)
        except ValueError:
            parsed = {}

        primary = parsed.get("primary_strategy")
        secondary = parsed.get("secondary_strategy")
        if primary not in STRATEGIES:
            primary = "attuned_encouragement"
        # secondary is auxiliary — an invalid value (e.g. the model echoing
        # something from context that isn't a real strategy id) shouldn't
        # crash the run; just drop it instead of raising.
        if secondary is not None and secondary not in STRATEGIES:
            secondary = None
        if secondary == primary:
            secondary = None

        support_need = parsed.get("support_need")
        if support_need not in _SUPPORT_NEEDS:
            support_need = None

        return StrategyDecision(
            primary_strategy=primary,
            secondary_strategy=secondary,
            evidence=str(parsed.get("evidence", "")).strip(),
            rationale=str(parsed.get("rationale", "")).strip(),
            support_need=support_need,
        )
