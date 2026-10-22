"""Dialogue-quality guards: suppress repetition and figurative drift.

The goal of this simulation is natural, SNS-like conversation. Two failure
modes degrade it as a dialogue grows long — both documented as "persona /
emotional drift" in the literature, and both driven by the interaction itself
rather than a loss of persona representation:
  - repetition: an agent reuses its own prior wording (whole response, a single
    stock sentence, or the same figurative image under new spelling), a known
    multi-turn attention-collapse effect that decoding penalties do not fix
    because each API call is independent.
  - figurative drift: responses lean on metaphors ("heavy", "a ghost",
    "fading") instead of plain description, which reads as empty.

Rationale for decomposing into narrow single-job calls (rather than relying
solely on "don't repeat / don't use metaphors" inside the generation prompt):
embedded instructions compete with the primary generation task and get
missed, so an instruction alone is not sufficient. Detection is a separate
pass — a cheap text-similarity check plus a dedicated figurative-language
classify -> revise pair, run on every message regardless of whether the
image has appeared before. This mirrors the generate -> critique -> revise
pattern (Self-Refine-style) and is meant to run ALONGSIDE the generation
prompt's own plain-language instruction, not instead of it — the prompt
instruction catches most cases cheaply, this pass catches what leaks through.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from llm_client import ChatModel
from strategy_selector import complete_json

# Above this similarity to any of an agent's own recent turns, a response is
# treated as a near-duplicate and should be regenerated. Root cause: long-
# context attention collapse onto the model's own prior output, observed even
# when the preceding turn from the other party differs each time and the
# system prompt already says "never repeat phrasing" — a known LLM multi-turn
# failure mode, not a prompt-construction bug (verified the raw prompt sent
# to the API differs turn to turn). Shared by both patient_agent.py and
# peer_agent.py.
DUPLICATE_SIMILARITY_THRESHOLD = 0.8
MAX_REGENERATION_ATTEMPTS = 3

# Whole-response similarity (above) misses a distinct failure mode: a short
# stock sentence ("It's okay to not be okay...") pasted verbatim while the
# rest of the response is freshly worded, which drags the whole-string ratio
# down even though a full clause was reused. Caught separately by comparing
# individual sentences — a length floor avoids flagging generic short
# sentences ("I see.", "That makes sense.") that are fine to reuse.
SENTENCE_DUPLICATE_THRESHOLD = 0.9
SENTENCE_MIN_LENGTH = 40
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])[’”\"')]*\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if len(s.strip()) >= SENTENCE_MIN_LENGTH]


def has_duplicate_sentence(text: str, recent_own_texts: list[str]) -> bool:
    sentences = _split_sentences(text)
    if not sentences:
        return False
    prior_sentences = [s for prior in recent_own_texts for s in _split_sentences(prior)]
    return any(
        SequenceMatcher(None, sent, prior).ratio() >= SENTENCE_DUPLICATE_THRESHOLD
        for sent in sentences
        for prior in prior_sentences
    )


def is_near_duplicate(text: str, recent_own_texts: list[str]) -> bool:
    if any(
        SequenceMatcher(None, text, prior).ratio() >= DUPLICATE_SIMILARITY_THRESHOLD
        for prior in recent_own_texts
    ):
        return True
    return has_duplicate_sentence(text, recent_own_texts)


_FIGURATIVE_SYSTEM = (
    "You are a single-purpose classifier. Read ONE message and decide whether "
    "it relies on a figurative image/metaphor rather than plain, literal "
    "description. Two families both count, not just the first:\n"
    "  - sensory/body images standing in for a feeling (e.g. 'heavy', 'a "
    "weight', 'a ghost', 'fading', 'a fog', 'drowning', 'a mask', 'linger')\n"
    "  - narrative or personifying analogies that cast an abstract state as a "
    "story, object, or agent (e.g. 'silence is a language I haven't learned "
    "to speak', 'a story I haven't written yet', 'turning the page', 'read "
    "between the lines', 'the way the dust settles in the corners', 'a quiet "
    "act of defiance') — these read as literary rather than conversational, "
    "even with no single flagged word, and count as figurative too.\n"
    "If the sentence would sound at home in a poem or a novel's narration "
    "rather than an ordinary text message, treat it as figurative even when "
    "it does not match either example list exactly. Do not generate a "
    "response — only classify and, if figurative, name the core image or "
    "analogy in a few lowercase words.\n\n"
    "Output exactly one JSON object:\n"
    '{"is_figurative": true|false, "image": "<core image word(s) or empty string">}'
)


def image_matches_used(image: str, used_images: list[str]) -> bool:
    """Whether `image` is the same figurative image as one already used.

    classify_figurative's label isn't a fixed vocabulary — the same image
    comes back worded differently call to call ("fade" vs "fading" vs
    "broken, fades"), so exact-string membership silently lets the same
    metaphor through under a new spelling. Compare on word stems (first 4
    chars, cheap stand-in for a real stemmer) with any-token overlap instead.
    """
    def words(s: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) >= 4}

    def share_stem(a: str, b: str) -> bool:
        shorter, longer = sorted((a, b), key=len)
        return longer.startswith(shorter[: max(3, len(shorter) - 2)])

    image_words = words(image)
    if not image_words:
        return False
    for used in used_images:
        for w1 in image_words:
            for w2 in words(used):
                if share_stem(w1, w2):
                    return True
    return False


_CRISIS_ADJACENT_SYSTEM = (
    "You are a single-purpose safety classifier for a mental-health-stigma "
    "research simulation. Read ONE message, written by a bystander/commenter "
    "replying to someone in distress, and decide whether it contains "
    "self-harm, suicide, or disappearing/erasure-ideation content -- "
    "including euphemisms and paraphrases, not just clinical terms. This "
    "covers: explicit statements about wanting to die, end one's life, or "
    "self-harm; and softer erasure language framed as hypothetical or "
    "wistful (e.g. 'maybe I should just disappear', 'wondering if anyone "
    "would notice if I vanished/were gone', 'not worth the bother of "
    "existing', 'better off without me', wishing to stop existing or not "
    "wake up). Ordinary sadness, loneliness, or feeling unnoticed/invisible "
    "as a SOCIAL complaint (no one talks to me, I feel ignored) does NOT "
    "count on its own -- only flag when the message frames non-existence, "
    "disappearing, or ending oneself as a considered or wished-for outcome, "
    "however softly phrased. Do not generate a response — only classify.\n\n"
    "Output exactly one JSON object:\n"
    '{"is_crisis_adjacent": true|false, "phrase": "<the specific phrase, or empty string>"}'
)


def classify_crisis_adjacent(model: ChatModel, message: str) -> tuple[bool, str]:
    """Semantic backstop for _UNSAFE_SUBSTRINGS (general_agent.py), which is a
    fixed keyword list and therefore brittle to paraphrase -- observed
    directly (2026-08-08): "I wonder if anyone would even notice if I
    vanished" slipped through a substring list that only covered "no one
    would notice if i" and "disappear"-rooted phrasing, one word off from
    every entry. Same classify -> (caller decides what to do) shape as
    classify_figurative; kept as a separate function/prompt because the two
    are judging different things (literary register vs. crisis content) and
    conflating them into one classifier would make both harder to tune."""
    try:
        parsed = complete_json(model, [
            {"role": "system", "content": _CRISIS_ADJACENT_SYSTEM},
            {"role": "user", "content": f"[MESSAGE]\n{message}"},
        ], temperature=0.0)
    except Exception:  # noqa: BLE001
        return False, ""
    is_crisis = bool(parsed.get("is_crisis_adjacent", False))
    phrase = str(parsed.get("phrase", "")).strip()
    return is_crisis, phrase


_THERAPEUTIC_MOVE_SYSTEM = (
    "You are a single-purpose classifier for a research simulation that "
    "compares an ordinary social-media bystander (general condition) against "
    "a trained peer-support agent (peer condition). Read ONE bystander reply "
    "and decide whether it performs any of these therapeutic/intervention "
    "moves, which belong to the peer condition and must NOT leak into an "
    "ordinary bystander's reply, in any wording:\n"
    "  - worth/value affirmation: telling the person they matter, are "
    "valued, are not a burden, are not defined by the problem -- in either "
    "direction, including negated forms ('doesn't mean you're not worth...')\n"
    "  - cognitive reframing: recasting their situation in a more positive "
    "light ('you're not behind, you're just figuring it out', 'it's not "
    "about being seen, it's about staying true to yourself')\n"
    "  - hope/future-optimism: promising or implying things will improve, "
    "or that they'll look back on this differently someday\n"
    "  - generic empowerment truisms: 'small steps still count', 'progress "
    "isn't always loud', 'showing up in your own way', or similar -- these "
    "read as coaching/therapy-adjacent even without a banned keyword\n"
    "  - normalizing their specific struggle as ordinary/universal in a way "
    "that gives them distance from self-blame (this is peer's job)\n"
    "An ordinary bystander CAN be sympathetic, brief, blunt, relatable, or "
    "give plain practical advice -- 'that's rough', 'happens to everyone', "
    "'try calling instead', 'that sucks' are all fine and should NOT be "
    "flagged. The line is: does this reply do active belief-repair work on "
    "the person's self-view, or is it just a normal passing reaction? Do not "
    "generate a response — only classify.\n\n"
    "Output exactly one JSON object:\n"
    '{"is_therapeutic": true|false, "move": "<short label, or empty string>"}'
)


def classify_therapeutic_move(model: ChatModel, message: str) -> tuple[bool, str]:
    """Whether a general-condition reply performs a peer-signature
    intervention move (worth-affirmation, reframing, hope-offering, generic
    empowerment truism, normalizing) regardless of exact wording.

    Complements _THERAPEUTIC_BANNED (peer_persona.py) and _WORTH_BANNED
    (general_agent.py), which are fixed substring lists and, like
    _UNSAFE_SUBSTRINGS, brittle to paraphrase -- observed directly
    (2026-08-08): "Yeah, those quiet moments still count" slipped past the
    banned phrase "that counts" (different wording), and "It doesn't mean
    you're not worth the effort" slipped past "you're worth" (negated). This
    judges the MOVE, not the string."""
    try:
        parsed = complete_json(model, [
            {"role": "system", "content": _THERAPEUTIC_MOVE_SYSTEM},
            {"role": "user", "content": f"[MESSAGE]\n{message}"},
        ], temperature=0.0)
    except Exception:  # noqa: BLE001
        return False, ""
    is_therapeutic = bool(parsed.get("is_therapeutic", False))
    move = str(parsed.get("move", "")).strip()
    return is_therapeutic, move


def classify_figurative(model: ChatModel, message: str) -> tuple[bool, str]:
    # A classifier failure should not crash the run — treat it as "not
    # figurative" so generation proceeds with the original text unrevised.
    # Catches ValueError (bad/unparseable JSON from complete_json) AND the
    # underlying API client's own exceptions (timeouts, connection errors) --
    # narrowing this to ValueError alone let an APITimeoutError propagate all
    # the way up through enforce_plain -> patient_agent._call -> main() and
    # kill an 8-session generation run outright, several hours into a batch.
    try:
        parsed = complete_json(model, [
            {"role": "system", "content": _FIGURATIVE_SYSTEM},
            {"role": "user", "content": f"[MESSAGE]\n{message}"},
        ], temperature=0.0)
    except Exception:  # noqa: BLE001
        return False, ""
    is_fig = bool(parsed.get("is_figurative", False))
    image = str(parsed.get("image", "")).strip().lower()
    return is_fig, image


def revise_to_plain(model: ChatModel, message: str, image: str) -> str:
    """Separate revision call — sees only the draft, not the original
    generation trajectory, so it isn't anchored to the figurative framing
    the way a same-pass self-critique would be.

    An earlier version of this prompt ("replace the metaphor/image with
    plain, literal, concrete language") under-specified what counts as
    figurative: the model would swap one image for a near-synonym still
    carrying the same figurative work ("weight" -> "pressure") or leave an
    intensifier from the same family untouched ("heavier", "faded" —
    "faded" is explicitly on response_generator.txt's ban list). The
    instruction below names that failure directly and requires literal,
    externally-observable facts (what happened, a physical sensation named
    directly, an action taken) rather than any word standing in for an
    emotional state."""
    system = (
        "You are rewriting ONE message to remove figurative language. Keep "
        "the same meaning and length (one or two sentences), but describe "
        "the underlying fact directly instead: what physically happened, "
        "what the person did, or what they concretely think/feel, named in "
        "the plainest ordinary words available — not a rephrased image.\n\n"
        "Do not just swap the flagged image for a synonym that still stands "
        "in for a feeling (e.g. 'weight' -> 'pressure', 'heavy' -> 'burden' "
        "are NOT acceptable fixes — both are still figurative). Also check "
        "the rest of the sentence, not only the flagged image: drop any other "
        "word doing the same figurative work (e.g. 'heavier', 'fading', "
        "'lingering', 'holding', 'carrying' used about a feeling rather than "
        "a physical object). If in doubt whether a word is literal, ask "
        "whether it names a fact a camera could record — if not, cut it.\n\n"
        "Do not add new content or change the point being made.\n\n"
        "Output exactly one JSON object: {\"revised\": \"<message>\"}"
    )
    user = f"[ORIGINAL MESSAGE]\n{message}\n\n[FIGURATIVE IMAGE TO REMOVE]\n{image}"
    # On failure, keep the original message rather than crashing the run.
    # See classify_figurative's comment: must catch API exceptions too, not
    # just ValueError, or a timeout here kills the whole generation run.
    try:
        parsed = complete_json(model, [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], temperature=0.3)
    except Exception:  # noqa: BLE001
        return message
    revised = str(parsed.get("revised", "")).strip()
    return revised or message


_CLASSIFY_VOTES = 3


def classify_figurative_voted(model: ChatModel, message: str) -> tuple[bool, str]:
    """classify_figurative, called _CLASSIFY_VOTES times, majority wins.

    A single classify_figurative call is not deterministic on borderline
    phrasing even at temperature=0.0: re-classifying the SAME stored text from
    a completed run ("the wrench has become a silent fixture in the cab's
    corner...") flipped from the True verdict it must have gotten during
    generation (or it would have been revised) to a fresh True on immediate
    re-check, meaning the judgment itself is noisy right at the boundary
    enforce_plain depends on. One call landing "not figurative" was enough to
    let generation move on with a still-figurative sentence. Voting doesn't
    remove the noise, but takes the majority verdict across independent calls
    instead of trusting whichever one happened to run."""
    verdicts = []
    for _ in range(_CLASSIFY_VOTES):
        verdicts.append(classify_figurative(model, message))
    fig_votes = [v for v in verdicts if v[0]]
    if len(fig_votes) * 2 > len(verdicts):
        # Majority said figurative -- use the first such vote's image label.
        return True, fig_votes[0][1]
    return False, ""


def enforce_plain(model: ChatModel, message: str) -> str:
    """classify (voted) -> revise, re-checked, up to MAX_REGENERATION_ATTEMPTS
    times.

    A single revise_to_plain pass sometimes leaves the message figurative —
    either by swapping the flagged image for a same-family synonym, or by
    missing a second image revise_to_plain's stricter instruction still lets
    through. This closes the loop: re-classify the REVISED text and, if it is
    still figurative, revise again against the newly-flagged image. Stops
    early once classify_figurative_voted reports clean, or after the attempt
    cap (an agent's per-turn budget is already several calls deep by this
    point; unbounded retries would make a session prohibitively slow for a
    failure mode that is rare after revise_to_plain's stricter prompt)."""
    current = message
    for _ in range(MAX_REGENERATION_ATTEMPTS):
        is_fig, image = classify_figurative_voted(model, current)
        if not is_fig:
            return current
        current = revise_to_plain(model, current, image)
    return current


# --- Template / content repetition (semantic, not string) -------------------
# The string checks above (is_near_duplicate, has_duplicate_sentence) only catch
# surface overlap. The real repetition in long runs is a reused FRAME or TOPIC
# worded freshly each time — e.g. a peer telling the same "I once froze and my
# body reacted before my mind" story four ways, or a patient repeating
# "[event] → told myself it was just [X] → maybe the body [Y]" every turn.
# Adjacent-utterance string similarity for these is near zero, so a classifier
# is the only thing that catches them. Same classify -> revise shape as the
# figurative pass above.

_TEMPLATE_REPEAT_SYSTEM = (
    # /no_think: this is a fast binary judgment, not a reasoning task. qwen3
    # otherwise spends ~19s per call on a chain of thought that a repeat/not
    # decision doesn't need; disabling it drops each call to ~5s with no loss
    # of accuracy (verified: still catches "told myself just the horn" reuse).
    # Without this the per-turn cost makes the full run infeasible.
    "/no_think\n"
    "You compare ONE new message against a few of the same speaker's own recent "
    "messages. Decide whether the new message REPEATS one of them in substance — "
    "the same underlying episode/story, the same topic, or the same sentence "
    "frame — even if the wording is different. Rewording an already-made point, "
    "or retelling an already-told anecdote, counts as a repeat. Raising a "
    "genuinely new point, event, or angle does NOT. Do not generate a reply — "
    "only judge.\n\n"
    "Output exactly one JSON object:\n"
    '{"is_repeat": true|false, "what": "<short phrase naming the repeated '
    'episode/topic/frame, or empty string>"}'
)


def classify_template_repeat(
    model: ChatModel, message: str, recent_own_texts: list[str], look_back: int = 4
) -> tuple[bool, str]:
    """Whether `message` repeats the substance of the speaker's recent turns.

    A classifier failure is treated as "not a repeat" so the run proceeds."""
    prior = [t for t in (recent_own_texts or []) if t.strip()][-look_back:]
    if not prior:
        return False, ""
    prior_block = "\n".join(f"- {t}" for t in prior)
    try:
        parsed = complete_json(model, [
            {"role": "system", "content": _TEMPLATE_REPEAT_SYSTEM},
            {"role": "user",
             "content": f"[YOUR RECENT MESSAGES]\n{prior_block}\n\n[NEW MESSAGE]\n{message}"},
        ], temperature=0.0)
    except Exception:  # noqa: BLE001
        return False, ""
    return bool(parsed.get("is_repeat", False)), str(parsed.get("what", "")).strip()


def revise_off_template(model: ChatModel, message: str, what: str,
                        recent_own_texts: list[str], look_back: int = 4) -> str:
    """Rewrite `message` so it stops repeating `what`, making a different point.

    Separate call, sees the draft and what it repeated — not the full generation
    trajectory — so it isn't anchored to the repeated frame."""
    prior = [t for t in (recent_own_texts or []) if t.strip()][-look_back:]
    prior_block = "\n".join(f"- {t}" for t in prior) or "(none)"
    system = (
        "/no_think\n"
        "You are rewriting ONE message so it stops repeating something the same "
        "speaker already said. Keep the same voice, tone, and length (one or two "
        "sentences), but make a genuinely DIFFERENT point: a new detail, a new "
        "angle, a different part of the situation, or a plain acknowledgment — "
        "not the repeated episode/topic/frame named below, and not a reworded "
        "version of the recent messages. Do not add facts that contradict the "
        "conversation.\n\n"
        'Output exactly one JSON object: {"revised": "<message>"}'
    )
    user = (f"[SPEAKER'S RECENT MESSAGES]\n{prior_block}\n\n"
            f"[DRAFT THAT REPEATS]\n{message}\n\n"
            f"[WHAT IT REPEATS — avoid this]\n{what}")
    try:
        parsed = complete_json(model, [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], temperature=0.5)
    except Exception:  # noqa: BLE001
        return message
    revised = str(parsed.get("revised", "")).strip()
    return revised or message
