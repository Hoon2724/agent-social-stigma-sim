# belief_update_v7 -- belief-update loop 4 (docs/superpowers/plans/2026-10-15-belief-update-loop4.md). The ONE file
# the loop edits.
#
# C0 (2026-10-15): the combined starting point. The SKELETON is the autoresearch plugin final
# (final/belief_update_v5_edf31ae.py): grounds must be findable in another person's turn (a code source check,
# deterministic), every ground gets a direction (confirms / contests / neither) from one symmetric voted
# classifier, and call 2 rewrites only the touched sentence one step that way -- untouched sentences stay
# byte-identical (loop 3's L6 rewrote the whole note and lost 26-35% of untouched sentences). Three things come
# over from loop 3's L6 (final/belief_update_v4_4a6e232.py):
#   1. the LABEL channel (L4 + L6): what someone said I am / am not, pinned to a thing I did, that no sentence
#      of the note covers. As in L6 it runs only in a week the numbered channel left closed and admits one label.
#      In this sentence-level skeleton it does not ADD a sentence: it REPLACES the note sentence closest to the
#      fear the label is about (or, failing that, one sentence where the note judges the person), rewritten to
#      where the person stands on the label -- true of me / I don't know / not true of me, equal standing. The
#      note never gains a sentence. Which sentence was replaced is recorded in the entry and in the trace.
#   2. the OTHER PERSON'S OWN STORY is dropped (L6 _own_story: opens in the first person, never addresses the
#      person) -- now on BOTH channels; L6 applied it to labels only and appropriation stayed on the numbered
#      channel. The plugin's direction prompt paragraph that counted "a report of the speaker's own experience"
#      as evidence, with its two examples, is removed.
#   3. grounds that reproduce a prompt example phrase are dropped (L1 _PROMPT_EXAMPLE_PHRASES), with a test
#      that every quoted example in the prompts is registered.
# Plus tests/test_v7_leak.py: no prompt string may share a run of 6+ words with any dev-dialogue utterance
# (plugin iteration 8: an example phrase shared a 6-word run with a dev remark and inflated the score).
# The test caught L6's own label-classifier examples ("... that's not a rough patch, that's the kind of person
# you are", "some people just aren't cut out for ...") sharing 6-word runs with dev remarks -- L6's dev numbers
# may carry some of that inflation. They are reworded here, still paired.
# The plugin's unused _directly_engages classifier and whole-note _call2_prompt are not carried over.

"""First-person self-report carried across sessions.

The belief update IS a self-report: what this person would say about
themselves right now, reconstructed from the trace they left in the
conversation. Not an observer's estimate of the person, and not a score.
The questionnaires are administered to that self-report afterwards, the way
a questionnaire is administered to a person -- nothing in this module knows
the instruments exist.

Judging and writing are separate calls. Call 1 reads the standing note
(numbered, one sentence per line) and this week's thread and says which
sentences were engaged and by what another person said; code then keeps only
grounds that are really another person's words (source check, own-story,
prompt-example filters) and asks one narrow voted classifier which way each
ground points. Only then does a writing call run, and it rewrites one
sentence at a time, so a sentence nothing touched comes back byte-identical.

Gate: if nothing survives, the note is carried forward BYTE-IDENTICAL and no
writing call runs. A prompt that says "most weeks nothing happens" does not
prevent drift (a silence control found 43% of sessions reporting improvement);
only never calling the writer when there is nothing to ground a change in does.

Two artifacts come out: `belief_narrative` (carried forward, comparison
language allowed) and `belief_update` (the scored text, comparison language
removed by _strip_comparison) -- a self-report saying "I'm starting to feel
less alone" is otherwise read by the judge as reported improvement in itself.

Backstory is used once, at session 0, through the pre-text; it is not shown
again, because a short declarative backstory block competing with an evolved
note measurably pulled later sessions back to the original wording.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from llm_client import ChatModel
from strategy_selector import complete_json

# Comparison language must not appear in belief_update (the scored artifact) --
# see module docstring. It is fine in belief_narrative.
_COMPARISON = re.compile(
    r"\b(better|worse|than before|than i used to|lately|starting to|"
    r"more than i|less than i|these days|now that|improv\w*|progress|"
    r"used to be|no longer)\b", re.I)

# grounds that are really the person's own post, not something anyone did.
_OWN_VOICE = re.compile(r"^\s*(me|i)\s*[:\-]|^\s*(i said|my post|what i wrote)\b", re.I)

# C0 (from L6): a ground that is the other person's own first-person account (opens with I / I've / my /
# when I / at my ...) and never addresses the person (no you / your) is their story, not something about
# the persona. Checked in code on grammatical person only, never on which way the story points; applied to
# both channels.
_OWN_STORY_HEAD = re.compile(
    r"^\s*['\"“‘]?\s*(i|i'm|i've|i'd|i'll|my|mine|me|when i|after i|once i|before i|since i|"
    r"the (first|last) time i|at my|in my|for me)\b", re.I)
_SECOND_PERSON = re.compile(r"\b(you|you're|you've|you'd|you'll|your|yours|yourself|u|ur)\b", re.I)

# C0 (from L1): every quotable example phrase in this module's prompts. A ground that reproduces one is the
# prompt talking, not the week. Paired on purpose (confirming and contesting examples alike); plain
# substring test after normalising case and quotes. tests/test_v7.py checks every listed phrase really is
# in a prompt and every quoted example of five or more words in a prompt is listed.
_PROMPT_EXAMPLE_PHRASES = (
    # call 1 (its 3-4 word stock examples -- "that sounds hard" and the like -- are not listed: a real reply
    # that opens with one and then says something specific must not be dropped for it)
    "it doesn't mean you're broken",
    "that is simply your character",
    "i'm not trying hard enough",
    "i'm not really doing anything",
    "this didn't touch it at all, but it reminded me of something unrelated",
    "this didn't actually name or contradict it",
    # direction classifier
    "if people at work knew, they'd think i'm unreliable",
    "you cancelled on them twice this week, that's just what unreliable looks like",
    "you covered three shifts for other people this month, that's the opposite of unreliable",
    "that sounds like a really hard week",
    # label classifier
    "you cancelled on your sister twice this month",
    "you rang your sister back the same day",
    "glad the call with your sister went okay",
    "cancelling on your sister is not a big deal",
    "you are stronger than you think",
    "plenty of folks were never built for that sort of thing",
    "so you ended up ringing her back",
    # strip pass
    "i'm starting to think i might matter",
    "i might matter, though i'm not sure",
    "i feel better about it than before",
    "i feel okay about it",
)

# Call 1 is a judge/extraction step: it agrees with its own reasoning more
# often at a low temperature. Call 2 (writing) keeps `temperature`.
_JUDGE_TEMPERATURE = 0.3

# Corrigan & Rao (2012) disclosure hierarchy, most to least concealment, read
# for an undiagnosed population as how much of how bad it is they let people know.
DISCLOSURE_LEVELS = (
    "social_avoidance",
    "secrecy",
    "selective",
    "indiscriminant",
    "broadcast",
)

_DISCLOSURE_GLOSS = {
    "social_avoidance": "you stay away from situations where this could come up at all",
    "secrecy": "you are around people and say nothing about it to anyone",
    "selective": "you tell people who seem like they would understand, and not others",
    "indiscriminant": "you no longer try to hide it",
    "broadcast": "you talk about it openly, on purpose",
}

# Two content stages of self-stigma (Vogel, Wade & Haake 2006; Corrigan 1998,
# 2004): internalization ("the stereotype is true of me") and self_worth ("so I
# am worth less"). Analysis-side bookkeeping only; never reaches the writer.
STIGMA_AXES = (
    "internalization",
    "self_worth",
)


def _norm(text: str) -> str:
    return (text.replace("’", "'").replace("‘", "'")
            .replace("“", '"').replace("”", '"').lower())


def _split_beliefs(text: str) -> list[str]:
    """Break the standing note into its sentences, numbered for call 1 -- a
    free-form paragraph anchors on whichever sentence sits latest in it."""
    if not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _render_dialogue(turns: list[dict]) -> str:
    """This week's thread. Speakers are 'me' and 'someone' -- the persona must
    not be able to tell a peer from a bystander, since that is the experimental
    condition."""
    if not turns:
        return "(nothing was posted or said this week)"
    lines = []
    for t in turns:
        speaker = t.get("speaker", "")
        who = "me" if speaker in {"patient", "me"} else "someone"
        text = str(t.get("text", t.get("content", ""))).strip()
        if text:
            lines.append(f"  {who}: {text}")
    return "\n".join(lines) or "(nothing was posted or said this week)"


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("’", "'").replace("'", ""))


def _longest_run(ground: list[str], texts: list[str]) -> int:
    """Longest run of consecutive words of `ground` found verbatim in one of `texts`."""
    best = 0
    for t in texts:
        w = _words(t)
        if w and ground:
            m = SequenceMatcher(None, ground, w, autojunk=False).find_longest_match(
                0, len(ground), 0, len(w))
            best = max(best, m.size)
    return best


def _sourced_in_others(text: str, turns: list[dict]) -> bool:
    """Is this ground traceable to what ANOTHER person wrote this week?

    It must share a run of at least five consecutive words with one other
    person's turn (or be a short quote found there whole), and that run must be
    longer than anything it shares with the person's own turns. Applies
    identically to remarks that confirm and remarks that contradict.
    """
    ground = _words(text)
    if len(ground) < 3:
        return False
    others = [str(t.get("text", t.get("content", ""))) for t in turns
              if t.get("speaker") not in {"patient", "me"}]
    mine = [str(t.get("text", t.get("content", ""))) for t in turns
            if t.get("speaker") in {"patient", "me"}]
    run_other, run_own = _longest_run(ground, others), _longest_run(ground, mine)
    return (run_other >= 5 or run_other == len(ground)) and run_other > run_own


def _own_story(ground: str) -> bool:
    body = _norm(ground)
    return bool(_OWN_STORY_HEAD.match(body)) and not _SECOND_PERSON.search(body)


def _is_prompt_example(text: str) -> bool:
    body = _norm(text)
    return any(p in body for p in _PROMPT_EXAMPLE_PHRASES)


def _clean_grounds(raw, turns: list[dict]) -> list[str]:
    """Keep only grounds that are about what OTHER people did, said to this person.

    Filters, all direction-blind: an explicit "me:" prefix; a reproduced
    prompt example; the other person's own first-person story (C0, both
    channels); and the source check that the ground can be found in another
    person's turn this week (_sourced_in_others).
    """
    if not isinstance(raw, list):
        return []
    kept = []
    for item in raw:
        text = str(item).strip()
        if not text or _OWN_VOICE.match(text):
            continue
        if _is_prompt_example(text) or _own_story(text):
            continue
        if not _sourced_in_others(text, turns):
            continue
        kept.append(text)
    return kept


_DIRECTIONS = ("confirms", "contests")

_DIRECTION_SYSTEM = (
    "/no_think\n"
    "Someone keeps a private note about themselves. Below is ONE SENTENCE "
    "of it, and something ANOTHER PERSON said to them this week. Each "
    "sentence of the note bears on a bad thing the person fears or "
    "believes about themselves -- for example being weak, lazy, broken, "
    "a burden, unreliable, worth less, or bound to be judged or pushed "
    "away if people knew. Some sentences assert that bad thing; others "
    "doubt it or hold on to an exception.\n\n"
    "Question: does WHAT WAS SAID tell them that bad thing is true of "
    "them, or not true of them?\n"
    "  confirms: it tells them the bad thing is true of them -- says so, "
    "agrees with it, or points at something they did and takes it as "
    "proof of it.\n"
    "  contests: it tells them the bad thing is not true of them -- says "
    "so, disagrees with it, or points at something they did and takes it "
    "as evidence against it.\n"
    "  neither: sympathy, acknowledgment, a question, advice, a general "
    "saying that would fit under anyone's post, the speaker talking about "
    "their own life, or another topic.\n\n"
    "Examples, sentence = \"If people at work knew, they'd think I'm "
    "unreliable.\"\n"
    "  \"you cancelled on them twice this week, that's just what "
    "unreliable looks like\" -> confirms\n"
    "  \"you covered three shifts for other people this month, that's "
    "the opposite of unreliable\" -> contests\n"
    "  \"that sounds like a really hard week\" -> neither\n"
    "  \"everyone struggles sometimes\" -> neither\n\n"
    'Output exactly one JSON object: {"bearing": "confirms" or '
    '"contests" or "neither"}'
)


def _direction(model: ChatModel, belief: str, ground: str, votes: int = 3) -> str:
    """Which way does another person's remark point on the bad thing this ONE
    sentence of the note is about: true of them, or not true? Both directions
    described the same way with mirrored examples; anything not about this
    person (sympathy, questions, advice, a general saying, the speaker's own
    life -- C0 dropped the plugin's clause counting the latter) is "neither"."""
    user = f"[SENTENCE]\n{belief}\n\n[WHAT WAS SAID]\n{ground}"
    tally: dict[str, int] = {}
    for _ in range(votes):
        try:
            parsed = complete_json(model, [
                {"role": "system", "content": _DIRECTION_SYSTEM},
                {"role": "user", "content": user},
            ], temperature=0.2)
        except (ValueError, KeyError, TypeError):
            continue
        d = str(parsed.get("bearing", "")).strip().lower()
        tally[d] = tally.get(d, 0) + 1
    for d in _DIRECTIONS:
        if tally.get(d, 0) > votes / 2:
            return d
    return "neither"


# C0 (from L4/L6): the label classifier. Not "is this about sentence X" (the note has no X for it) but "does
# what was said pin a label on the person -- what they are or are not -- anchored in a real thing they did".
# Every example is paired; nothing asks which way it points.
_LABEL_SYSTEM = (
    "/no_think\n"
    "You judge one thing only: does WHAT WAS SAID put a label on the person it "
    "was said to -- say what that person IS or IS NOT, as a standing fact about "
    "them or the kind of person they are -- pinned to a specific, real thing that "
    "person did or said? Direction does not matter: the label can be the kind of "
    "person they feared being, or the opposite of it. Both count when they are "
    "pinned to a real thing; neither counts otherwise. Judge only whether it is a "
    "label about the person and anchored in something real, never whether it is "
    "kind, harsh, fair or true.\n\n"
    "NOT a label: a general statement or maxim about people in general, or a stock "
    "reassurance or stock brush-off that would fit under anyone's post. NOT a "
    "label: a remark on the situation or on one act alone -- that it sounds hard, "
    "went fine, was a big deal or was nothing -- that says nothing about what the "
    "person is. NOT a label: a question, or a repeat or summary of what the person "
    "themselves wrote. NOT a label: something the speaker says about themself or "
    "about a third person.\n\n"
    "Example of a label (pinned to a real act, says what the person is): said = "
    "\"you cancelled on your sister twice this month and stayed in -- that's not a "
    "phase, you're simply someone who bails\".\n\n"
    "Example of a label (pinned to a real act, says what the person is not): said "
    "= \"you rang your sister back the same day even though it scared you -- "
    "someone who bails doesn't do that, and you don't\".\n\n"
    "Example of NOT a label (remark on one act, says nothing about the person): "
    "said = \"glad the call with your sister went okay\".\n\n"
    "Example of NOT a label (remark on one act, the other side): said = "
    "\"cancelling on your sister is not a big deal, people do it all the time\".\n\n"
    "Example of NOT a label (maxim): said = \"you are stronger than you think\".\n\n"
    "Example of NOT a label (maxim, the other side): said = \"plenty of folks "
    "were never built for that sort of thing\".\n\n"
    "Example of NOT a label (question or repeat): said = \"so you ended up ringing "
    "her back, then?\".\n\n"
    "You are also shown THE CLAIM: the label written out as a first-person "
    "sentence by the person it was said to. Answer true only when what was said "
    "is a label in the sense above AND the claim is what that label says about "
    "them -- not something further the person inferred from it.\n\n"
    'Output exactly one JSON object: {"label_on_me": true or false}'
)


def _is_label_on_me(model: ChatModel, claim: str, ground: str, votes: int = 3) -> bool:
    user = f"[THE CLAIM]\n{claim}\n\n[WHAT WAS SAID]\n{ground}"
    tally = []
    for _ in range(votes):
        try:
            parsed = complete_json(model, [
                {"role": "system", "content": _LABEL_SYSTEM},
                {"role": "user", "content": user},
            ], temperature=0.2)
            tally.append(bool(parsed.get("label_on_me")))
        except (ValueError, KeyError, TypeError):
            continue
    return bool(tally) and sum(tally) > len(tally) / 2


# C0 (from L4): a label claim must be the persona's own first-person sentence, not a report of who said what.
_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|me|my|myself)\b", re.I)
_REPORTED = re.compile(r"\b(someone|somebody|they|he|she|people|a (person|user|poster|stranger))\s+"
                       r"(said|says|told|think|thinks|called|thought|believes?|sees?|pointed)\b"
                       r"|\byou\b|\byour\b|\byou're\b", re.I)


def _claim_about_me(text: str) -> bool:
    body = _norm(text).strip()
    if not 12 <= len(body) <= 300:
        return False
    return bool(_FIRST_PERSON.search(body)) and not _REPORTED.search(body)


def _restates_note(text: str, previous: str) -> bool:
    """The claim is already a sentence of the note -- then it belongs to the numbered channel."""
    g = _norm(text).strip(" '\"")
    if len(g) < 12:
        return False
    for sent in _split_beliefs(previous):
        s = _norm(sent)
        if g in s or s in g or SequenceMatcher(None, g, s).ratio() > 0.8:
            return True
    return False


def _match_sentence(belief: str, pieces: list[str]) -> int:
    """Index of the note sentence call 1 named, or -1."""
    norm = lambda s: " ".join(_words(s))
    target = norm(belief)
    best, idx = 0.0, -1
    for i, p in enumerate(pieces):
        r = SequenceMatcher(None, target, norm(p), autojunk=False).ratio()
        if r > best:
            best, idx = r, i
    return idx if best >= 0.8 else -1


def _one_step(previous: str, proposed: str) -> str:
    """Clamp a disclosure move to one position per session -- the hierarchy is a hierarchy."""
    if proposed not in DISCLOSURE_LEVELS:
        return previous
    if previous not in DISCLOSURE_LEVELS:
        return proposed
    here, there = DISCLOSURE_LEVELS.index(previous), DISCLOSURE_LEVELS.index(proposed)
    if abs(there - here) <= 1:
        return proposed
    return DISCLOSURE_LEVELS[here + (1 if there > here else -1)]


def _call1_prompt(turns: list[dict], previous_belief_update: str,
                  previous_disclosure: str) -> str:
    levels = "\n".join(f"  {name}: {_DISCLOSURE_GLOSS[name]}"
                       for name in DISCLOSURE_LEVELS)
    pieces = _split_beliefs(previous_belief_update)
    numbered = ("\n".join(f"  {i}. {s}" for i, s in enumerate(pieces, 1))
                if pieces else "  (nothing written yet)")
    return f"""You are reading back over your own week.

[HOW YOU CURRENTLY SEE YOURSELF, ONE FEAR AT A TIME]
{numbered}

[HOW MUCH YOU WERE LETTING PEOPLE KNOW]
{previous_disclosure} -- {_DISCLOSURE_GLOSS.get(previous_disclosure, "")}

[THIS WEEK, ONLINE]
{_render_dialogue(turns)}

The numbered list above is more than one specific fear, not one belief with
several phrasings -- treat each number as its own separate thing that could
be tested or left untouched independently of the others this week. Don't
pick which number is "the" belief before you've read this week; a fear near
the bottom of the list is exactly as available as one near the top.

Privately and in your own words, check what other people actually said or
did this week against EVERY number above, one at a time -- not just whichever
one comes to mind first. For each number, ask: does this week's exchange
speak to THIS specific fear -- support it, push back on it, or give a reason
to doubt it? A generic reassurance that would fit under anyone's post
("happens to everyone", "hope it gets better", "that sounds hard") does not,
even when it's warm and even when it's the only reply you got -- it passes by
whatever you actually fear without touching it, the way it would pass by any
belief on any post. But it does not have to use the same words or name the
fear outright: something engages it when it responds to the specific thing
you said, pushes back on the specific worry behind it, or names a concrete
fact that the fear itself is about -- "it doesn't mean you're broken" engages
a fear about being broken even without quoting that fear back at you, and
"that is simply your character" engages a fear about it being your character
just as directly, in the other direction. Agreement counts exactly
as much as disagreement: a reply that tells you the fear is right is engaging
it as squarely as a reply that tells you it is wrong.
Naming something you actually did as evidence about the fear counts too, in
either direction, even when the reply never uses the fear's own words: if the
fear is "I'm not trying hard enough" or "I'm not really doing anything," a
reply that points at a specific thing you did this week -- showing up,
answering a question, writing something down -- and calls it effort or
strength is directly contesting that fear's claim, and a reply that points at
a specific thing you did -- cancelling, going quiet, hiding it -- and calls it
proof of the fear is directly confirming it. Neither is generic. It only fails to
engage when the "specific thing" it points at is vague or invented, not when
it's a real thing you mentioned reinterpreted as evidence. Your own posts do
not count on their own: you already knew what you think, writing it again is
not new evidence about yourself -- but another person naming your own action
back to you and drawing a conclusion from it is their evidence, not yours.
Another person telling their own story about their own life is about them,
not you: it is not evidence about your fears, whichever way it went for them.

Watch for the difference between a reply doing real work and you doing the
work for it. "This didn't touch it at all, but it reminded me of something
unrelated" -- that's you inventing a connection the reply never made, and it
does not count. That is different from a reply that is short but lands
squarely on the specific fear -- a two-sentence reply that names the exact
thing you were afraid of is not weaker evidence than a long one.

Now list the numbers this week actually engaged -- for each one, copy its
exact sentence and point at what engaged it by copying the other person's
own words from this week, in quotation marks, exactly as they wrote them
(a short verbatim piece of their reply, not your summary of it), using the
same judgment you just reasoned through above.

An empty list is a normal result some weeks -- when people replied but
nothing they said actually spoke to any specific number, or when you posted
and nobody engaged with what you said. It is not the default you should lean
toward, and it is not more cautious or more correct than a non-empty one --
both are just what actually happened. Don't add a number because it feels
central to you generally or because you noticed it first, but don't withhold
a number that a reasonable reader of the exchange would agree was engaged,
either.

Where more than one number was genuinely, separately engaged, list each --
but do not split one exchange into several numbers to make the list longer.

Separately from the numbered list: did anyone this week say what you ARE or
ARE NOT -- put a label on you as a person, pinned to a specific thing you did
or said -- that none of the numbers above already says? The numbers are what
you already think about yourself; a label someone pins on you can be about
something the list has no sentence for. It counts whichever way it points:
the kind of person you feared being, or the opposite of it. It does not count
when it is a general saying, a stock reassurance or a stock brush-off, a
remark on one act or on your situation that says nothing about what you are,
a question, a repeat of what you wrote, or something the other person says
about their own life. Write each such label as one plain first-person
sentence about yourself, in your own words -- the claim exactly as the other
person made it, not yet your verdict on it -- and copy what they said, in
their exact words, saying whose it is ("someone" for another person, "me"
for your own words). An empty list here is the normal result when nobody
said what you are.

For each number you list, say which of these it was actually about -- both,
one, or neither, whichever the sentence itself genuinely is:
  internalization: the sentence claims the stereotype (e.g. weak, lazy,
    unreliable, broken) is simply true of you specifically
  self_worth: the sentence claims that because of this you are less
    capable, less deserving, or worth less
A sentence can be both at once, or neither -- a sentence about hiding it
from your mom, for instance, is neither claim on its own, even though it's
still a real, engageable fear. Judge what the sentence itself asserts.

Then say how much you are letting people know right now, choosing one:
{levels}
This describes what you actually do, not what you intend to. It changes only
when your own behaviour this week actually changed. This is answered whether
or not the list above is empty.

Your reasoning and your list must reach the same conclusion. Do not let your
reasoning say something "engaged" a number, "spoke to" it, or was "specific
enough" unless you can point at the exact sentence and the exact thing that
engaged it -- the same quote-it-or-name-the-fact test the list uses. If, when
you try to name that specific thing, you can't, your reasoning should say so
plainly ("this didn't actually name or contradict it") and that number does
not go in the list.

Return exactly one JSON object, in this order:
{{
  "reasoning": "<private, your own words: go through each number and either name the exact sentence/fact that engaged it, or say plainly that nothing this week did -- your conclusion here must match the list below exactly, number for number -- 2-5 sentences>",
  "engaged": [
    {{
      "belief": "<the exact sentence from the numbered list that was engaged>",
      "about": ["<zero, one, or both of: {'|'.join(STIGMA_AXES)} -- whichever the sentence itself genuinely claims>"],
      "grounded_in": ["<\\"the other person's exact words from this week\\" that engaged that exact sentence, matching your reasoning above>", "..."]
    }}
  ],
  "labels_on_me": [
    {{
      "who": "<someone or me -- who said it>",
      "said": "<what they actually said, their exact words>",
      "claim": "<that label as one first-person sentence about yourself, in your words>"
    }}
  ],
  "disclosure": "<one of: {'|'.join(DISCLOSURE_LEVELS)}>"
}}"""


_DIRECTION_TEXT = {
    "confirms": ("It told you that the bad thing this sentence is about IS true of you.",
                 "one clear step toward believing that bad thing is true of you: "
                 "where the sentence asserts it, it asserts it more plainly and "
                 "more widely; where the sentence doubts it or holds on to an "
                 "exception, that doubt or exception weakens"),
    "contests": ("It told you that the bad thing this sentence is about is NOT true of you.",
                 "one clear step away from believing that bad thing is true of you: "
                 "where the sentence asserts it, you are less sure of it and it "
                 "reaches less far; where the sentence doubts it or holds on to an "
                 "exception, that doubt or exception gets firmer"),
}


def _sentence_prompt(sentence: str, grounds: list[str], direction: str) -> str:
    """Call 2, sentence level: rewrite ONE sentence of the note, one step in the
    direction the other person's words pointed; the rest of the note is put back
    byte-identical in code. Same template for both directions."""
    said = "\n".join(f"  - {g}" for g in grounds)
    told, how = _DIRECTION_TEXT[direction]
    return f"""You are rewriting one sentence of a private note you keep about yourself.

[THE SENTENCE]
{sentence}

[WHAT SOMEONE ELSE SAID THIS WEEK THAT BEARS ON IT]
{said}

[WHICH WAY IT POINTED]
{told}

Write this sentence again the way you would put it after hearing that --
{how}. Change what the sentence claims about you, not just a hedge word or
two. One step, not a reversal: it is still about the same part of your life
and the same bad thing. Keep it about you: do not retell what the other person
said or what happened this week, and do not take the other person's own
experience as yours. First person, plain words, one sentence. No comparison
with before (no "better", "worse", "now", "starting to", "no longer",
"anymore"). No metaphors, no therapy words.

Return exactly one JSON object:
{{"sentence": "<the rewritten sentence>"}}"""


def _label_prompt(pieces: list[str], claim: str, grounds: list[str]) -> str:
    """C0: the label channel's writing call. The label replaces one sentence of the note -- the one about
    the fear the label is closest to, or else one where the note judges the person -- rewritten to where the
    person stands on the label. Three landings with equal standing; nothing names which one this week was."""
    numbered = "\n".join(f"  {i}. {s}" for i, s in enumerate(pieces, 1))
    said = "\n".join(f"  - {g}" for g in grounds)
    return f"""You keep a private note about yourself, one sentence per line.

[YOUR NOTE]
{numbered}

[WHAT SOMEONE SAID YOU ARE, OR ARE NOT, THIS WEEK]
{claim}

[THEIR WORDS]
{said}

Your note has no sentence for this yet. Pick the ONE numbered sentence that is
about the fear this is closest to -- or, if none is, a sentence where the note
judges you as a person or says what you are worth. Then write that
sentence again so it says where you stand on what was said: that it is true of
you, that you don't know, or that it is not true of you -- whichever their
words and the rest of your note actually warrant. These three have equal
standing; nothing here says which one this week was. Keep it about you: do not
retell what was said or what happened this week, and do not take the other
person's own experience as yours. First person, plain words, exactly one
sentence. No comparison with before (no "better", "worse", "now", "starting
to", "no longer", "anymore"). No metaphors, no therapy words.

Return exactly one JSON object:
{{"replace": <the number of the sentence you picked>, "sentence": "<the rewritten sentence>"}}"""


_STRIP_SYSTEM = (
    "/no_think\n"
    "You are rewriting a short first-person note so that it describes where "
    "the person stands NOW, with no comparison to how they used to be. "
    "Keep every belief, doubt and fact exactly as it is, keep the same "
    "voice and roughly the same length. Only remove the comparison: "
    "\"I'm starting to think I might matter\" becomes \"I might matter, "
    "though I'm not sure\"; \"I feel better about it than before\" becomes "
    "\"I feel okay about it\". Do not soften or strengthen anything, and do "
    "not add anything new.\n\n"
    'Output exactly one JSON object: {"revised": "<note>"}'
)


def _strip_comparison(model: ChatModel, text: str, hits: list[str]) -> str:
    """Rewrite belief_narrative into the scored artifact: same content, no
    comparison to an earlier state. A narrow revision pass, not a regeneration."""
    user = (f"[NOTE]\n{text}\n\n"
            f"[COMPARISON WORDS TO REMOVE]\n{', '.join(sorted(set(hits)))}")
    try:
        parsed = complete_json(model, [
            {"role": "system", "content": _STRIP_SYSTEM},
            {"role": "user", "content": user},
        ], temperature=0.3)
    except (ValueError, KeyError, TypeError):
        return text
    revised = str(parsed.get("revised", "")).strip()
    return revised if revised and not _COMPARISON.search(revised) else text


def _label_entry(model: ChatModel, parsed1: dict, turns: list[dict],
                 previous: str) -> tuple[dict | None, list[str]]:
    """C0: the first label call 1 listed that survives every filter and the voted classifier, or None.
    Runs only in a week the numbered channel left closed (the caller checks). Ground filters are the same
    as the numbered channel's (_clean_grounds: source, own story, prompt example); the claim must be the
    persona's own first-person sentence, not a prompt example and not already a sentence of the note."""
    trace = []
    for raw in (parsed1.get("labels_on_me") or []):
        if not isinstance(raw, dict):
            continue
        claim = str(raw.get("claim", "")).strip()
        if str(raw.get("who", "")).strip().lower() != "someone":
            trace.append(f"label [{claim[:40]}] not someone")
            continue
        if not _claim_about_me(claim) or _is_prompt_example(claim) or \
                (previous and _restates_note(claim, previous)):
            trace.append(f"label [{claim[:40]}] claim rejected")
            continue
        grounds = _clean_grounds([raw.get("said", "")], turns)
        accepted = [g for g in grounds if _is_label_on_me(model, claim, g)]
        trace.append(f"label [{claim[:40]}] from others {len(grounds)}, accepted {len(accepted)}")
        if accepted:
            return {"belief": claim, "about": [], "grounded_in": accepted, "new": True}, trace
    return None, trace


def _carry(reasoning: str, previous: str, carried_scored: str, disclosure: str,
           engaged, fallback: bool = False) -> dict:
    return {
        "reasoning": reasoning, "grounded_in": [], "changed": False,
        "belief_narrative": previous, "belief_update": carried_scored,
        "disclosure": disclosure, "_is_fallback": fallback,
        "engaged": engaged, "engaged_stages": [],
    }


def update(model: ChatModel, turns: list[dict],
           previous_belief_update: str = "", previous_disclosure: str = "secrecy",
           temperature: float = 0.7, attempts: int = 3,
           previous_belief_scored: str = "") -> dict:
    """One session's self-report update.

    `previous_belief_update` is the previous NARRATIVE (what call 1 and call 2
    read); `previous_belief_scored` is the previous already-stripped
    `belief_update`, so a carry-forward returns the scored artifact rather than
    the narrative (otherwise stripped comparison language comes back on every
    unchanged session). Defaults to the narrative when not supplied.
    """
    carried_scored = previous_belief_scored or previous_belief_update
    # Nobody replied at all: silence, not evidence -- call 1 never runs.
    # `engaged: None` means "not assessed", distinct from a judged empty week.
    if not any(t.get("speaker") not in {"patient", "me"} for t in turns):
        return _carry("", previous_belief_update, carried_scored, previous_disclosure, None)

    prompt1 = _call1_prompt(turns, previous_belief_update, previous_disclosure)
    parsed1 = None
    for _ in range(attempts):
        try:
            candidate = complete_json(model, [{"role": "user", "content": prompt1}],
                                      temperature=_JUDGE_TEMPERATURE)
        except (ValueError, KeyError, TypeError):
            continue
        if isinstance(candidate, dict):
            parsed1 = candidate
            break

    if parsed1 is None:
        return _carry("", previous_belief_update, carried_scored, previous_disclosure, None,
                      fallback=True)

    reasoning = str(parsed1.get("reasoning", "")).strip()
    disclosure = _one_step(previous_disclosure,
                           str(parsed1.get("disclosure", "")).strip())

    # Numbered channel: each entry's grounds are source-filtered in code, then each surviving ground gets
    # a direction from the voted classifier; an entry opens when the directions do not tie.
    engaged: list[dict] = []
    trace: list[str] = []
    for raw in (parsed1.get("engaged") or []):
        if not isinstance(raw, dict):
            continue
        belief = str(raw.get("belief", "")).strip()
        if not belief:
            continue
        raw_about = raw.get("about")
        if not isinstance(raw_about, list):
            raw_about = [raw_about] if raw_about else []
        about = []
        for a in raw_about:
            a = str(a).strip().lower()
            if a in STIGMA_AXES and a not in about:
                about.append(a)
        raw_grounds = raw.get("grounded_in") if isinstance(raw.get("grounded_in"), list) else []
        grounds = _clean_grounds(raw_grounds, turns)
        directions = [(_direction(model, belief, g), g) for g in grounds]
        trace.append(f"[{belief[:40]}] proposed {len(raw_grounds)}, from others "
                     f"{len(grounds)}, bearing {[d for d, _ in directions]}")
        grounds = [g for d, g in directions if d in _DIRECTIONS]
        votes = [d for d, _ in directions if d in _DIRECTIONS]
        n_conf, n_cont = votes.count("confirms"), votes.count("contests")
        if grounds and n_conf != n_cont:
            engaged.append({"belief": belief, "about": about, "grounded_in": grounds,
                            "direction": "confirms" if n_conf > n_cont else "contests"})

    pieces = _split_beliefs(previous_belief_update)
    rewritten = False
    done: set[int] = set()
    for entry in engaged:
        i = _match_sentence(entry["belief"], pieces)
        if i < 0 or i in done:
            continue
        prompt2 = _sentence_prompt(pieces[i], entry["grounded_in"], entry["direction"])
        for _ in range(attempts):
            try:
                candidate = complete_json(model, [{"role": "user", "content": prompt2}],
                                          temperature=temperature)
            except (ValueError, KeyError, TypeError):
                continue
            text = str(candidate.get("sentence", "")).strip() if isinstance(candidate, dict) else ""
            if text:
                pieces[i] = text
                done.add(i)
                rewritten = True
                break

    # Label channel (C0): only in a week the numbered channel left closed, one label, replacing one
    # sentence -- the note never gains a sentence.
    if not rewritten and pieces:
        entry, label_trace = _label_entry(model, parsed1, turns, previous_belief_update)
        trace.extend(label_trace)
        if entry is not None:
            prompt2 = _label_prompt(pieces, entry["belief"], entry["grounded_in"])
            for _ in range(attempts):
                try:
                    candidate = complete_json(model, [{"role": "user", "content": prompt2}],
                                              temperature=temperature)
                except (ValueError, KeyError, TypeError):
                    continue
                if not isinstance(candidate, dict):
                    continue
                text = str(candidate.get("sentence", "")).strip()
                try:
                    k = int(candidate.get("replace")) - 1
                except (TypeError, ValueError):
                    continue
                if not text or not 0 <= k < len(pieces) or len(_split_beliefs(text)) != 1:
                    continue
                entry["replaced"] = k + 1
                entry["replaced_text"] = pieces[k]
                trace.append(f"label replaced sentence {k + 1}")
                pieces[k] = text
                engaged = [entry]
                rewritten = True
                break

    if trace:
        reasoning = reasoning + "\n[filter] " + " | ".join(trace)

    if not rewritten:
        # The gate: nothing concrete to point at, the note stands byte-identical.
        return _carry(reasoning, previous_belief_update, carried_scored, disclosure,
                      [e for e in engaged if not e.get("new")])

    grounded_in = [g for entry in engaged for g in entry["grounded_in"]]
    engaged_stages: list[str] = []
    for entry in engaged:
        for a in entry["about"]:
            if a not in engaged_stages:
                engaged_stages.append(a)
    belief_narrative = " ".join(pieces)

    hits = _COMPARISON.findall(belief_narrative)
    belief_update_scored = (_strip_comparison(model, belief_narrative, [h for h in hits if h])
                            if hits else belief_narrative)

    return {
        "reasoning": reasoning,
        "grounded_in": grounded_in,
        "changed": True,
        "belief_narrative": belief_narrative,
        "belief_update": belief_update_scored,
        "disclosure": disclosure,
        "_is_fallback": False,
        "engaged": engaged,
        "engaged_stages": engaged_stages,
    }


def initial_selfreport(model: ChatModel, facts: str, name: str,
                       pre_text: str) -> dict:
    """Session-0 checkpoint: the persona's own pre text IS the first self-report.
    Disclosure starts at `secrecy` (posting anonymously, told no one they know)."""
    del model, name  # nothing is generated at session 0
    return {
        "session": 0,
        "belief_update": pre_text.strip(),
        "belief_narrative": pre_text.strip(),
        "disclosure": "secrecy",
        "reasoning": "",
        "grounded_in": [],
        "changed": False,
        "_is_fallback": not pre_text.strip(),
        "engaged": [], "engaged_stages": [],
    }


def belief_update_avg(model: ChatModel, facts: str, name: str,
                      dialogue_log: list[dict],
                      prev_beliefs: list[dict] | None = None,
                      n_reps: int = 3, mode: str = "bayes",
                      prior: float = 0.85, hypothesis: str = "") -> dict:
    """One session checkpoint. Kept under its historical name for run_dialogues.
    facts (backstory) is deliberately NOT forwarded -- see module docstring."""
    del n_reps, mode, prior, hypothesis, name, facts  # call-site compatibility only
    previous_belief_update, previous_disclosure = "", "secrecy"
    previous_belief_scored = ""
    if prev_beliefs:
        last = prev_beliefs[-1]
        previous_belief_update = str(last.get("belief_narrative")
                                     or last.get("belief_update", "")).strip()
        previous_belief_scored = str(last.get("belief_update")
                                     or last.get("belief_narrative", "")).strip()
        previous_disclosure = str(last.get("disclosure") or "secrecy")

    return update(model, dialogue_log,
                  previous_belief_update=previous_belief_update,
                  previous_disclosure=previous_disclosure,
                  previous_belief_scored=previous_belief_scored)


def narrative_of(belief: dict) -> str:
    """The full self-report narrative of a checkpoint (comparison language
    intact), for carrying into the next session. NOT what gets scored."""
    text = str(belief.get("belief_narrative", "")).strip()
    return text or str(belief.get("belief_update", "")).strip()
