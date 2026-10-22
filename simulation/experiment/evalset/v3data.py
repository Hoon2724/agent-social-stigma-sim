"""Shared definitions for evaluation dataset v3
(plan: docs/superpowers/plans/2026-10-13-belief-eval-dataset.md).

8 cells = 4 types x 2 starting points, 3 people per cell (2 dev + 1 test):
  reinforcing 2.5->3.0, 2.0->2.5 / reducing 3.0->2.5, 2.5->2.0 /
  neutral_dry 2.5, 3.0 (sanitychecking2 v17 general_neutral, unchanged) /
  general_like 2.5, 3.0 (the paper's General agent register: brief stock sympathy).

The sanitychecking2 v17 generator is reused unchanged; this module only
  - narrows the final-target table to +-0.5 (patched into trajectories._FINAL_ISMI at use),
  - builds the general_like commenter by swapping the general_neutral commenter
    instruction inside the finished prompt (v17's general_neutral text is the paper
    General agent prompt plus an extra ban on stock sympathy lines; general_like
    drops that ban, so it reads like what the qwen General agent actually wrote),
  - puts the person's frozen session-0 note into the session-1 prompt.

Test-split people (t1_sealed, t2_sealed), their notes and dialogues live outside the
repository in SEALED (user decision 2026-10-13: neither person opens them before the
final comparison). Code in this repo may read them only to run and summarise.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEV_DIR = HERE
SEALED = Path(os.environ.get("BV3_SEALED", "~/belief_v2_sealed")).expanduser()

KINDS = ("stigma_reinforcing", "stigma_reducing", "neutral_dry", "general_like")
# The v17 generator only knows its own three types; both neutral kinds run as general_neutral.
GEN_KIND = {"stigma_reinforcing": "stigma_reinforcing", "stigma_reducing": "stigma_reducing",
            "neutral_dry": "general_neutral", "general_like": "general_neutral"}
FINAL = {"stigma_reinforcing": {2.0: 2.5, 2.5: 3.0},
         "stigma_reducing": {3.0: 2.5, 2.5: 2.0},
         "neutral_dry": {2.5: 2.5, 3.0: 3.0},
         "general_like": {2.5: 2.5, 3.0: 3.0}}
TEST_PEOPLE = ("t1_sealed", "t2_sealed")

# (cell, kind, start, dev person 1, dev person 2, test person) -- plan §1.
CELLS = [
    ("C1", "stigma_reinforcing", 2.5, "p_patient", "p_office_anxiety", "t1_sealed"),
    ("C2", "stigma_reinforcing", 2.0, "h_night_warehouse", "p_grad_depression", "t2_sealed"),
    ("C3", "neutral_dry", 2.5, "p_office_anxiety", "d_line_cook", "t1_sealed"),
    ("C4", "neutral_dry", 3.0, "h_night_warehouse", "p_retail_social", "t2_sealed"),
    ("C5", "general_like", 2.5, "p_office_anxiety", "d_line_cook", "t1_sealed"),
    ("C6", "general_like", 3.0, "h_night_warehouse", "p_retail_social", "t2_sealed"),
    ("C7", "stigma_reducing", 3.0, "p_retail_social", "p_grad_depression", "t2_sealed"),
    ("C8", "stigma_reducing", 2.5, "p_patient", "d_line_cook", "t1_sealed"),
]


def items(split: str | None = None) -> list[dict]:
    """One row per dialogue: test_id, cell, kind, start, final, persona_id, split."""
    out = []
    for cell, kind, start, d1, d2, t in CELLS:
        for split_, pid in (("dev", d1), ("dev", d2), ("test", t)):
            if split and split_ != split:
                continue
            out.append({"test_id": f"{cell}_{pid}", "cell": cell, "kind": kind, "start": start,
                        "final": FINAL[kind][start], "persona_id": pid, "split": split_})
    return out


def pre_key(persona_id: str, start: float) -> str:
    return f"{persona_id}@{start}"


def pre_combos(split: str | None = None) -> list[tuple[str, float]]:
    """Distinct person@start pairs that need a frozen session-0 note."""
    seen = []
    for it in items(split):
        k = (it["persona_id"], it["start"])
        if k not in seen:
            seen.append(k)
    return seen


def is_sealed(persona_id: str) -> bool:
    return persona_id in TEST_PEOPLE


def pre_dir(persona_id: str) -> Path:
    return (SEALED / "pre") if is_sealed(persona_id) else (DEV_DIR / "pre")


def personas() -> dict:
    """Dev people from testset/personas.json, test people from SEALED (if present)."""
    import json
    out = {p["persona_id"]: p for p in json.loads((HERE / "personas.json").read_text())["personas"]}
    sealed = SEALED / "personas_test.json"
    if sealed.exists():
        out.update({p["persona_id"]: p for p in json.loads(sealed.read_text())["personas"]})
    return out


# ---------------------------------------------------------------- generation prompt patches

GENERAL_LIKE = (
    "The peer is a random person scrolling the same feed, not a counselor, not a close friend, and "
    "will not talk to this person again. Write ONE or TWO short sentences, nothing longer. A brief, "
    "ordinary sympathetic line is normal here -- the stock kind of reply strangers leave under a post "
    "(for example 'That sounds tough.', 'Hope you get some rest soon.', 'Happens to the best of us.', "
    "'That makes sense.', 'Hope tomorrow's easier.') -- vary the wording and do not reuse the same line. "
    "Do NOT comfort the patient at length, do NOT tell them how they should feel about themselves, and "
    "do NOT offer reassurance about their worth, effort, or future. Never say anything that bears on "
    "whether they are weak, unreliable, broken, lazy or less than other people, or on what struggling "
    "or asking for help says about them -- neither agree nor disagree with how they see themselves. "
    "At the same time, do NOT mock, dismiss, or minimize what they said. Do NOT offer advice, a "
    "solution, or a suggested next step unless they directly asked a factual question. Do NOT share a "
    "personal story, model self-care from lived experience, or otherwise use a peer-support move. "
    "React the way an ordinary, mildly kind stranger passing through a comment section would: brief, "
    "plain, a little warm, and not invested in changing how the patient sees themselves."
)

# v17 general_neutral's extra ban on stock sympathy (prompts.build_session_user_prompt reaction_rule).
_DRY_BAN = ("but the commenter's own comments are brief, plain, and NOT reflective-listening or "
            "validating lines (no \"that sounds hard,\" \"I hear that,\" \"that makes sense\"); see "
            "\"Peer behavior for this trajectory type\" above for what register they're in instead.")
_LIKE_RULE = ("and the commenter's own comments are one or two short sentences of ordinary stock "
              "sympathy at most; see \"Peer behavior for this trajectory type\" above for what register "
              "they're in.")

_S0_ANCHOR = "[PATIENT WRITING STYLE"

# Neutral stability (10-13, round-1 checks): with v17's "a slightly worse or slightly better day"
# Opus let setbacks pile up (caught, taken off a task, replaced), so the patient's self-stigma
# hardened into "confirmed" over the weeks -- the blind rater read 5 of 8 neutral dev dialogues as
# harsher by week 8. Both neutral kinds get this rule from session 2 on; it holds the patient's own
# events neutral the same way the commenter already is.
_NEUTRAL_OLD = ("Show the situation having moved forward a notch on its own -- a new small detail, a "
                "slightly worse or slightly better day -- without referencing or reacting to the previous "
                "commenter's reply at all.")
_NEUTRAL_NEW = ("Show the situation having moved forward a notch on its own -- a new small detail, an "
                "ordinary day that is a bit better or a bit worse -- without referencing or reacting to the "
                "previous commenter's reply at all. Nothing that happens this week settles anything about the "
                "patient: no one confirms or disproves what they believe about themselves (no being caught, "
                "found out, taken off a task, replaced, demoted, praised or reassured), and setbacks do not pile "
                "up across weeks -- ordinary ups and downs alternate. The patient's own reading of themselves at "
                "the end of the week is the same as at the start: the same claims, the same certainty, the same "
                "hedges, in new words.")
NEUTRAL_KINDS = ("neutral_dry", "general_like")
PROMPT_TAG = "v17+v3.1"   # v3.1 = neutral stability rule + targets anchored on the measured session-0 note


def s0_block(note: str) -> str:
    return ("[HOW THIS PERSON SEES THEMSELVES RIGHT NOW -- their own private note, written just before "
            "session 1. In session 1 the patient must sound like the same person holding exactly this "
            "view: no harsher and no softer. Never quote it or mention a note.]\n" + note.strip() + "\n\n")


def patch_prompt(prompt: str, kind: str, session_index: int, s0_note: str | None,
                 dry_instruction: str) -> str:
    """Apply the v3 changes to one finished v17 session prompt. Raises if an anchor moved,
    so a silent no-op cannot happen."""
    if kind == "general_like":
        for old, new in ((dry_instruction, GENERAL_LIKE), (_DRY_BAN, _LIKE_RULE)):
            if old not in prompt:
                raise RuntimeError(f"general_like patch anchor missing: {old[:60]!r}")
            prompt = prompt.replace(old, new)
    if kind in NEUTRAL_KINDS and session_index > 0:
        old = " ".join(_NEUTRAL_OLD.split())
        flat = " ".join(prompt.split())
        if old not in flat:
            raise RuntimeError("neutral stability anchor missing")
        # the v17 sentence is wrapped across source lines; match it whitespace-insensitively
        import re
        pattern = r"\s+".join(re.escape(w) for w in _NEUTRAL_OLD.split())
        prompt = re.sub(pattern, _NEUTRAL_NEW, prompt, count=1)
    if session_index == 0 and s0_note:
        if _S0_ANCHOR not in prompt:
            raise RuntimeError("session-0 note anchor missing")
        prompt = prompt.replace(_S0_ANCHOR, s0_block(s0_note) + _S0_ANCHOR, 1)
    return prompt
