"""Offline mock generator.

Produces a schema-valid `{visible_dialogue, author_evidence}` object WITHOUT any
API call, driven by the ground-truth session targets. It is intentionally
formulaic (not realistic prose); its only purpose is to verify file writing,
schema conformance, and the audit path offline. Samples produced this way are
tagged model="mock" by the pipeline.
"""

from __future__ import annotations



def _severity(target_ismi: float) -> float:
    """Map a 1-4 ISMI target to a 0-1 stigma severity (the mock's phrase banks are keyed by band)."""
    return max(0.0, min(1.0, (target_ismi - 1.0) / 3.0))


def _band(severity: float) -> int:
    """3-level band: 2 = high stigma, 1 = mid, 0 = low stigma."""
    if severity >= 0.58:
        return 2
    if severity >= 0.42:
        return 1
    return 0

# Patient utterance banks by stigma band (2=high, 1=mid, 0=low). Deliberately
# free of any measurement vocabulary.
_PATIENT_LINES = {
    2: [
        "honestly i feel like something is just wrong with me and i don't belong anywhere.",
        "it's like this is who i am now, and it's never going to change.",
        "i can't really do the things other people do, so i mostly stay out of the way.",
    ],
    1: [
        "some days are heavy but maybe it isn't every single time... i've been noticing that.",
        "i keep wondering if other people feel a bit like this too, honestly.",
        "part of me wants to say something to a friend, i just haven't yet.",
    ],
    0: [
        "i've been realizing this is one part of me, not the whole story of who i am.",
        "i think i could actually try reaching out more, and that feels okay.",
        "looking back i've handled hard stretches before, so maybe ahead of me isn't all bad.",
    ],
}

_PEER_LINES = {
    "stigma_reducing": [
        "that sounds really hard, and honestly a lot of people feel that way too.",
        "thank you for saying that. what's one small thing that's felt even a little lighter?",
        "you're describing this really thoughtfully. that takes something.",
    ],
    "general_neutral": [
        "ah that's rough. anyway, did you catch that show everyone's been posting about?",
        "yeah i get busy weeks like that. what are you up to this weekend?",
        "totally. oh, the weather's been wild lately huh.",
    ],
    "stigma_reinforcing": [
        "aw, i'm so sorry. maybe it's safer to keep that stuff to yourself so people don't worry.",
        "yeah, it's probably just easier not to expect too much for now.",
        "i feel bad for you, honestly. some people just have a harder lot with this.",
    ],
}


def build_mock_output(
    condition: dict, persona: dict, session_targets: list[dict]
) -> dict:
    trajectory_type = condition["trajectory_type"]
    turns_per_session = condition["turns_per_session"]
    peer_bank = _PEER_LINES[trajectory_type]

    visible_dialogue = []
    markers_by_session = []

    for si, target in enumerate(session_targets):
        band = _band(_severity(target["target_ismi"]))
        patient_variants = _PATIENT_LINES[band]
        turns = []
        first_patient_quote = None
        for t in range(turns_per_session):
            turn_no = t + 1
            if t == 0:
                text = patient_variants[si % len(patient_variants)]
                first_patient_quote = text
                turns.append(
                    {"turn": turn_no, "speaker": "patient", "kind": "post", "reply_to": None, "text": text}
                )
            elif t % 2 == 1:
                text = peer_bank[(si + t) % len(peer_bank)]
                turns.append(
                    {"turn": turn_no, "speaker": "peer", "kind": "comment", "reply_to": turn_no - 1, "text": text}
                )
            else:
                text = patient_variants[(si + t) % len(patient_variants)]
                turns.append(
                    {"turn": turn_no, "speaker": "patient", "kind": "reply", "reply_to": turn_no - 1, "text": text}
                )
        visible_dialogue.append({"session": target["session"], "turns": turns})

        markers_by_session.append(
            {
                "session": target["session"],
                "evidence": [
                    {
                        "marker": f"identity_fusion={({2: 'high', 1: 'moderate', 0: 'low'})[band]}",
                        "quote": first_patient_quote,
                        "explanation": (
                            "mock line selected for stigma band "
                            f"{band} (target_ismi={target['target_ismi']})"
                        ),
                    }
                ],
            }
        )

    return {
        "visible_dialogue": visible_dialogue,
        "author_evidence": {
            "summary": (
                "MOCK sample. Patient lines were selected from a fixed bank keyed to the "
                "session stigma band; peer lines match the trajectory type. Not realistic "
                "prose -- for offline schema/pipeline validation only."
            ),
            "markers_by_session": markers_by_session,
        },
    }
