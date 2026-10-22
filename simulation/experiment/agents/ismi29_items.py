"""Official ISMI-29 item text and scoring — MEASUREMENT ONLY.

Ported verbatim from a co-author's earlier code, simulation/ismi_controlled_sanity/scale_items.py.
Item wording, administration order, subscale grouping, and reverse-scoring flags
are unchanged from that file; see its header for the cross-verified sources
(VA VISN 5 MIRECC handout reproducing the developer's copy; Ritsher, Otilingam
& Grajales 2003, Psychiatry Research 121(1), 31-49).

*** DO NOT IMPORT THIS MODULE FROM ANY PERSONA-GENERATION CODE. ***

The personas being measured must never see, reference, or be written against
these items. If the item text reaches the generation side, a measured ISMI
score stops being evidence about the persona and becomes an echo of the
questionnaire -- the persona would have been written to answer it. Only
measure_ismi29.py may import this module; verification greps for exactly that.

All items use a 1 (strongly disagree) - 4 (strongly agree) response scale.
Reverse-scored items are scored 5 - raw before averaging, so a higher final
score always means MORE internalized stigma. Stigma Resistance is the sole
reverse-scored subscale.

Standard interpretation cutoffs for the total mean (used to label a persona's
stigma tier, never shown to the persona):
    <= 2.00          minimal
    2.00 < x <= 2.50 mild
    2.50 < x <= 3.00 moderate
    > 3.00           severe
"""

from __future__ import annotations

ISMI_SUBSCALES = (
    "alienation",
    "stereotype_endorsement",
    "discrimination_experience",
    "social_withdrawal",
    "stigma_resistance",
)

# id = original 1-29 administration order from the official questionnaire.
# subscale + reverse per Table 2 grouping (Stigma Resistance is the sole
# reverse-scored subscale).
ISMI_29_ITEMS = [
    {"id": 1, "text": "I feel out of place in the world because I have a mental illness.", "subscale": "alienation", "reverse": False},
    {"id": 2, "text": "Mentally ill people tend to be violent.", "subscale": "stereotype_endorsement", "reverse": False},
    {"id": 3, "text": "People discriminate against me because I have a mental illness.", "subscale": "discrimination_experience", "reverse": False},
    {"id": 4, "text": "I avoid getting close to people who don't have a mental illness to avoid rejection.", "subscale": "social_withdrawal", "reverse": False},
    {"id": 5, "text": "I am embarrassed or ashamed that I have a mental illness.", "subscale": "alienation", "reverse": False},
    {"id": 6, "text": "Mentally ill people shouldn't get married.", "subscale": "stereotype_endorsement", "reverse": False},
    {"id": 7, "text": "People with mental illness make important contributions to society.", "subscale": "stigma_resistance", "reverse": True},
    {"id": 8, "text": "I feel inferior to others who don't have a mental illness.", "subscale": "alienation", "reverse": False},
    {"id": 9, "text": "I don't socialize as much as I used to because my mental illness might make me look or behave “weird.”", "subscale": "social_withdrawal", "reverse": False},
    {"id": 10, "text": "People with mental illness cannot live a good, rewarding life.", "subscale": "stereotype_endorsement", "reverse": False},
    {"id": 11, "text": "I don't talk about myself much because I don't want to burden others with my mental illness.", "subscale": "social_withdrawal", "reverse": False},
    {"id": 12, "text": "Negative stereotypes about mental illness keep me isolated from the “normal” world.", "subscale": "social_withdrawal", "reverse": False},
    {"id": 13, "text": "Being around people who don't have a mental illness makes me feel out of place or inadequate.", "subscale": "social_withdrawal", "reverse": False},
    {"id": 14, "text": "I feel comfortable being seen in public with an obviously mentally ill person.", "subscale": "stigma_resistance", "reverse": True},
    {"id": 15, "text": "People often patronize me, or treat me like a child, just because I have a mental illness.", "subscale": "discrimination_experience", "reverse": False},
    {"id": 16, "text": "I am disappointed in myself for having a mental illness.", "subscale": "alienation", "reverse": False},
    {"id": 17, "text": "Having a mental illness has spoiled my life.", "subscale": "alienation", "reverse": False},
    {"id": 18, "text": "People can tell that I have a mental illness by the way I look.", "subscale": "stereotype_endorsement", "reverse": False},
    {"id": 19, "text": "Because I have a mental illness, I need others to make most decisions for me.", "subscale": "stereotype_endorsement", "reverse": False},
    {"id": 20, "text": "I stay away from social situations in order to protect my family or friends from embarrassment.", "subscale": "social_withdrawal", "reverse": False},
    {"id": 21, "text": "People without mental illness could not possibly understand me.", "subscale": "alienation", "reverse": False},
    {"id": 22, "text": "People ignore me or take me less seriously just because I have a mental illness.", "subscale": "discrimination_experience", "reverse": False},
    {"id": 23, "text": "I can't contribute anything to society because I have a mental illness.", "subscale": "stereotype_endorsement", "reverse": False},
    {"id": 24, "text": "Living with mental illness has made me a tough survivor.", "subscale": "stigma_resistance", "reverse": True},
    {"id": 25, "text": "Nobody would be interested in getting close to me because I have a mental illness.", "subscale": "discrimination_experience", "reverse": False},
    {"id": 26, "text": "In general, I am able to live my life the way I want to.", "subscale": "stigma_resistance", "reverse": True},
    {"id": 27, "text": "I can have a good, fulfilling life, despite my mental illness.", "subscale": "stigma_resistance", "reverse": True},
    {"id": 28, "text": "Others think that I can't achieve much in life because I have a mental illness.", "subscale": "discrimination_experience", "reverse": False},
    {"id": 29, "text": "Stereotypes about the mentally ill apply to me.", "subscale": "stereotype_endorsement", "reverse": False},
]


def score_ismi_29(raw_scores: dict) -> dict:
    """raw_scores: {item_id (1-29): 1-4 raw response}. Returns per-item scored
    values, per-subscale means, and the overall mean (higher = more stigma)."""
    scored: dict[int, int] = {}
    by_subscale: dict[str, list[int]] = {s: [] for s in ISMI_SUBSCALES}
    for item in ISMI_29_ITEMS:
        raw = int(raw_scores[item["id"]])
        if raw not in (1, 2, 3, 4):
            raise ValueError(f"ISMI item {item['id']} score out of range: {raw}")
        val = 5 - raw if item["reverse"] else raw
        scored[item["id"]] = val
        by_subscale[item["subscale"]].append(val)
    subscale_means = {s: round(sum(v) / len(v), 3) for s, v in by_subscale.items()}
    overall_mean = round(sum(scored.values()) / len(scored), 3)
    return {"items_scored": scored, "subscale_means": subscale_means, "total_mean": overall_mean}


# Plan/verification-facing name; score_ismi_29 keeps the source's spelling so a
# diff against that original stays clean.
score_ismi29 = score_ismi_29
