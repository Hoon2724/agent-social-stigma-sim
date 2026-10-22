"""Audit logic for generated dataset items.

Two modes:
  - LLM audit (temperature 0): an auditor model checks each sample against its
    own hidden ground truth for leaks, trajectory match, gradualness, marker
    coverage, safety, and schema.
  - Heuristic audit (offline): deterministic lexical/structural checks so the
    audit path runs without an API key and as a cheap pre-filter.

The auditor is NOT the blinded judge: it is allowed to see hidden_ground_truth
because its job is to confirm the sample actually expresses that ground truth.
"""

from __future__ import annotations

import json

from schema import find_leak_terms, find_ungrounded_quotes, validate_item

AUDIT_OUTPUT_JSON_SCHEMA = {
    "name": "ismi_sample_audit",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["sample_id", "pass", "issues", "recommendation"],
        "properties": {
            "sample_id": {"type": "string"},
            "pass": {"type": "boolean"},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["severity", "type", "message", "session"],
                    "properties": {
                        "severity": {"type": "string", "enum": ["minor", "major"]},
                        "type": {
                            "type": "string",
                            "enum": [
                                "ground_truth_leak",
                                "trajectory_mismatch",
                                "abrupt_change",
                                "missing_marker",
                                "unsafe_content",
                                "schema_problem",
                                "accidental_intervention_effect",
                                "ungrounded_evidence",
                                "weak_peer_uptake",
                            ],
                        },
                        "message": {"type": "string"},
                        "session": {"type": ["integer", "null"]},
                    },
                },
            },
            "recommendation": {
                "type": "string",
                "enum": ["keep", "regenerate", "human_review"],
            },
        },
    },
}

# Lexicons for the offline heuristic.
_ABSOLUTIST = (
    "always", "never", "nothing", "no one", "nobody", "everyone", "hopeless",
    "ruined", "broken", "worthless", "can't", "cannot", "impossible", "forever",
)
_RECOVERY = (
    "maybe", "sometimes", "could", "might", "try", "part of me", "still me",
    "reach out", "realized", "realizing", "part of", "not the whole", "okay",
)
_CRISIS = ("suicide", "kill myself", "killing myself", "end my life", "self-harm", "self harm", "cut myself")

_STOPWORDS = frozenset(
    "the a an and or but if then so because that this these those i you he she it we "
    "they is are was were be been being have has had do does did will would can could "
    "should may might must to of in on at for with as by from about into over after "
    "before between during without within my your his her its our their me him them us "
    "not no yes just really very still even also".split()
)


def _content_words(text: str) -> set[str]:
    return {
        w.strip(".,!?;:\"'()").lower()
        for w in text.split()
        if len(w.strip(".,!?;:\"'()")) >= 4
    } - _STOPWORDS


def find_weak_peer_uptake(item: dict) -> list[dict]:
    """Return comment/reply turns that share NO substantive word with the turn
    they claim to respond to (reply_to). A crude lexical proxy for "did this
    turn actually react to what the other party specifically said" -- misses
    paraphrase/synonym uptake, so treated as a soft (minor) signal only."""
    hits: list[dict] = []
    for s in item.get("visible_dialogue", []):
        turns = s.get("turns", [])
        by_number = {t.get("turn"): t for t in turns}
        for t in turns:
            if t.get("kind") not in ("comment", "reply") or t.get("reply_to") is None:
                continue
            parent = by_number.get(t["reply_to"])
            if not parent:
                continue
            overlap = _content_words(t.get("text", "")) & _content_words(parent.get("text", ""))
            if not overlap:
                hits.append({"session": s.get("session"), "turn": t.get("turn"), "reply_to": t["reply_to"]})
    return hits


def _patient_text_by_session(item: dict) -> list[str]:
    out = []
    for s in item.get("visible_dialogue", []):
        text = " ".join(
            t.get("text", "").lower()
            for t in s.get("turns", [])
            if t.get("speaker") == "patient"
        )
        out.append(text)
    return out


def _stigma_score(text: str) -> int:
    return sum(text.count(w) for w in _ABSOLUTIST) - sum(text.count(w) for w in _RECOVERY)


def heuristic_audit(item: dict) -> dict:
    """Deterministic offline audit. Returns the audit-result dict."""
    sample_id = item.get("sample_id", "unknown")
    issues: list[dict] = []

    for p in validate_item(item):
        issues.append({"severity": "major", "type": "schema_problem", "message": p, "session": None})

    for hit in find_leak_terms(item):
        issues.append(
            {
                "severity": hit["severity"],
                "type": "ground_truth_leak",
                "message": f"forbidden terms {hit['terms']} in visible dialogue",
                "session": hit["session"],
            }
        )

    for hit in find_ungrounded_quotes(item):
        # minor, not major: author_evidence is analysis metadata, never seen by
        # the judge. In practice this usually means the model tidied a
        # mid-sentence clause into a standalone quote (dropped a connector word,
        # re-cased the first letter) rather than fabricating content. It should
        # be visible for cleanup, but must not force discarding an otherwise
        # good visible_dialogue via the "regenerate" recommendation.
        issues.append(
            {
                "severity": "minor",
                "type": "ungrounded_evidence",
                "message": f"author_evidence quote not found verbatim in session {hit['session']} dialogue: {hit['quote']!r}",
                "session": hit["session"],
            }
        )

    # minor: crude lexical proxy, misses genuine paraphrase/synonym uptake.
    # Flags turns that neither quote nor share any substantive word with the
    # message they reply_to -- a sign the reply may not actually be reacting
    # to the other party's specific content.
    for hit in find_weak_peer_uptake(item):
        issues.append(
            {
                "severity": "minor",
                "type": "weak_peer_uptake",
                "message": (
                    f"turn {hit['turn']} (reply_to={hit['reply_to']}) shares no "
                    f"substantive word with the turn it replies to"
                ),
                "session": hit["session"],
            }
        )

    # Safety scan across all visible text.
    for s in item.get("visible_dialogue", []):
        for t in s.get("turns", []):
            low = (t.get("text") or "").lower()
            if any(c in low for c in _CRISIS):
                issues.append(
                    {
                        "severity": "major",
                        "type": "unsafe_content",
                        "message": "possible crisis/self-harm content; needs review",
                        "session": s.get("session"),
                    }
                )

    # Trajectory direction (patient language only). We check that the direction
    # of the patient's stigma-language is CONSISTENT with the intended change,
    # with a tolerance -- not a fixed magnitude. A hard magnitude would unfairly
    # penalize reinforcing trajectories that start already-severe (ceiling: e.g.
    # 3.5 -> 3.9 has little room to rise) or short conversations.
    per_session = _patient_text_by_session(item)
    traj = item.get("condition", {}).get("trajectory_type")
    tol = 0.3
    if len(per_session) >= 3:
        third = max(1, len(per_session) // 3)
        early = sum(_stigma_score(t) for t in per_session[:third]) / third
        late = sum(_stigma_score(t) for t in per_session[-third:]) / third
        delta = late - early  # negative = patient stigma language decreased
        if traj == "stigma_reducing" and delta > tol:
            issues.append(
                {
                    "severity": "major",
                    "type": "trajectory_mismatch",
                    "message": f"stigma_reducing but patient stigma-language rose (delta={delta:.2f})",
                    "session": None,
                }
            )
        if traj == "stigma_reinforcing" and delta < -tol:
            issues.append(
                {
                    "severity": "major",
                    "type": "trajectory_mismatch",
                    "message": f"stigma_reinforcing but patient stigma-language fell (delta={delta:.2f})",
                    "session": None,
                }
            )
        if traj == "general_neutral" and delta <= -1.0:
            issues.append(
                {
                    "severity": "major",
                    "type": "accidental_intervention_effect",
                    "message": f"general_neutral but stigma-language dropped strongly (delta={delta:.2f})",
                    "session": None,
                }
            )

    markers = item.get("author_evidence", {}).get("markers_by_session", [])
    if not markers:
        issues.append(
            {"severity": "minor", "type": "missing_marker", "message": "no author_evidence markers", "session": None}
        )

    has_major = any(i["severity"] == "major" for i in issues)
    recommendation = "regenerate" if has_major else ("human_review" if issues else "keep")
    return {
        "sample_id": sample_id,
        "pass": not has_major,
        "issues": issues,
        "recommendation": recommendation,
    }


def build_audit_prompt(item: dict) -> tuple[str, str]:
    system = (
        "You are a strict dataset auditor for a synthetic internalized-stigma "
        "measurement study. You are given ONE generated sample, including its hidden "
        "ground truth. Your job is to verify the sample faithfully expresses that "
        "ground truth and is safe to use. You are NOT the blinded judge.\n\n"
        "Return JSON matching the provided schema."
    )
    payload = {
        "sample_id": item.get("sample_id"),
        "condition": item.get("condition"),
        "hidden_ground_truth": item.get("hidden_ground_truth"),
        "visible_dialogue": item.get("visible_dialogue"),
        "author_evidence": item.get("author_evidence"),
    }
    user = (
        "Audit this sample.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Audit rules:\n"
        "- visible_dialogue must NOT leak target scores, ISMI/RSES terms, or subscale "
        "names as jargon/labels (type=ground_truth_leak, MAJOR). A common word like "
        "'measurement', 'trajectory', or 'alienation' used in an ordinary, non-technical "
        "sense (e.g. 'that's not a fair measurement of who I am') is NOT a leak -- only "
        "flag it (MINOR) if it reads like it's naming the study/instrument itself.\n"
        "- stigma_reducing must show a GRADUAL reduction in the patient's self-stigma "
        "language across sessions (trajectory_mismatch / abrupt_change).\n"
        "- general_neutral must NOT show strong stigma reduction "
        "(accidental_intervention_effect).\n"
        "- stigma_reinforcing must NOT accidentally comfort or reframe too effectively "
        "(accidental_intervention_effect / trajectory_mismatch).\n"
        "- the PATIENT's language (not only the peer's) must carry the trajectory "
        "evidence (missing_marker).\n"
        "- every comment/reply must react to something SPECIFIC the other party just "
        "wrote, not a generic reflective-listening line unconnected to the prior "
        "turn's actual content (weak_peer_uptake). A programmatic lexical-overlap "
        "check has ALREADY flagged the clearest cases; you should ALSO judge real "
        "semantic uptake it cannot detect (e.g. a reply that only loosely gestures at "
        "the topic without engaging the specific point).\n"
        "- the peer must read as a lived-experience person (disclosure, direct pushback, "
        "noticing progress, or a genuine question), never a counselor giving "
        "reflective-listening validation ('that sounds hard', 'it makes sense "
        "that...') (weak_peer_uptake, minor).\n"
        "- CRITICAL: the core purpose of this dataset is dialogue where the patient's "
        "stigma trajectory is legibly DRIVEN BY the peer, not a self-narrated arc the "
        "peer merely sits beside. For each session, at least ONE peer comment must "
        "clearly and concretely enact that session's hidden 'peer_intervention_mechanism', "
        "and the patient's very next reply must visibly react to it (agree, resist, ask, "
        "or act on it). If a session shows the patient's language shifting with no peer "
        "comment that plausibly caused it, flag weak_peer_uptake as MAJOR for that "
        "session, not minor -- this defeats the dataset's purpose.\n"
        "- the patient must not introduce a brand-new unrelated life event every turn; "
        "one problem thread should visibly carry through the session.\n"
        "- CRITICAL: session 2+ must show real continuity with the PREVIOUS session's "
        "peer comments -- but this does NOT require a literal quote every time. Any of "
        "these count: (a) a direct reference/quote of something the peer actually said, "
        "(b) reporting an action taken or resisted because of the previous session, or "
        "(c) a situational update where the previous session's influence is clearly "
        "implicit even without being stated. What FAILS this check: (i) the opening "
        "shows no plausible connection to the previous session at all (patient just "
        "started a fresh, unrelated topic), or (ii) it DOES make a specific claim about "
        "what the peer said/did, but that detail was never actually there -- check the "
        "previous session's peer turns yourself. Either failure fakes or erases "
        "continuity and defeats this dataset's purpose just as much as a fabricated "
        "author_evidence quote. Flag as trajectory_mismatch, MAJOR. Also flag (MINOR) if "
        "every session in this sample opens with the same literal-quote formula (e.g. "
        "always \"I kept thinking about what you said...\") -- that reads as repetitive "
        "and unnatural, not like real SNS posts.\n"
        "- every author_evidence quote must be an exact, verbatim substring of that "
        "session's visible_dialogue text (ungrounded_evidence). This check has ALSO "
        "been run programmatically. Treat a trimmed clause or re-cased first letter "
        "(the model tidying a mid-sentence fragment into a standalone quote) as "
        "MINOR severity -- it does not affect visible_dialogue and should not by "
        "itself justify recommendation=regenerate. Treat a quote that misrepresents "
        "or invents content not actually said as MAJOR.\n"
        "- no real identifying details; no unsafe crisis content (unsafe_content).\n"
        "- structure must match the schema (schema_problem).\n\n"
        "Set pass=false if there is any major issue. recommendation is one of "
        "keep | regenerate | human_review."
    )
    return system, user


def llm_audit(item: dict, client) -> dict:
    system, user = build_audit_prompt(item)
    result = client.complete_json(
        system,
        user,
        AUDIT_OUTPUT_JSON_SCHEMA,
        temperature=0.0,
        max_output_tokens=2000,
    )
    result.setdefault("sample_id", item.get("sample_id", "unknown"))
    return result
