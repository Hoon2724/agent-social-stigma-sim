"""Transparent local conversation memory; never used by the post-hoc judge."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from embedding_utils import embed_text, top_n_by_cosine


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9가-힣]{2,}", text.lower()))


@dataclass
class ConversationMemory:
    """Short-term = this session so far + the previous `recent_sessions`
    sessions in full. Long-term = everything older, reached only by retrieve().

    Three channels carry context and each should do a different job: recent()
    is the conversation happening now, the inner-state checkpoint (belief.py)
    is the accumulated self-description, and retrieve() reaches back for older
    relevant turns. The two dialogue channels are a strict partition (_split),
    so a turn is in exactly one of them.

    Why sessions rather than a turn count (changed 2026-08-02): the window used
    to be a fixed 11 turns, which was exactly one session of the C2 config it
    was tuned on (1 opening + 2 rounds x (3 bystanders + peer + patient)). That
    number silently stops meaning "one session" the moment the thread shape
    changes -- a 1:1 condition is 5 turns per session, so 11 spanned 2.2
    sessions and fed the patient the whole previous session verbatim on every
    turn. It also was not condition-neutral: C1 (no peer) runs 9 turns per
    session against C2's 11, so a fixed 11 gave the C1 patient ~20% more past
    dialogue than the C2 patient, mixing memory depth into the very comparison
    the study is about. A session-scoped window is the same size in every
    condition by construction.

    An earlier note here warned that widening the window to 22 made the patient
    restate session 1's phrasing throughout session 2. That observation
    (2026-07-24) predates quality.classify_template_repeat / revise_off_template
    (2026-07-29), which exist to catch exactly that and which only see what is
    inside this window in the first place (patient_agent.respond builds
    recent_own_texts from recent()). It is kept as a caution, not as a
    constraint: verify on a pilot run before a large batch.
    """

    # How many PREVIOUS sessions stay in short-term memory alongside the
    # current one. 1 = "last week's conversation is still fresh".
    recent_sessions: int = 1
    # Legacy escape hatch: when set, short-term is the last N turns regardless
    # of session boundaries (the pre-2026-08-02 behavior). Kept so an older run
    # can be reproduced exactly; leave as None for the session-scoped default.
    recent_window: int | None = None
    turns: list[dict] = field(default_factory=list)
    current_session: int = 1

    def add(self, speaker: str, content: str, round_number: int, metadata: dict | None = None) -> None:
        text = content.strip()
        self.turns.append({
            "speaker": speaker,
            "content": text,
            "round": round_number,
            "session": self.current_session,
            "metadata": metadata or {},
            "embedding": embed_text(text),
        })

    def _ensure_embedding(self, turn: dict) -> list[float]:
        """Backfill embeddings for older saved runs loaded before this field existed."""
        embedding = turn.get("embedding")
        if isinstance(embedding, list) and embedding:
            return embedding
        embedding = embed_text(str(turn.get("content", "")))
        turn["embedding"] = embedding
        return embedding

    def _split(self) -> tuple[list[dict], list[dict]]:
        """(short-term, older) — a partition of self.turns.

        Returned as one operation so recent() and retrieve() can never overlap
        or leave a gap between them, whichever windowing mode is active.
        """
        if self.recent_window is not None:
            if len(self.turns) <= self.recent_window:
                return list(self.turns), []
            return self.turns[-self.recent_window:], self.turns[:-self.recent_window]
        cutoff = self.current_session - self.recent_sessions
        short, older = [], []
        for turn in self.turns:
            (short if int(turn.get("session", 1)) >= cutoff else older).append(turn)
        return short, older

    def recent(self) -> list[dict]:
        """Short-term dialogue, with a synthetic narrator line inserted at each
        session boundary so the model doesn't see two consecutive same-speaker
        turns straddling a session gap and mistake the new session's opening
        for a continuation of its own prior line."""
        window, _ = self._split()
        out: list[dict] = []
        prev_session = None
        for t in window:
            session = t.get("session", 1)
            if prev_session is not None and session != prev_session:
                out.append({"speaker": "narrator", "content": "(About a week passes.)"})
            out.append({
                "speaker": t["speaker"],
                "content": t["content"],
                # Carried so an agent can narrow this window further on its own
                # (GeneralAgent keeps only the current session — a bystander
                # has no memory of previous weeks, but does see the thread it
                # is replying into).
                "session": session,
                # Which bystander this was. Several people reply to the same
                # post, and they all carry speaker "general", so without this
                # the reader sees one stranger talking repeatedly.
                "commenter_id": (t.get("metadata") or {}).get("commenter_id"),
            })
            prev_session = session
        return out

    def retrieve(self, query: str, viewer: str, top_k: int = 3, min_overlap: int = 2) -> str:
        """Long-term recall: the sessions that have fallen out of short-term.

        Retrieval uses sentence-transformers/all-MiniLM-L6-v2 embeddings and
        cosine similarity over normalized vectors. The min_overlap argument is
        kept for API compatibility with the previous token-overlap retrieval
        and is intentionally ignored.

        Empty for the first `recent_sessions + 1` sessions of a run, which is
        correct rather than a failure — nothing has aged out yet.
        """
        _, older = self._split()
        if not older or top_k <= 0:
            return ""
        query_embedding = embed_text(query)
        candidates = []
        for index, turn in enumerate(older):
            candidates.append((index, self._ensure_embedding(turn), turn))
        selected = [item[2] for item in top_n_by_cosine(query_embedding, candidates, top_k)]
        return "\n".join(
            f"{'SELF' if t['speaker'] == viewer else 'PARTNER'} (earlier): {t['content']}"
            for t in selected
        )

    def public_turns(self) -> list[dict]:
        return [dict(turn) for turn in self.turns]
