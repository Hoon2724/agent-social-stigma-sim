"""Strict Minibook transport for the main experiment's social interaction loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from io_utils import read_json


@dataclass(frozen=True)
class AgentCredential:
    name: str
    api_key: str


@dataclass
class MinibookThreadStore:
    """Persist one experimental trajectory as one Minibook post per session."""

    base_url: str
    project_id: str
    agents: dict[str, AgentCredential]
    trajectory_id: str
    timeout: float = 30.0
    post_ids: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path, trajectory_id: str) -> "MinibookThreadStore":
        raw = read_json(path)
        required = {"patient", "peer", "general"}
        supplied = set(raw.get("agents", {}))
        missing = sorted(required - supplied)
        if missing:
            raise ValueError(f"Minibook config is missing agents: {', '.join(missing)}")
        agents = {
            role: AgentCredential(name=str(value["name"]), api_key=str(value["api_key"]))
            for role, value in raw["agents"].items()
        }
        return cls(
            base_url=str(raw.get("base_url", "http://localhost:3456/api/v1")).rstrip("/"),
            project_id=str(raw["project_id"]),
            agents=agents,
            trajectory_id=trajectory_id,
            timeout=float(raw.get("timeout", 30.0)),
        )

    def _headers(self, role: str) -> dict[str, str]:
        credential = self.agents[role]
        return {
            "Authorization": f"Bearer {credential.api_key}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, *, role: str | None = None,
                 json: dict | None = None) -> Any:
        headers = self._headers(role) if role else None
        response = requests.request(
            method,
            f"{self.base_url}/{path.lstrip('/')}",
            headers=headers,
            json=json,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:500]
            raise RuntimeError(
                f"Minibook {method} {path} failed ({response.status_code}): {detail}"
            ) from exc
        return response.json()

    def start_session(self, session: int, content: str) -> str:
        if session in self.post_ids:
            raise RuntimeError(f"Minibook session {session} already has a post")
        result = self._request(
            "POST",
            f"projects/{self.project_id}/posts",
            role="patient",
            json={
                "title": f"{self.trajectory_id} / session {session}",
                "content": content,
                "type": "discussion",
                "tags": ["main-experiment", self.trajectory_id],
            },
        )
        self.post_ids[session] = str(result["id"])
        return str(result["content"])

    def add_comment(self, session: int, role: str, content: str) -> str:
        post_id = self.post_ids.get(session)
        if not post_id:
            raise RuntimeError(f"Minibook session {session} has no opening post")
        result = self._request(
            "POST",
            f"posts/{post_id}/comments",
            role=role,
            json={"content": content},
        )
        return str(result["content"])

    def read_session(self, session: int) -> list[dict]:
        post_id = self.post_ids.get(session)
        if not post_id:
            return []
        posts = self._request("GET", f"projects/{self.project_id}/posts")
        post = next((item for item in posts if str(item.get("id")) == post_id), None)
        if post is None:
            raise RuntimeError(f"Minibook post {post_id} is no longer visible")
        comments = self._request("GET", f"posts/{post_id}/comments")
        name_to_role = {credential.name: role for role, credential in self.agents.items()}
        turns = [{"speaker": "patient", "content": str(post["content"])}]
        for comment in comments:
            author = str(comment.get("author_name", ""))
            if author not in name_to_role:
                raise RuntimeError(
                    f"Unexpected Minibook participant {author!r} in post {post_id}"
                )
            turns.append({
                "speaker": name_to_role[author],
                "content": str(comment["content"]),
            })
        return turns

    def sync_context(self, local_turns: list[dict], session: int) -> list[dict]:
        """Replace the current session with the thread read back from Minibook."""
        remote = self.read_session(session)
        current = [turn for turn in local_turns if int(turn.get("session", 1)) == session]
        if len(remote) != len(current):
            raise RuntimeError(
                f"Minibook/local turn mismatch in session {session}: "
                f"remote={len(remote)}, local={len(current)}"
            )
        synced = []
        for local, persisted in zip(current, remote):
            expected = "general" if local["speaker"] == "general" else local["speaker"]
            if persisted["speaker"] != expected:
                raise RuntimeError(
                    f"Minibook/local speaker mismatch in session {session}: "
                    f"remote={persisted['speaker']}, local={expected}"
                )
            synced.append({**local, "content": persisted["content"]})
        prior = [turn for turn in local_turns if int(turn.get("session", 1)) != session]
        return prior + synced

    def public_metadata(self) -> dict:
        return {
            "base_url": self.base_url,
            "project_id": self.project_id,
            "trajectory_id": self.trajectory_id,
            "post_ids": {str(k): v for k, v in sorted(self.post_ids.items())},
        }
