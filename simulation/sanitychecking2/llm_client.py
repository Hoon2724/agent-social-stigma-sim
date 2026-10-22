"""Thin OpenAI Structured Outputs client for generation and audit calls.

Defaults to the OpenAI API for gpt-* models. Any non-gpt model name is routed to
an OpenAI-compatible endpoint (e.g. a local Ollama server) so the later judge or
a cross-family auditor can reuse this client. Secrets come only from the
OPENAI_API_KEY environment variable.

Newer OpenAI models restrict some sampling params (temperature must be default,
max_tokens renamed to max_completion_tokens). This client sends the modern
params and gracefully retries without a param the model rejects, so callers can
keep a single interface across model generations.
"""

from __future__ import annotations

import json
import os
from typing import Any


class StructuredClient:
    def __init__(self, model: str, base_url: str | None = None) -> None:
        from openai import OpenAI

        self.model = model
        self.is_local = not (model.startswith("gpt-") or model.startswith("o"))
        if not self.is_local:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        else:
            endpoint = base_url or os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11436/v1"
            )
            # Local judge calls (39 scale items) are large; give Ollama room to
            # load/serve without the default client timeout tripping.
            import httpx

            self.client = OpenAI(
                base_url=endpoint, api_key="ollama",
                timeout=httpx.Timeout(600.0, connect=60.0),
            )

    def complete_json(
        self,
        system: str,
        user: str,
        json_schema: dict,
        temperature: float = 0.7,
        max_output_tokens: int = 8000,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Return parsed JSON validated against `json_schema` (strict)."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        base_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": json_schema},
            "max_completion_tokens": max_output_tokens,
        }
        # qwen3-family Ollama models default to a slow "thinking" mode that
        # massively inflates latency on large item lists (39-item judge
        # schema went from timing out at 600s to completing quickly with this
        # off). Not relevant to the OpenAI API, so only set for local models.
        if self.is_local:
            base_kwargs["extra_body"] = {"think": False}

        # Optional params that some models reject; dropped on demand below.
        optional = {"temperature": temperature}
        if seed is not None:
            optional["seed"] = seed

        # Long unattended sweeps hit occasional transient failures (server
        # 500s, empty content on a dropped/malformed response). Retry a
        # couple of times before giving up, instead of killing the whole
        # multi-hour run on one bad call.
        last_exc: Exception | None = None
        for attempt in range(3):
            content = self._create_with_fallback(base_kwargs, optional)
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                last_exc = exc
                print(f"complete_json: empty/invalid response on attempt {attempt + 1}/3 "
                      f"(content={content!r:.200}), retrying", flush=True)
        raise last_exc

    def complete_text(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_output_tokens: int = 2000,
        seed: int | None = None,
    ) -> str:
        """Plain free-text completion (no JSON schema). Used for Stage-1
        reflection generation, which produces a short first-person note rather
        than structured output. Same client, param-fallback, and retry policy
        as complete_json."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        base_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_output_tokens,
        }
        if self.is_local:
            base_kwargs["extra_body"] = {"think": False}

        optional = {"temperature": temperature}
        if seed is not None:
            optional["seed"] = seed

        content = self._create_with_fallback(base_kwargs, optional)
        return (content or "").strip()

    def _create_with_fallback(
        self, base_kwargs: dict[str, Any], optional: dict[str, Any]
    ) -> str:
        """Call the API, retrying without any single param the model rejects."""
        attempt_optional = dict(optional)
        while True:
            try:
                resp = self.client.chat.completions.create(
                    **base_kwargs, **attempt_optional
                )
                return resp.choices[0].message.content
            except Exception as exc:  # openai.BadRequestError et al.
                msg = str(exc).lower()
                dropped = None
                for param in list(attempt_optional.keys()):
                    if param in msg and (
                        "unsupported" in msg
                        or "not support" in msg
                        or "must be" in msg
                        or "invalid" in msg
                        or "unknown" in msg
                    ):
                        dropped = param
                        break
                if dropped is None:
                    raise
                attempt_optional.pop(dropped)
