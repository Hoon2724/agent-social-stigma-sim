"""Small OpenAI-compatible client used only by this experiment."""

from __future__ import annotations

import os
from typing import Protocol


class ChatModel(Protocol):
    def complete(self, messages: list[dict], temperature: float) -> str:
        """Return one assistant message."""


class OpenAICompatibleModel:
    def __init__(self, model: str, base_url: str | None = None,
                 num_ctx: int | None = None, num_gpu: int | None = None) -> None:
        from openai import OpenAI

        self.model = model
        # Per-request context window for local models. The ollama server sets a
        # global OLLAMA_CONTEXT_LENGTH sized for the longest job it runs, and
        # multiplies it by OLLAMA_NUM_PARALLEL slots of KV cache. For a large
        # model that cache competes with the weights for VRAM and can push
        # layers onto the CPU — the GPU then idles at low power. Measurement
        # jobs whose prompts are far shorter than the global setting should pass
        # a smaller num_ctx so the whole model stays resident. Left as None the
        # server default applies, so existing callers are unaffected.
        self.num_ctx = num_ctx
        # ollama's automatic GPU/CPU layer split was observed to be flaky even
        # with a small num_ctx: repeated identical reloads landed some
        # endpoints fully resident and others ~2.97GB-on-GPU/rest-on-CPU, with
        # no consistent pattern across which endpoint failed (2026-08-02).
        # num_gpu=99 (more than qwen3:32b has layers) forces every layer onto
        # GPU, skipping that heuristic entirely -- reliable where num_ctx sizing
        # alone was not. Only applied via the native path (see complete()).
        self.num_gpu = num_gpu
        if model.startswith("gpt-"):
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self.is_local = False
        else:
            endpoint = base_url or os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434/v1"
            )
            self.client = OpenAI(base_url=endpoint, api_key="ollama", timeout=600.0)
            self.is_local = True

    def complete(self, messages: list[dict], temperature: float) -> str:
        if self.is_local and (self.num_ctx is not None or self.num_gpu is not None):
            return self._complete_native(messages, temperature)
        kwargs = {}
        if self.is_local:
            kwargs["extra_body"] = {"think": False}
        else:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content.strip()

    def _complete_native(self, messages: list[dict], temperature: float) -> str:
        """Call ollama's native /api/chat so options.num_ctx is actually applied.

        The OpenAI-compatible /v1/chat/completions layer silently drops the
        options block, so a model loaded through it always uses the server-wide
        OLLAMA_CONTEXT_LENGTH. That matters for large models: the KV cache is
        sized from that value and counts against VRAM alongside the weights, so
        an oversized context can push layers onto the CPU. Sending num_ctx here
        keeps qwen3:32b fully resident (22.6GB at num_ctx=2048 vs 29.2GB at the
        8192 default). Same request/response shape as merged/ismi_direct_from_belief.py.
        """
        import requests

        base = str(self.client.base_url).rstrip("/")
        url = base[: -len("/v1")] + "/api/chat" if base.endswith("/v1") else base + "/api/chat"
        options = {"temperature": temperature}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        response = requests.post(
            url,
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": options,
            },
            timeout=600.0,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
