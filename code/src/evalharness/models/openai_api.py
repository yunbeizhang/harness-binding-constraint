"""OpenAI Responses API adapter.

NOTE on determinism:
    GPT-5 series (gpt-5, gpt-5.x including gpt-5.4) does NOT accept the
    `temperature` parameter on the Responses API — passing any value returns
    a 400 error. Temperature is fixed at the model's internal default (~1.0)
    and cannot be set to 0. This is because GPT-5 is a reasoning model that
    runs multiple internal generation/verification rounds; exposing
    temperature would break those calibrations.

    Consequence: every call to GPT-5.4 has inherent nondeterminism, even
    when we want reproducible behavior. The `seed` parameter may help
    ("best effort" per OpenAI docs) but is not guaranteed on the Responses
    API. Do NOT add a `temperature` field to this adapter for GPT-5 models.

    Kimi and GLM/ZAI are configured in their respective chat-completions
    adapters; GPT-5.4 remains temperature-less here by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..core import ModelResponse, UsageStats
from .http import post_json, resolve_api_key


@dataclass
class OpenAIResponsesModel:
    """Minimal OpenAI adapter using the official Responses API."""

    name: str
    model_id: str
    api_key_env: str = "OPENAI_API_KEY"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    timeout_sec: int = 180
    reasoning_effort: Optional[str] = None
    verbosity: Optional[str] = None
    default_max_output_tokens: int = 4_096

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> ModelResponse:
        input_items: list[dict[str, Any]] = []
        if system_prompt:
            input_items.append(
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": system_prompt}],
                }
            )
        input_items.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        )

        payload: dict[str, Any] = {
            "model": self.model_id,
            "input": input_items,
            "max_output_tokens": max_output_tokens or self.default_max_output_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.verbosity:
            payload["text"] = {"verbosity": self.verbosity}

        data = post_json(
            url=f"{self.base_url.rstrip('/')}/responses",
            headers={
                "Authorization": f"Bearer {resolve_api_key(api_key=self.api_key, api_key_env=self.api_key_env)}",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout_sec=self.timeout_sec,
        )
        text = _extract_openai_text(data)
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        reasoning_tokens = usage.get("reasoning_tokens", 0)
        # OpenAI reports cached tokens inside input_tokens_details
        cached_tokens = 0
        input_details = usage.get("input_tokens_details") or {}
        cached_tokens = input_details.get("cached_tokens", 0)
        # Cost: $2.50/1M input, $1.25/1M cached, $15.00/1M output for gpt-5.4
        uncached = (input_tokens or 0) - (cached_tokens or 0)
        cost = uncached * 2.50 / 1e6 + (cached_tokens or 0) * 1.25 / 1e6 + (output_tokens or 0) * 15.00 / 1e6
        return ModelResponse(
            text=text,
            raw=data,
            usage=UsageStats(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                cached_tokens=cached_tokens,
                cost=round(cost, 6),
            ),
            stop_reason=data.get("status"),
        )


def _extract_openai_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str) and data["output_text"]:
        return data["output_text"]
    pieces: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                pieces.append(text)
    return "\n".join(pieces).strip()
