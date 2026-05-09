"""Z.AI OpenAI-compatible Chat Completions adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..core import ModelResponse, UsageStats
from .http import post_json, resolve_api_key


@dataclass
class ZaiChatModel:
    """Adapter for Z.AI's OpenAI-compatible `/chat/completions` API."""

    name: str
    model_id: str
    api_key_env: str = "ZAI_API_KEY"
    api_key: Optional[str] = None
    base_url: str = "https://api.z.ai/api/paas/v4"
    timeout_sec: int = 180
    default_max_output_tokens: int = 4_096
    thinking_mode: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> ModelResponse:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_output_tokens or self.default_max_output_tokens,
        }
        if self.thinking_mode:
            payload["thinking"] = {"type": self.thinking_mode}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p

        data = post_json(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {resolve_api_key(api_key=self.api_key, api_key_env=self.api_key_env)}",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout_sec=self.timeout_sec,
        )
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = _extract_zai_text(message.get("content", ""))
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        prompt_details = usage.get("prompt_tokens_details", {}) or {}
        completion_details = usage.get("completion_tokens_details", {}) or {}
        cached_tokens = (
            usage.get("cached_tokens")
            or usage.get("cache_hit_tokens")
            or usage.get("prompt_cache_hit_tokens")
            or prompt_details.get("cached_tokens")
            or prompt_details.get("cache_hit_tokens")
            or 0
        )
        reasoning_tokens = completion_details.get("reasoning_tokens")
        uncached_tokens = max((input_tokens or 0) - cached_tokens, 0)
        # GLM-5.1 official pricing: input $1.40/MTok, cached input
        # $0.26/MTok, output $4.40/MTok.
        cost = (
            uncached_tokens * 1.40 / 1e6
            + cached_tokens * 0.26 / 1e6
            + (output_tokens or 0) * 4.40 / 1e6
        )
        return ModelResponse(
            text=text,
            raw=data,
            usage=UsageStats(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                cached_tokens=cached_tokens,
                cost=cost,
            ),
            stop_reason=choice.get("finish_reason"),
        )


def _extract_zai_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                pieces.append(part.get("text", ""))
        return "\n".join(pieces).strip()
    return str(content).strip()
