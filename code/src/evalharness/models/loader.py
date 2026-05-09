"""Config-driven model loader."""

from __future__ import annotations

from pathlib import Path

from ..core import Model
from ..utils import load_mapping_file
from .kimi_api import KimiChatModel
from .openai_api import OpenAIResponsesModel
from .zai_api import ZaiChatModel


def load_models_from_config(path: str | Path) -> dict[str, Model]:
    """Load model adapters from a TOML or JSON config file."""
    data = load_mapping_file(path)
    model_configs = data.get("models", data)
    models: dict[str, Model] = {}
    for label, cfg in model_configs.items():
        provider = cfg["provider"].lower()
        if provider == "openai":
            models[label] = OpenAIResponsesModel(
                name=label,
                model_id=cfg["model"],
                api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
                api_key=cfg.get("api_key"),
                base_url=cfg.get("base_url", "https://api.openai.com/v1"),
                timeout_sec=int(cfg.get("timeout_sec", 180)),
                reasoning_effort=cfg.get("reasoning_effort"),
                verbosity=cfg.get("verbosity"),
                default_max_output_tokens=int(cfg.get("max_output_tokens", 4096)),
            )
        elif provider == "kimi":
            thinking_mode = cfg.get("thinking_mode")
            models[label] = KimiChatModel(
                name=label,
                model_id=cfg["model"],
                api_key_env=cfg.get("api_key_env", "KIMI_API_KEY"),
                api_key=cfg.get("api_key"),
                base_url=cfg.get("base_url", "https://api.moonshot.ai/v1"),
                timeout_sec=int(cfg.get("timeout_sec", 180)),
                default_max_output_tokens=int(cfg.get("max_output_tokens", 4096)),
                thinking_mode=thinking_mode,
            )
        elif provider == "zai":
            models[label] = ZaiChatModel(
                name=label,
                model_id=cfg["model"],
                api_key_env=cfg.get("api_key_env", "ZAI_API_KEY"),
                api_key=cfg.get("api_key"),
                base_url=cfg.get("base_url", "https://api.z.ai/api/paas/v4"),
                timeout_sec=int(cfg.get("timeout_sec", 180)),
                default_max_output_tokens=int(cfg.get("max_output_tokens", 4096)),
                thinking_mode=cfg.get("thinking_mode"),
                temperature=cfg.get("temperature"),
                top_p=cfg.get("top_p"),
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    return models
