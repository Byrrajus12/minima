"""Configuration loading for the minima baseline agent."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .model_selector import parse_allowed_models


class ConfigError(ValueError):
    """Raised when runtime configuration is incomplete or invalid."""


@dataclass(frozen=True)
class Config:
    fireworks_api_key: str | None
    fireworks_base_url: str | None
    allowed_models: tuple[str, ...]
    placeholder_mode: bool
    request_timeout_seconds: float = 30.0

    @property
    def model(self) -> str:
        if not self.allowed_models:
            raise ConfigError("ALLOWED_MODELS must contain at least one model.")
        return self.allowed_models[0]


def load_config() -> Config:
    api_key = os.getenv("FIREWORKS_API_KEY")
    base_url = os.getenv("FIREWORKS_BASE_URL")
    allowed_models = parse_allowed_models(os.getenv("ALLOWED_MODELS"))

    any_fireworks_env = any([api_key, base_url, allowed_models])
    if not any_fireworks_env:
        return Config(
            fireworks_api_key=None,
            fireworks_base_url=None,
            allowed_models=(),
            placeholder_mode=True,
        )

    missing: list[str] = []
    if not api_key:
        missing.append("FIREWORKS_API_KEY")
    if not base_url:
        missing.append("FIREWORKS_BASE_URL")
    if not allowed_models:
        missing.append("ALLOWED_MODELS")
    if missing:
        names = ", ".join(missing)
        raise ConfigError(f"Missing required Fireworks environment variables: {names}")

    return Config(
        fireworks_api_key=api_key,
        fireworks_base_url=base_url.rstrip("/"),
        allowed_models=allowed_models,
        placeholder_mode=False,
    )
