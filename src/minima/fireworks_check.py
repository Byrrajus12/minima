"""Fireworks connectivity check for minima."""

from __future__ import annotations

import sys
from typing import Sequence

from .config import ConfigError, load_config
from .fireworks_client import FireworksClient, FireworksClientError


def run_check() -> int:
    config = load_config()
    if config.placeholder_mode:
        raise ConfigError(
            "Fireworks connectivity check requires FIREWORKS_API_KEY, "
            "FIREWORKS_BASE_URL, and ALLOWED_MODELS."
        )

    client = FireworksClient(config)
    answer = client.answer(prompt="Reply with OK.", category="unknown")
    print(
        "Fireworks connectivity check succeeded "
        f"with model={config.model}; answer_chars={len(answer)}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _ = argv
    try:
        return run_check()
    except (ConfigError, FireworksClientError) as exc:
        print(f"Fireworks connectivity check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
