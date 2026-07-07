"""Minimal Fireworks chat-completions client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import Config
from .prompts import SYSTEM_PROMPT, build_user_prompt


class FireworksClientError(RuntimeError):
    """Raised when a Fireworks request fails."""


@dataclass(frozen=True)
class FireworksClient:
    config: Config

    def answer(self, prompt: str, category: str) -> str:
        if self.config.placeholder_mode:
            return (
                "[LOCAL TEST PLACEHOLDER - Fireworks env vars not set] "
                f"category={category}; prompt_chars={len(prompt)}"
            )

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(category, prompt)},
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }

        body = json.dumps(payload).encode("utf-8")
        endpoint = f"{self.config.fireworks_base_url}/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.fireworks_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self.config.request_timeout_seconds
            ) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise FireworksClientError(f"Fireworks HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise FireworksClientError(f"Fireworks request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise FireworksClientError("Fireworks request timed out.") from exc

        try:
            data = json.loads(response_body)
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise FireworksClientError("Fireworks response did not match chat format.") from exc

        if not isinstance(answer, str) or not answer.strip():
            raise FireworksClientError("Fireworks returned an empty answer.")
        return answer.strip()
