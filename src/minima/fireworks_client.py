"""Minimal Fireworks chat-completions client."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import Config
from .prompts import build_user_prompt


class FireworksClientError(RuntimeError):
    """Raised when a Fireworks request fails."""


def _decode_error_body(exc: urllib.error.HTTPError) -> str:
    detail = exc.read().decode("utf-8", errors="replace").strip()
    if not detail:
        return "no response body"
    return detail[:1000]


def _parse_chat_response(response_body: str) -> str:
    if not response_body.strip():
        raise FireworksClientError("Fireworks returned an empty response.")

    try:
        data: Any = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise FireworksClientError("Fireworks returned malformed JSON.") from exc

    if not isinstance(data, dict):
        raise FireworksClientError("Fireworks response was not a JSON object.")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise FireworksClientError("Fireworks response did not include choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise FireworksClientError("Fireworks choice did not match chat format.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise FireworksClientError("Fireworks response did not include a chat message.")

    answer = message.get("content")
    if not isinstance(answer, str) or not answer.strip():
        raise FireworksClientError("Fireworks returned an empty answer.")

    return answer.strip()


def _max_tokens_for_category(category: str) -> int:
    return {
        "factual": 96,
        "math": 96,
        "sentiment": 128,
        "summarization": 192,
        "ner": 192,
        "logic": 96,
        "code_debugging": 512,
        "code_generation": 512,
    }.get(category, 192)


@dataclass(frozen=True)
class FireworksClient:
    config: Config

    def answer(self, prompt: str, category: str, model: str | None = None) -> str:
        if self.config.placeholder_mode:
            return (
                "[LOCAL TEST PLACEHOLDER - Fireworks env vars not set] "
                f"category={category}; prompt_chars={len(prompt)}"
            )

        if model is None:
            model = self.config.model

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": build_user_prompt(category, prompt)},
            ],
            "temperature": 0.1,
            "max_tokens": _max_tokens_for_category(category),
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
            detail = _decode_error_body(exc)
            raise FireworksClientError(f"Fireworks HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise FireworksClientError(f"Fireworks request failed: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise FireworksClientError("Fireworks request timed out.") from exc

        return _parse_chat_response(response_body)
