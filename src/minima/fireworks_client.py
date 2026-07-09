"""Minimal Fireworks chat-completions client."""

from __future__ import annotations

import json
import os
import socket
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import Config
from .prompts import build_user_prompt, max_tokens_for_category


class FireworksClientError(RuntimeError):
    """Raised when a Fireworks request fails."""


_REASONING_EFFORT_UNSUPPORTED_MODELS: set[str] = set()


def _decode_error_body(exc: urllib.error.HTTPError) -> str:
    detail = exc.read().decode("utf-8", errors="replace").strip()
    if not detail:
        return "no response body"
    return detail[:1000]


def _log_usage(
    data: dict[str, Any],
    first_choice: dict[str, Any],
    category: str,
    model: str,
    task_id: str | None,
    reasoning_effort_retry: bool,
) -> None:
    if os.getenv("MINIMA_LOG_USAGE") != "1":
        return

    payload: dict[str, Any] = {
        "category": category,
        "model": model,
        "reasoning_effort_retry": reasoning_effort_retry,
    }
    if task_id:
        payload["task_id"] = task_id

    finish_reason = first_choice.get("finish_reason")
    if isinstance(finish_reason, str):
        payload["finish_reason"] = finish_reason

    usage = data.get("usage")
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                payload[key] = value

    print(
        "minima usage "
        + json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        file=sys.stderr,
    )


def _parse_chat_response(
    response_body: str,
    category: str,
    model: str,
    task_id: str | None = None,
    reasoning_effort_retry: bool = False,
) -> str:
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
    if first_choice.get("finish_reason") == "length":
        print(
            f"minima warning finish_reason=length category={category} model={model}",
            file=sys.stderr,
        )

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise FireworksClientError("Fireworks response did not include a chat message.")

    answer = message.get("content")
    if not isinstance(answer, str) or not answer.strip():
        raise FireworksClientError("Fireworks returned an empty answer.")

    _log_usage(
        data=data,
        first_choice=first_choice,
        category=category,
        model=model,
        task_id=task_id,
        reasoning_effort_retry=reasoning_effort_retry,
    )
    return answer.strip()


def _rejects_reasoning_effort(detail: str) -> bool:
    lowered = detail.casefold()
    return "reasoning_effort" in lowered or (
        "reasoning" in lowered and "unsupported" in lowered
    )


@dataclass(frozen=True)
class FireworksClient:
    config: Config

    def answer(
        self,
        prompt: str,
        category: str,
        model: str | None = None,
        task_id: str | None = None,
    ) -> str:
        if self.config.placeholder_mode:
            return (
                "[LOCAL TEST PLACEHOLDER - Fireworks env vars not set] "
                f"category={category}; prompt_chars={len(prompt)}"
            )

        if model is None:
            model = self.config.model

        response_body, reasoning_effort_retry = self._post_chat_completion(
            prompt=prompt,
            category=category,
            model=model,
            include_reasoning_effort=model not in _REASONING_EFFORT_UNSUPPORTED_MODELS,
        )
        return _parse_chat_response(
            response_body,
            category=category,
            model=model,
            task_id=task_id,
            reasoning_effort_retry=reasoning_effort_retry,
        )

    def _post_chat_completion(
        self,
        prompt: str,
        category: str,
        model: str,
        include_reasoning_effort: bool,
    ) -> tuple[str, bool]:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": build_user_prompt(category, prompt)},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens_for_category(category),
        }
        if include_reasoning_effort:
            payload["reasoning_effort"] = "none"

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
            if include_reasoning_effort and _rejects_reasoning_effort(detail):
                _REASONING_EFFORT_UNSUPPORTED_MODELS.add(model)
                response_body, _ = self._post_chat_completion(
                    prompt=prompt,
                    category=category,
                    model=model,
                    include_reasoning_effort=False,
                )
                return response_body, True
            raise FireworksClientError(f"Fireworks HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise FireworksClientError(f"Fireworks request failed: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise FireworksClientError("Fireworks request timed out.") from exc

        return response_body, False
