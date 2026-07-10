"""Task routing for the minima agent."""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys

from .fireworks_client import FireworksClient, FireworksClientError
from .local_solver import try_local_answer
from .model_selector import select_model_candidates
from .task_classifier import classify_task


def _log_routing(task_id: str | None, category: str, model: str, retry: int) -> None:
    if os.getenv("MINIMA_LOG_ROUTING") != "1":
        return
    label = task_id or "-"
    print(
        f"minima route task_id={label} category={category} model={model} retry={retry}",
        file=sys.stderr,
    )


def _log_model_failure(
    task_id: str | None,
    category: str,
    model: str,
    retry: int,
    reason: str,
) -> None:
    if os.getenv("MINIMA_LOG_ROUTING") != "1":
        return
    label = task_id or "-"
    print(
        f"minima route_failed task_id={label} category={category} "
        f"model={model} retry={retry} reason={reason}",
        file=sys.stderr,
    )


def _log_local_solved(task_id: str | None, category: str) -> None:
    if os.getenv("MINIMA_LOG_ROUTING") != "1":
        return
    label = task_id or "-"
    print(
        f"minima local_solved task_id={label} category={category}",
        file=sys.stderr,
    )


def _safe_fallback_answer(category: str) -> str:
    if category == "sentiment":
        return "neutral"
    if category == "logic":
        return "unknown"
    if category == "math":
        return "0"
    return "Unable to determine."


@dataclass(frozen=True)
class Router:
    client: FireworksClient

    def answer(self, prompt: str, task_id: str | None = None) -> str:
        category = classify_task(prompt)
        local_answer = try_local_answer(category, prompt)
        if local_answer is not None:
            _log_local_solved(task_id, category)
            return local_answer

        if self.client.config.placeholder_mode:
            _log_routing(task_id, category, "placeholder", retry=0)
            return self.client.answer(prompt=prompt, category=category, task_id=task_id)

        for retry, model in enumerate(
            select_model_candidates(category, self.client.config.allowed_models)
        ):
            _log_routing(task_id, category, model, retry=retry)
            try:
                answer = self.client.answer(
                    prompt=prompt,
                    category=category,
                    model=model,
                    task_id=task_id,
                )
            except FireworksClientError:
                _log_model_failure(task_id, category, model, retry, "client_error")
                continue
            stripped = answer.strip()
            if stripped:
                return stripped
            _log_model_failure(task_id, category, model, retry, "empty_answer")

        _log_model_failure(task_id, category, "all", retry=0, reason="no_model_succeeded")
        return _safe_fallback_answer(category)
