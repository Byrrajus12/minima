"""Task routing for the minima agent."""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys

from .answer_validator import is_valid_answer
from .fireworks_client import FireworksClient, FireworksClientError
from .local_solver import solve_local
from .model_selector import select_model_candidates
from .task_classifier import classify_task


STRICT_RETRY_INSTRUCTION = (
    "The previous answer was invalid or incomplete. Follow the requested format "
    "exactly and provide the final answer directly."
)


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


def _safe_fallback_answer(category: str) -> str:
    if category == "sentiment":
        return "neutral"
    if category == "logic":
        return "unknown"
    if category == "math":
        return "0"
    return "Unable to determine."


def _local_solver_enabled() -> bool:
    return os.getenv("MINIMA_ENABLE_LOCAL_SOLVER") == "1"


@dataclass(frozen=True)
class Router:
    client: FireworksClient

    def answer(self, prompt: str, task_id: str | None = None) -> str:
        category = classify_task(prompt)
        if _local_solver_enabled():
            local_answer = solve_local(category, prompt)
            if local_answer is not None:
                _log_routing(task_id, category, f"local:{category}", retry=0)
                return local_answer

        if self.client.config.placeholder_mode:
            _log_routing(task_id, category, "placeholder", retry=0)
            return self.client.answer(prompt=prompt, category=category)

        best_non_empty_answer: str | None = None
        retry_instruction: str | None = None
        for retry, model in enumerate(
            select_model_candidates(category, self.client.config.allowed_models)
        ):
            _log_routing(task_id, category, model, retry=retry)
            try:
                answer = self.client.answer(
                    prompt=prompt,
                    category=category,
                    model=model,
                    retry_instruction=retry_instruction,
                )
            except FireworksClientError:
                _log_model_failure(task_id, category, model, retry, "client_error")
                continue
            if answer.strip() and best_non_empty_answer is None:
                best_non_empty_answer = answer
            if is_valid_answer(category, prompt, answer):
                return answer
            _log_model_failure(task_id, category, model, retry, "invalid_answer")
            retry_instruction = STRICT_RETRY_INSTRUCTION

        if best_non_empty_answer:
            return best_non_empty_answer
        _log_model_failure(task_id, category, "all", retry=0, reason="no_model_succeeded")
        return _safe_fallback_answer(category)
