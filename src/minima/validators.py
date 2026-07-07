"""Input and output validation helpers."""

from __future__ import annotations

from typing import Any


class ValidationError(ValueError):
    """Raised when task input or result output is malformed."""


def validate_tasks(data: Any) -> list[dict[str, str]]:
    if not isinstance(data, list):
        raise ValidationError("Input must be a JSON array.")

    tasks: list[dict[str, str]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValidationError(f"Task at index {index} must be an object.")
        task_id = item.get("task_id")
        prompt = item.get("prompt")
        if not isinstance(task_id, str) or not task_id:
            raise ValidationError(f"Task at index {index} must have a non-empty string task_id.")
        if not isinstance(prompt, str) or not prompt:
            raise ValidationError(f"Task {task_id} must have a non-empty string prompt.")
        tasks.append({"task_id": task_id, "prompt": prompt})
    return tasks


def validate_results(data: Any) -> list[dict[str, str]]:
    if not isinstance(data, list):
        raise ValidationError("Results must be a JSON array.")

    results: list[dict[str, str]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValidationError(f"Result at index {index} must be an object.")
        task_id = item.get("task_id")
        answer = item.get("answer")
        if not isinstance(task_id, str) or not task_id:
            raise ValidationError(f"Result at index {index} must have a non-empty string task_id.")
        if not isinstance(answer, str):
            raise ValidationError(f"Result {task_id} must have a string answer.")
        results.append({"task_id": task_id, "answer": answer})
    return results
