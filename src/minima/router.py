"""Task routing for the minima agent."""

from __future__ import annotations

from dataclasses import dataclass

from .fireworks_client import FireworksClient
from .task_classifier import classify_task


@dataclass(frozen=True)
class Router:
    client: FireworksClient

    def answer(self, prompt: str) -> str:
        category = classify_task(prompt)
        return self.client.answer(prompt=prompt, category=category)
