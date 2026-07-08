"""Model selection helpers for allowed Fireworks models."""

from __future__ import annotations


SIMPLE_CATEGORIES = {"factual", "sentiment", "ner", "summarization"}


def parse_allowed_models(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(model.strip() for model in raw.split(",") if model.strip())


def _contains(model: str, value: str) -> bool:
    return value in model.casefold()


def _first_matching(models: tuple[str, ...], *needles: str) -> str | None:
    for model in models:
        lowered = model.casefold()
        if any(needle in lowered for needle in needles):
            return model
    return None


def _matching(models: tuple[str, ...], *needles: str) -> list[str]:
    return [
        model
        for model in models
        if any(needle in model.casefold() for needle in needles)
    ]


def _without_family(models: tuple[str, ...], *needles: str) -> list[str]:
    return [
        model
        for model in models
        if not any(needle in model.casefold() for needle in needles)
    ]


def _dedupe(models: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        ordered.append(model)
    return tuple(ordered)


def _cheapest_gemma(models: tuple[str, ...]) -> str | None:
    gemmas = [model for model in models if _contains(model, "gemma")]
    if not gemmas:
        return None

    def cost_key(item: tuple[int, str]) -> tuple[int, int, int]:
        index, model = item
        lowered = model.casefold()
        if "nvfp4" in lowered:
            family_cost = 0
        elif "26b" in lowered:
            family_cost = 1
        elif "31b" in lowered:
            family_cost = 2
        else:
            family_cost = 3
        return family_cost, len(model), index

    indexed = [(models.index(model), model) for model in gemmas]
    return min(indexed, key=cost_key)[1]


def select_model(category: str, allowed_models: tuple[str, ...]) -> str:
    if not allowed_models:
        raise ValueError("At least one allowed model is required.")

    return select_model_candidates(category, allowed_models)[0]


def select_model_candidates(category: str, allowed_models: tuple[str, ...]) -> tuple[str, ...]:
    if not allowed_models:
        raise ValueError("At least one allowed model is required.")

    kimi = _matching(allowed_models, "kimi")
    minimax = _matching(allowed_models, "minimax")
    gemma = _matching(allowed_models, "gemma")
    other = _without_family(allowed_models, "kimi", "minimax", "gemma")

    return _dedupe(minimax + kimi + other + gemma)


def select_fallback_model(
    category: str,
    allowed_models: tuple[str, ...],
    current_model: str,
) -> str | None:
    if len(allowed_models) < 2:
        return None

    if category in {"math", "logic", "unknown"}:
        preferences = ("minimax", "kimi")
    else:
        preferences = ("minimax", "kimi")

    for needle in preferences:
        candidate = _first_matching(allowed_models, needle)
        if candidate and candidate != current_model:
            return candidate

    for candidate in allowed_models:
        if candidate != current_model:
            return candidate
    return None
