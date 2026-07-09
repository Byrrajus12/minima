"""Model selection helpers for allowed Fireworks models."""

from __future__ import annotations


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


def _matches_any(model: str, *needles: str) -> bool:
    lowered = model.casefold()
    return any(needle in lowered for needle in needles)


def _matching(models: tuple[str, ...], *needles: str) -> list[str]:
    return [
        model
        for model in models
        if _matches_any(model, *needles)
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


def _partition_families(
    models: tuple[str, ...],
) -> tuple[list[str], list[str], list[str], list[str]]:
    code: list[str] = []
    minimax: list[str] = []
    gemma: list[str] = []
    other: list[str] = []
    for model in models:
        if _matches_any(model, "kimi", "code"):
            code.append(model)
        elif _matches_any(model, "minimax"):
            minimax.append(model)
        elif _matches_any(model, "gemma"):
            gemma.append(model)
        else:
            other.append(model)
    return code, minimax, gemma, other


def _cheapest_gemma(gemmas: list[str]) -> str | None:
    if not gemmas:
        return None

    def cost_key(item: tuple[int, str]) -> tuple[int, int, int]:
        index, model = item
        lowered = model.casefold()
        if "a4b" in lowered:
            family_cost = 0
        elif "26b" in lowered:
            family_cost = 1
        elif "nvfp4" in lowered:
            family_cost = 2
        elif "31b" in lowered:
            family_cost = 3
        else:
            family_cost = 4
        return family_cost, len(model), index

    indexed = list(enumerate(gemmas))
    return min(indexed, key=cost_key)[1]


def _split_gemma(gemmas: list[str]) -> tuple[list[str], list[str]]:
    explicit_small = [
        model for model in gemmas if _matches_any(model, "a4b", "26b")
    ]
    if explicit_small:
        gemma_small = explicit_small
    else:
        cheapest = _cheapest_gemma(gemmas)
        gemma_small = [cheapest] if cheapest else []

    small_set = set(gemma_small)
    remaining = [model for model in gemmas if model not in small_set]
    quant_markers = ("nvfp4", "fp8", "int8")
    full_precision = [
        model for model in remaining if not _matches_any(model, *quant_markers)
    ]
    quantized = [
        model for model in remaining if _matches_any(model, *quant_markers)
    ]
    return gemma_small, full_precision + quantized


def select_model(category: str, allowed_models: tuple[str, ...]) -> str:
    if not allowed_models:
        raise ValueError("At least one allowed model is required.")

    return select_model_candidates(category, allowed_models)[0]


def select_model_candidates(category: str, allowed_models: tuple[str, ...]) -> tuple[str, ...]:
    if not allowed_models:
        raise ValueError("At least one allowed model is required.")

    code, minimax, gemma, other = _partition_families(allowed_models)
    gemma_small, gemma_mid = _split_gemma(gemma)

    if category in {"code_debugging", "code_generation"}:
        return _dedupe(code + gemma_mid + minimax + gemma_small + other)
    if category in {"math", "logic"}:
        return _dedupe(gemma_mid + gemma_small + minimax + code + other)
    if category in {"sentiment", "ner", "summarization", "factual", "unknown"}:
        return _dedupe(gemma_small + gemma_mid + minimax + code + other)
    return _dedupe(gemma_small + gemma_mid + minimax + code + other)


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
