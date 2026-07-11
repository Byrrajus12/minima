"""Best-effort local GGUF generation through llama-cpp-python."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import time
from typing import Any


_DEFAULT_MODEL_PATHS = (Path("/app/model.gguf"), Path("/app/models/model.gguf"))
_MODEL: Any | None = None
_MODEL_LOAD_ATTEMPTED = False
_MODEL_INITIALIZATIONS = 0

_SPECIAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
)

_REFUSAL_MARKERS = (
    "as an ai",
    "cannot comply",
    "can't help",
    "i am unable",
    "i cannot",
    "i can't",
    "i'm unable",
    "sorry, i",
)


@dataclass(frozen=True)
class LocalGeneration:
    text: str | None
    raw_text: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    runtime_ms: float
    token_count_estimated: bool
    error: str | None = None
    model_load_ms: float = 0.0
    finish_reason: str | None = None
    max_completion_tokens: int = 0
    reached_token_cap: bool = False


def _configured_model_paths() -> tuple[Path, ...]:
    env_path = os.getenv("MINIMA_LOCAL_MODEL_PATH")
    if env_path:
        return (Path(env_path),)
    return _DEFAULT_MODEL_PATHS


def _resolve_model_path() -> Path | None:
    for path in _configured_model_paths():
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def _int_env(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _load_model() -> Any | None:
    global _MODEL, _MODEL_LOAD_ATTEMPTED, _MODEL_INITIALIZATIONS
    if _MODEL_LOAD_ATTEMPTED:
        return _MODEL

    _MODEL_LOAD_ATTEMPTED = True
    model_path = _resolve_model_path()
    if model_path is None:
        return None

    try:
        from llama_cpp import Llama

        _MODEL = Llama(
            model_path=str(model_path),
            n_ctx=_int_env("MINIMA_LOCAL_N_CTX", 1024, 256),
            n_threads=_int_env("MINIMA_LOCAL_THREADS", 2, 1),
            n_gpu_layers=0,
            verbose=False,
        )
        _MODEL_INITIALIZATIONS += 1
    except Exception:
        _MODEL = None
    return _MODEL


def model_initialization_count() -> int:
    return _MODEL_INITIALIZATIONS


def _format_qwen_chat(system: str, prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{system.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _completion_text(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    text = first_choice.get("text")
    return text if isinstance(text, str) else None


def _finish_reason(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    reason = first_choice.get("finish_reason")
    return str(reason) if reason is not None else None


def _usage_counts(response: Any) -> tuple[int, int, int] | None:
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if all(isinstance(value, int) for value in (prompt_tokens, completion_tokens, total_tokens)):
        return int(prompt_tokens), int(completion_tokens), int(total_tokens)
    return None


def _count_tokens(model: Any, text: str) -> int | None:
    tokenize = getattr(model, "tokenize", None)
    if not callable(tokenize):
        return None
    try:
        return len(tokenize(text.encode("utf-8")))
    except Exception:
        return None


def _estimated_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def _looks_like_refusal(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def _clean_output(text: str) -> str | None:
    for token in _SPECIAL_TOKENS:
        text = text.replace(token, "")
    text = re.sub(r"^\s*assistant\s*:\s*", "", text, flags=re.I).strip()

    fenced = re.search(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1).strip()
    elif text.startswith("```"):
        text = re.sub(r"^```(?:[A-Za-z0-9_-]+)?\s*", "", text).strip()
        text = re.sub(r"\s*```$", "", text).strip()

    if not text or _looks_like_refusal(text):
        return None
    return text.strip()


def local_generate(system: str, prompt: str, max_tokens: int) -> str | None:
    result = local_generate_with_metadata(system=system, prompt=prompt, max_tokens=max_tokens)
    return result.text


def local_generate_with_metadata(system: str, prompt: str, max_tokens: int) -> LocalGeneration:
    started = time.perf_counter()
    load_started = time.perf_counter()
    model = _load_model()
    model_load_ms = (time.perf_counter() - load_started) * 1000
    if model is None:
        return LocalGeneration(
            text=None,
            raw_text=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            runtime_ms=(time.perf_counter() - started) * 1000,
            token_count_estimated=False,
            error="model unavailable",
            model_load_ms=model_load_ms,
            max_completion_tokens=max_tokens,
        )

    request = _format_qwen_chat(system, prompt)
    token_cap = max(1, min(max_tokens, 256))
    response: Any | None = None
    try:
        response = model(
            request,
            max_tokens=token_cap,
            temperature=0.0,
            top_p=1.0,
            repeat_penalty=1.05,
            stop=["<|im_end|>", "<|endoftext|>"],
            echo=False,
        )
    except Exception:
        return LocalGeneration(
            text=None,
            raw_text=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            runtime_ms=(time.perf_counter() - started) * 1000,
            token_count_estimated=False,
            error="generation failed",
            model_load_ms=model_load_ms,
            max_completion_tokens=token_cap,
        )

    text = _completion_text(response)
    finish_reason = _finish_reason(response)
    raw_text = text
    runtime_ms = (time.perf_counter() - started) * 1000
    usage = _usage_counts(response)
    estimated = False
    if usage is None:
        prompt_tokens = _count_tokens(model, request)
        completion_tokens = _count_tokens(model, text or "")
        if prompt_tokens is None or completion_tokens is None:
            prompt_tokens = _estimated_tokens(request)
            completion_tokens = _estimated_tokens(text or "")
            estimated = True
        total_tokens = prompt_tokens + completion_tokens
    else:
        prompt_tokens, completion_tokens, total_tokens = usage
    if text is None:
        return LocalGeneration(
            text=None,
            raw_text=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            runtime_ms=runtime_ms,
            token_count_estimated=estimated,
            error="empty completion",
            model_load_ms=model_load_ms,
            finish_reason=finish_reason,
            max_completion_tokens=token_cap,
            reached_token_cap=completion_tokens >= token_cap,
        )
    return LocalGeneration(
        text=_clean_output(text),
        raw_text=raw_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        runtime_ms=runtime_ms,
        token_count_estimated=estimated,
        error=None,
        model_load_ms=model_load_ms,
        finish_reason=finish_reason,
        max_completion_tokens=token_cap,
        reached_token_cap=completion_tokens >= token_cap,
    )
