"""Best-effort local GGUF generation through llama-cpp-python."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any


_DEFAULT_MODEL_PATHS = (Path("/app/model.gguf"), Path("/app/models/model.gguf"))
_MODEL: Any | None = None
_MODEL_LOAD_ATTEMPTED = False

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
    global _MODEL, _MODEL_LOAD_ATTEMPTED
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
    except Exception:
        _MODEL = None
    return _MODEL


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
    model = _load_model()
    if model is None:
        return None

    request = _format_qwen_chat(system, prompt)
    token_cap = max(1, min(max_tokens, 256))
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
        try:
            response = model(
                request,
                max_tokens=token_cap,
                temperature=0.01,
                top_p=1.0,
                repeat_penalty=1.05,
                stop=["<|im_end|>", "<|endoftext|>"],
                echo=False,
            )
        except Exception:
            return None

    text = _completion_text(response)
    if text is None:
        return None
    return _clean_output(text)
