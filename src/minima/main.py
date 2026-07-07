"""Command-line entry point for the minima Phase 2 baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import ConfigError, load_config
from .fireworks_client import FireworksClient, FireworksClientError
from .router import Router
from .validators import ValidationError, validate_results, validate_tasks


DEFAULT_INPUT = Path("/input/tasks.json")
DEFAULT_OUTPUT = Path("/output/results.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minima baseline agent.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to tasks JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to results JSON.")
    return parser.parse_args(argv)


def read_tasks(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return validate_tasks(data)


def write_results(path: Path, results: list[dict[str, str]]) -> None:
    validated = validate_results(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(validated, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def run(input_path: Path, output_path: Path) -> int:
    config = load_config()
    router = Router(FireworksClient(config))
    tasks = read_tasks(input_path)

    results: list[dict[str, str]] = []
    for task in tasks:
        answer = router.answer(task["prompt"])
        results.append({"task_id": task["task_id"], "answer": answer})

    write_results(output_path, results)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(Path(args.input), Path(args.output))
    except (
        OSError,
        json.JSONDecodeError,
        ValidationError,
        ConfigError,
        FireworksClientError,
    ) as exc:
        print(f"minima error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
