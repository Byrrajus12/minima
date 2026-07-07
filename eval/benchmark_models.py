"""Benchmark allowed Fireworks models against a small evaluation set."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minima.config import Config  # noqa: E402
from minima.fireworks_client import FireworksClient, FireworksClientError  # noqa: E402
from minima.model_selector import parse_allowed_models  # noqa: E402
from minima.task_classifier import classify_task  # noqa: E402
from minima.validators import validate_tasks  # noqa: E402
from score_outputs import load_json, score_answer  # noqa: E402


DEFAULT_TASKS = ROOT / "eval" / "live_mini_tasks.json"
DEFAULT_EXPECTED = ROOT / "eval" / "live_mini_expected.json"
DEFAULT_REPORT = ROOT / "eval" / "reports" / "model_benchmark.md"
DEFAULT_JSON = ROOT / "eval" / "reports" / "model_benchmark.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Fireworks models.")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--expected", default=str(DEFAULT_EXPECTED))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    return parser.parse_args()


def load_models() -> tuple[str, ...]:
    raw = os.getenv("BENCHMARK_MODELS") or os.getenv("ALLOWED_MODELS")
    return parse_allowed_models(raw)


def load_client(models: tuple[str, ...]) -> FireworksClient:
    api_key = os.getenv("FIREWORKS_API_KEY")
    base_url = os.getenv("FIREWORKS_BASE_URL")
    missing = []
    if not api_key:
        missing.append("FIREWORKS_API_KEY")
    if not base_url:
        missing.append("FIREWORKS_BASE_URL")
    if not models:
        missing.append("BENCHMARK_MODELS or ALLOWED_MODELS")
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Missing required benchmark environment variables: {names}")

    config = Config(
        fireworks_api_key=api_key,
        fireworks_base_url=base_url.rstrip("/"),
        allowed_models=models,
        placeholder_mode=False,
    )
    return FireworksClient(config)


def expected_by_id(expected: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["task_id"]): item for item in expected}


def benchmark(
    client: FireworksClient,
    models: tuple[str, ...],
    tasks: list[dict[str, str]],
    expected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_items = expected_by_id(expected)
    rows: list[dict[str, Any]] = []

    for model in models:
        for task in tasks:
            task_id = task["task_id"]
            category = classify_task(task["prompt"])
            started = time.perf_counter()
            answer = ""
            ok = False
            reason = "missing expected metadata"

            expected_item = expected_items.get(task_id)
            try:
                answer = client.answer(task["prompt"], category=category, model=model)
                if expected_item:
                    ok, reason = score_answer(answer, expected_item["scoring"])
            except FireworksClientError as exc:
                reason = f"api error: {exc}"
                api_error = True
            else:
                api_error = False

            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "model": model,
                    "category": category,
                    "task_id": task_id,
                    "passed": ok,
                    "api_error": api_error,
                    "reason": reason,
                    "answer_length": len(answer),
                    "runtime_seconds": round(elapsed, 3),
                    "answer": answer,
                }
            )

    return rows


def summarize_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: Counter[str] = Counter()
    passed: Counter[str] = Counter()
    for row in rows:
        model = str(row["model"])
        totals[model] += 1
        if row["passed"]:
            passed[model] += 1
    return [
        {
            "model": model,
            "passed": passed[model],
            "total": totals[model],
            "pass_rate": passed[model] / totals[model] if totals[model] else 0.0,
        }
        for model in totals
    ]


def summarize_by_model_category(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: defaultdict[tuple[str, str], int] = defaultdict(int)
    passed: defaultdict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        key = (str(row["model"]), str(row["category"]))
        totals[key] += 1
        if row["passed"]:
            passed[key] += 1
    return [
        {
            "model": model,
            "category": category,
            "passed": passed[(model, category)],
            "total": total,
            "pass_rate": passed[(model, category)] / total if total else 0.0,
        }
        for (model, category), total in totals.items()
    ]


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    data = {
        "summary_by_model": summarize_by_model(rows),
        "summary_by_model_category": summarize_by_model_category(rows),
        "results": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Model Benchmark",
        "",
        "## Summary by Model",
        "",
        "| Model | Passed | Total | Pass rate |",
        "| --- | ---: | ---: | ---: |",
    ]

    for item in summarize_by_model(rows):
        rate = item["pass_rate"] * 100
        lines.append(f"| `{item['model']}` | {item['passed']} | {item['total']} | {rate:.1f}% |")

    lines.extend(
        [
            "",
            "## Summary by Model and Category",
            "",
            "| Model | Category | Passed | Total | Pass rate |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for item in summarize_by_model_category(rows):
        rate = item["pass_rate"] * 100
        lines.append(
            f"| `{item['model']}` | {item['category']} | {item['passed']} | "
            f"{item['total']} | {rate:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Task Results",
            "",
            "| Model | Category | Task | Result | Answer chars | Seconds | Reason |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        result = "pass" if row["passed"] else "fail"
        reason = str(row["reason"]).replace("|", "\\|")
        lines.append(
            f"| `{row['model']}` | {row['category']} | {row['task_id']} | "
            f"{result} | {row['answer_length']} | {row['runtime_seconds']:.3f} | {reason} |"
        )

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> int:
    args = parse_args()
    models = load_models()
    client = load_client(models)
    tasks = validate_tasks(load_json(Path(args.tasks)))
    expected = load_json(Path(args.expected))

    rows = benchmark(client, models, tasks, expected)
    write_report(Path(args.report), rows)
    write_json(Path(args.json), rows)

    for item in summarize_by_model(rows):
        rate = item["pass_rate"] * 100
        print(f"{item['model']}: {item['passed']}/{item['total']} ({rate:.1f}%)")

    return 1 if any(row["api_error"] for row in rows) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        raise SystemExit(1)
