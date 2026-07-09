"""Repeat forced-model comparisons and aggregate accuracy/token results."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EVAL = ROOT / "eval"
for path in (SRC, EVAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_models import _run_forced_model, _slug  # noqa: E402
from report_usage import _aggregate, _load_usage, _token_value  # noqa: E402
from score_outputs import load_json, score_answer  # noqa: E402
from minima.validators import validate_results, validate_tasks  # noqa: E402


def _usage_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "prompt_tokens": sum(_token_value(record, "prompt_tokens") for record in records),
        "completion_tokens": sum(_token_value(record, "completion_tokens") for record in records),
        "total_tokens": sum(_token_value(record, "total_tokens") for record in records),
    }


def _score_run(
    tasks: list[dict[str, str]],
    expected: list[dict[str, Any]],
    results: list[dict[str, str]],
) -> dict[str, Any]:
    expected_by_id = {item["task_id"]: item for item in expected}
    result_by_id = {item["task_id"]: item for item in results}
    total_by_category: Counter[str] = Counter()
    passed_by_category: Counter[str] = Counter()
    failed_task_ids: list[str] = []

    for task in tasks:
        task_id = task["task_id"]
        expected_item = expected_by_id.get(task_id)
        if not expected_item:
            failed_task_ids.append(task_id)
            continue
        category = str(expected_item["category"])
        total_by_category[category] += 1
        result = result_by_id.get(task_id)
        answer = result.get("answer", "") if result else ""
        if not isinstance(answer, str) or not answer.strip():
            failed_task_ids.append(task_id)
            continue
        ok, _ = score_answer(answer, expected_item["scoring"])
        if ok:
            passed_by_category[category] += 1
        else:
            failed_task_ids.append(task_id)

    total_tasks = len(tasks)
    passed = total_tasks - len(failed_task_ids)
    return {
        "passed": passed,
        "total": total_tasks,
        "score": passed / total_tasks if total_tasks else 0.0,
        "failed_task_ids": failed_task_ids,
        "category_total": dict(total_by_category),
        "category_passed": dict(passed_by_category),
    }


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _run_one(
    label: str,
    model: str,
    run_index: int,
    tasks: list[dict[str, str]],
    expected: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    run_label = f"{_slug(label)}_run{run_index}"
    results_path = output_dir / f"compare_{run_label}_results.json"
    stderr_path = output_dir / f"compare_{run_label}_stderr.txt"
    _run_forced_model(tasks, model, results_path, stderr_path)

    results = validate_results(load_json(results_path))
    usage_records = _load_usage(stderr_path)
    score = _score_run(tasks, expected, results)
    usage = _usage_totals(usage_records)
    by_category = _aggregate(usage_records, ("category",))

    return {
        "label": label,
        "model": model,
        "run": run_index,
        "results_path": str(results_path),
        "stderr_path": str(stderr_path),
        "passed": score["passed"],
        "total": score["total"],
        "score": score["score"],
        "failed_task_ids": score["failed_task_ids"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "category_total": score["category_total"],
        "category_passed": score["category_passed"],
        "category_usage": {
            category[0]: values for category, values in by_category.items()
        },
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(run["score"]) for run in runs]
    totals = [int(run["total_tokens"]) for run in runs]
    prompt = [int(run["prompt_tokens"]) for run in runs]
    completion = [int(run["completion_tokens"]) for run in runs]
    repeated_failures: Counter[str] = Counter()
    categories = sorted(
        {
            category
            for run in runs
            for category in run["category_total"]
        }
    )

    for run in runs:
        repeated_failures.update(run["failed_task_ids"])

    category_summary: dict[str, dict[str, float]] = {}
    for category in categories:
        category_scores: list[float] = []
        category_tokens: list[int] = []
        for run in runs:
            total = int(run["category_total"].get(category, 0))
            passed = int(run["category_passed"].get(category, 0))
            if total:
                category_scores.append(passed / total)
            category_tokens.append(
                int(run["category_usage"].get(category, {}).get("total_tokens", 0))
            )
        category_summary[category] = {
            "mean_score": _mean(category_scores),
            "mean_total_tokens": _mean([float(value) for value in category_tokens]),
        }

    return {
        "runs": len(runs),
        "mean_score": _mean(scores),
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "mean_total_tokens": _mean([float(value) for value in totals]),
        "min_total_tokens": min(totals) if totals else 0,
        "max_total_tokens": max(totals) if totals else 0,
        "mean_prompt_tokens": _mean([float(value) for value in prompt]),
        "mean_completion_tokens": _mean([float(value) for value in completion]),
        "category_summary": category_summary,
        "repeated_failures": dict(sorted(repeated_failures.items())),
    }


def _print_run_table(runs: list[dict[str, Any]]) -> None:
    print("# Repeat Compare")
    print()
    print("## Per Run")
    print()
    print("| label | run | score | total_tokens | prompt_tokens | completion_tokens | failed_task_ids |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for run in runs:
        failures = ", ".join(run["failed_task_ids"]) if run["failed_task_ids"] else "-"
        print(
            f"| {run['label']} | {run['run']} | {run['passed']}/{run['total']} "
            f"({run['score'] * 100:.1f}%) | {run['total_tokens']} | "
            f"{run['prompt_tokens']} | {run['completion_tokens']} | {failures} |"
        )
    print()


def _print_aggregate(aggregates: dict[str, dict[str, Any]]) -> None:
    print("## Aggregate")
    print()
    print("| label | runs | mean_score | min_score | max_score | mean_total_tokens | min_tokens | max_tokens | mean_prompt | mean_completion | repeated_failures |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for label, summary in sorted(aggregates.items()):
        failures = ", ".join(
            f"{task_id} x{count}"
            for task_id, count in summary["repeated_failures"].items()
            if count > 1
        ) or "-"
        print(
            f"| {label} | {summary['runs']} | {summary['mean_score'] * 100:.1f}% | "
            f"{summary['min_score'] * 100:.1f}% | {summary['max_score'] * 100:.1f}% | "
            f"{summary['mean_total_tokens']:.1f} | {summary['min_total_tokens']} | "
            f"{summary['max_total_tokens']} | {summary['mean_prompt_tokens']:.1f} | "
            f"{summary['mean_completion_tokens']:.1f} | {failures} |"
        )
    print()


def _print_category_comparison(aggregates: dict[str, dict[str, Any]]) -> None:
    labels = list(aggregates)
    categories = sorted(
        {
            category
            for summary in aggregates.values()
            for category in summary["category_summary"]
        }
    )
    print("## Per-Category Means")
    print()
    print("| category | " + " | ".join(f"{label} score | {label} tokens" for label in labels) + " |")
    print("| --- | " + " | ".join("---: | ---:" for _ in labels) + " |")
    for category in categories:
        values: list[str] = []
        for label in labels:
            row = aggregates[label]["category_summary"].get(
                category,
                {"mean_score": 0.0, "mean_total_tokens": 0.0},
            )
            values.extend([f"{row['mean_score'] * 100:.1f}%", f"{row['mean_total_tokens']:.1f}"])
        print(f"| {category} | " + " | ".join(values) + " |")
    print()

    if len(labels) == 2:
        first, second = labels
        first_summary = aggregates[first]["category_summary"]
        second_summary = aggregates[second]["category_summary"]
        loses: list[str] = []
        saves: list[str] = []
        for category in categories:
            a = first_summary.get(category, {"mean_score": 0.0, "mean_total_tokens": 0.0})
            b = second_summary.get(category, {"mean_score": 0.0, "mean_total_tokens": 0.0})
            if b["mean_score"] < a["mean_score"]:
                loses.append(category)
            if b["mean_score"] >= a["mean_score"] and b["mean_total_tokens"] < a["mean_total_tokens"]:
                saves.append(category)
        print(f"- Categories where {second} loses accuracy vs {first}: {', '.join(loses) if loses else 'none'}")
        print(f"- Categories where {second} saves tokens while matching accuracy: {', '.join(saves) if saves else 'none'}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeat forced-model local comparisons.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "repeat_compare"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.models) != len(args.labels):
        raise SystemExit("--models and --labels must have the same length")
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    tasks = validate_tasks(load_json(Path(args.tasks)))
    expected = load_json(Path(args.expected))
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for run_index in range(1, args.runs + 1):
        for label, model in zip(args.labels, args.models):
            print(f"running label={label} run={run_index} model={model}", file=sys.stderr)
            runs.append(_run_one(label, model, run_index, tasks, expected, output_dir))

    aggregates = {
        label: _aggregate_runs([run for run in runs if run["label"] == label])
        for label in args.labels
    }
    summary = {"runs": runs, "aggregates": aggregates}
    summary_path = output_dir / "repeat_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    _print_run_table(runs)
    _print_aggregate(aggregates)
    _print_category_comparison(aggregates)
    print(f"wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
