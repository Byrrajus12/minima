"""Run a local eval set through one forced Fireworks model."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))

from minima.config import Config  # noqa: E402
from minima.fireworks_client import FireworksClient, FireworksClientError  # noqa: E402
from minima.task_classifier import classify_task  # noqa: E402
from minima.validators import validate_results, validate_tasks  # noqa: E402

from report_usage import _aggregate, _load_usage, _token_value  # noqa: E402
from score_outputs import build_report, load_json, score_answer  # noqa: E402


def _slug(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "model"


def _load_config(model: str) -> Config:
    api_key = os.getenv("FIREWORKS_API_KEY")
    base_url = os.getenv("FIREWORKS_BASE_URL")
    missing = [
        name
        for name, value in (
            ("FIREWORKS_API_KEY", api_key),
            ("FIREWORKS_BASE_URL", base_url),
        )
        if not value
    ]
    if missing:
        names = ", ".join(missing)
        raise SystemExit(f"Missing required Fireworks environment variables: {names}")

    return Config(
        fireworks_api_key=api_key,
        fireworks_base_url=str(base_url).rstrip("/"),
        allowed_models=(model,),
        placeholder_mode=False,
    )


def _write_results(path: Path, results: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_results(results)
    path.write_text(json.dumps(validated, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _run_forced_model(
    tasks: list[dict[str, str]],
    model: str,
    results_path: Path,
    stderr_path: Path,
) -> None:
    client = FireworksClient(_load_config(model))
    results: list[dict[str, str]] = []
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    previous_usage = os.environ.get("MINIMA_LOG_USAGE")
    os.environ["MINIMA_LOG_USAGE"] = "1"
    try:
        with stderr_path.open("w", encoding="utf-8") as stderr_handle:
            with contextlib.redirect_stderr(stderr_handle):
                for task in tasks:
                    task_id = task["task_id"]
                    prompt = task["prompt"]
                    category = classify_task(prompt)
                    try:
                        answer = client.answer(
                            prompt=prompt,
                            category=category,
                            model=model,
                            task_id=task_id,
                        )
                    except FireworksClientError as exc:
                        print(
                            f"compare model_failed task_id={task_id} "
                            f"category={category} model={model} reason={exc}",
                            file=sys.stderr,
                        )
                        answer = ""
                    results.append({"task_id": task_id, "answer": answer})
    finally:
        if previous_usage is None:
            os.environ.pop("MINIMA_LOG_USAGE", None)
        else:
            os.environ["MINIMA_LOG_USAGE"] = previous_usage

    _write_results(results_path, results)


def _run_command(command: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _print_text(text: str, *, stderr: bool = False) -> None:
    if not text:
        return
    stream = sys.stderr if stderr else sys.stdout
    encoding = stream.encoding or "utf-8"
    stream.buffer.write(text.encode(encoding, errors="replace"))
    stream.flush()


def _score_details(
    tasks: list[dict[str, str]],
    expected: list[dict[str, Any]],
    results: list[dict[str, str]],
) -> tuple[Counter[str], Counter[str], list[str]]:
    expected_by_id = {item["task_id"]: item for item in expected}
    result_by_id = {item["task_id"]: item for item in results}
    total_by_category: Counter[str] = Counter()
    passed_by_category: Counter[str] = Counter()
    failures: list[str] = []

    for task in tasks:
        task_id = task["task_id"]
        expected_item = expected_by_id.get(task_id)
        if not expected_item:
            failures.append(task_id)
            continue
        category = str(expected_item["category"])
        total_by_category[category] += 1
        result = result_by_id.get(task_id)
        answer = result.get("answer", "") if result else ""
        if not isinstance(answer, str) or not answer.strip():
            failures.append(task_id)
            continue
        ok, _ = score_answer(answer, expected_item["scoring"])
        if ok:
            passed_by_category[category] += 1
        else:
            failures.append(task_id)

    return total_by_category, passed_by_category, failures


def _usage_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "prompt_tokens": sum(_token_value(record, "prompt_tokens") for record in records),
        "completion_tokens": sum(_token_value(record, "completion_tokens") for record in records),
        "total_tokens": sum(_token_value(record, "total_tokens") for record in records),
    }


def _print_compact_summary(
    label: str,
    model: str,
    tasks: list[dict[str, str]],
    expected: list[dict[str, Any]],
    results: list[dict[str, str]],
    usage_records: list[dict[str, Any]],
) -> None:
    total_by_category, passed_by_category, failures = _score_details(tasks, expected, results)
    totals = _usage_totals(usage_records)
    task_count = len(tasks)
    passed = task_count - len(failures)
    pass_rate = passed / task_count * 100 if task_count else 0.0
    by_category = _aggregate(usage_records, ("category",))
    ranked_categories = sorted(
        by_category.items(),
        key=lambda item: item[1]["total_tokens"],
        reverse=True,
    )

    print("# Forced Model Summary")
    print()
    print(f"- Label: {label}")
    print(f"- Model: {model}")
    print(f"- Pass rate: {pass_rate:.1f}%")
    print(f"- Total tasks: {task_count}")
    print(f"- Failures: {len(failures)}")
    print(f"- Prompt tokens: {totals['prompt_tokens']}")
    print(f"- Completion tokens: {totals['completion_tokens']}")
    print(f"- Total tokens: {totals['total_tokens']}")
    print()
    print("## Accuracy And Tokens By Category")
    print()
    print("| category | passed | total | pass_rate | prompt_tokens | completion_tokens | total_tokens | avg_total |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for category in sorted(total_by_category):
        total = total_by_category[category]
        passed_category = passed_by_category[category]
        rate = passed_category / total * 100 if total else 0.0
        usage = by_category.get(
            (category,),
            {"count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        count = usage["count"]
        avg_total = usage["total_tokens"] / count if count else 0.0
        print(
            f"| {category} | {passed_category} | {total} | {rate:.1f}% | "
            f"{usage['prompt_tokens']} | {usage['completion_tokens']} | "
            f"{usage['total_tokens']} | {avg_total:.1f} |"
        )
    print()
    print("## Top Token Categories")
    print()
    for category, usage in ranked_categories[:5]:
        print(f"- {category[0]}: {usage['total_tokens']} total tokens")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare one forced model on a local eval set.")
    parser.add_argument("--tasks", required=True, help="Path to tasks JSON.")
    parser.add_argument("--expected", required=True, help="Path to expected scoring JSON.")
    parser.add_argument("--model", required=True, help="Fireworks model string to force for every task.")
    parser.add_argument("--label", required=True, help="Short label for output filenames.")
    parser.add_argument("--output-dir", default=str(ROOT / "output"), help="Directory for result and stderr files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    label = _slug(args.label)
    tasks_path = Path(args.tasks)
    expected_path = Path(args.expected)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    results_path = output_dir / f"compare_{label}_results.json"
    stderr_path = output_dir / f"compare_{label}_stderr.txt"
    report_path = ROOT / "eval" / "reports" / f"compare_{label}.md"

    tasks = validate_tasks(load_json(tasks_path))
    expected = load_json(expected_path)

    _run_forced_model(tasks, args.model, results_path, stderr_path)

    score_code, score_stdout, score_stderr = _run_command(
        [
            sys.executable,
            "eval/score_outputs.py",
            "--tasks",
            str(tasks_path),
            "--expected",
            str(expected_path),
            "--results",
            str(results_path),
            "--report",
            str(report_path),
        ]
    )
    _print_text(score_stdout)
    _print_text(score_stderr, stderr=True)

    usage_code, usage_stdout, usage_stderr = _run_command(
        [sys.executable, "eval/report_usage.py", str(stderr_path)]
    )
    _print_text(usage_stdout)
    _print_text(usage_stderr, stderr=True)

    results = validate_results(load_json(results_path))
    usage_records = _load_usage(stderr_path)
    _print_compact_summary(label, args.model, tasks, expected, results, usage_records)

    if usage_code != 0:
        return usage_code
    if score_code != 0:
        print("compare_models note: scoring found failures; comparison output was still written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
