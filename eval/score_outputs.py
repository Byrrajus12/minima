"""Score local evaluation outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minima.task_classifier import classify_task  # noqa: E402
from minima.validators import validate_results, validate_tasks  # noqa: E402


DEFAULT_TASKS = ROOT / "eval" / "tasks.json"
DEFAULT_EXPECTED = ROOT / "eval" / "expected.json"
DEFAULT_RESULTS = ROOT / "output" / "eval_results.json"
DEFAULT_REPORT = ROOT / "eval" / "reports" / "latest.md"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def extract_numbers(text: str) -> list[float]:
    return [float(match) for match in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def contains_any(text: str, values: list[str]) -> bool:
    lower = normalize(text)
    return any(normalize(value) in lower for value in values)


def contains_label(text: str, label: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(normalize(label)) + r"(?!\w)"
    return re.search(pattern, normalize(text)) is not None


def contains_all(text: str, values: list[str]) -> list[str]:
    lower = normalize(text)
    return [value for value in values if normalize(value) not in lower]


def score_answer(answer: str, scoring: dict[str, Any]) -> tuple[bool, str]:
    kind = scoring.get("type")

    if kind == "number":
        expected = float(scoring["value"])
        tolerance = float(scoring.get("tolerance", 0))
        numbers = extract_numbers(answer)
        if any(abs(value - expected) <= tolerance for value in numbers):
            return True, "numeric answer matched"
        return False, f"expected number {expected:g}"

    if kind == "label":
        label = str(scoring["label"])
        if contains_label(answer, label):
            return True, "expected label appeared"
        return False, f"expected label {label!r}"

    if kind == "entities":
        missing = contains_all(answer, [str(value) for value in scoring.get("entities", [])])
        if not missing:
            return True, "expected entities appeared"
        return False, "missing entities: " + ", ".join(missing)

    if kind == "summary":
        missing = contains_all(answer, [str(value) for value in scoring.get("keywords", [])])
        max_words = int(scoring.get("max_words", 60))
        count = word_count(answer)
        if missing:
            return False, "missing summary keywords: " + ", ".join(missing)
        if count > max_words:
            return False, f"summary too long: {count} words > {max_words}"
        return True, "summary checks passed"

    if kind == "keywords":
        values = [str(value) for value in scoring.get("keywords", [])]
        if contains_any(answer, values):
            return True, "keyword check passed"
        return False, "none of these keywords appeared: " + ", ".join(values)

    if kind == "substrings":
        missing = contains_all(answer, [str(value) for value in scoring.get("substrings", [])])
        if not missing:
            return True, "expected substrings appeared"
        return False, "missing substrings: " + ", ".join(missing)

    return False, f"unknown scoring type {kind!r}"


def validate_ids(
    tasks: list[dict[str, str]],
    results: list[dict[str, str]],
) -> tuple[list[str], list[str], list[str]]:
    task_ids = [task["task_id"] for task in tasks]
    result_ids = [result["task_id"] for result in results]
    counts = Counter(result_ids)
    missing = [task_id for task_id in task_ids if counts[task_id] == 0]
    duplicate = sorted(task_id for task_id, count in counts.items() if count > 1)
    extra = sorted(task_id for task_id in counts if task_id not in set(task_ids))
    return missing, duplicate, extra


def build_report(
    tasks: list[dict[str, str]],
    expected: list[dict[str, Any]],
    results: list[dict[str, str]],
) -> tuple[str, int]:
    expected_by_id = {item["task_id"]: item for item in expected}
    result_by_id = {item["task_id"]: item for item in results}
    task_by_id = {item["task_id"]: item for item in tasks}
    missing, duplicate, extra = validate_ids(tasks, results)

    failures: list[str] = []
    malformed: list[str] = []
    category_total: Counter[str] = Counter()
    category_passed: Counter[str] = Counter()
    route_total: Counter[str] = Counter()
    total_scored = 0
    passed = 0

    for task in tasks:
        task_id = task["task_id"]
        expected_item = expected_by_id.get(task_id)
        if not expected_item:
            failures.append(f"- `{task_id}`: missing expected metadata")
            continue

        category = str(expected_item["category"])
        route = classify_task(task["prompt"])
        category_total[category] += 1
        route_total[route] += 1

        result = result_by_id.get(task_id)
        if not result:
            continue

        answer = result.get("answer", "")
        if not isinstance(answer, str) or not answer.strip():
            malformed.append(task_id)
            failures.append(f"- `{task_id}`: empty or malformed answer")
            continue

        total_scored += 1
        ok, reason = score_answer(answer, expected_item["scoring"])
        if ok:
            passed += 1
            category_passed[category] += 1
        else:
            failures.append(
                f"- `{task_id}` ({category}, route `{route}`): {reason}; answer: {answer[:160]!r}"
            )

    failed = len(tasks) - passed
    pass_rate = (passed / len(tasks) * 100) if tasks else 0.0

    lines = [
        "# Evaluation Report",
        "",
        f"- Total tasks: {len(tasks)}",
        f"- Passed checks: {passed}",
        f"- Failed checks: {failed}",
        f"- Scored answers: {total_scored}",
        f"- Overall pass rate: {pass_rate:.1f}%",
        "",
        "## Pass Rate by Category",
        "",
        "| Category | Passed | Total | Pass rate |",
        "| --- | ---: | ---: | ---: |",
    ]

    for category in sorted(category_total):
        cat_total = category_total[category]
        cat_passed = category_passed[category]
        rate = cat_passed / cat_total * 100 if cat_total else 0.0
        lines.append(f"| {category} | {cat_passed} | {cat_total} | {rate:.1f}% |")

    lines.extend(["", "## Route Counts", "", "| Route | Count |", "| --- | ---: |"])
    for route in sorted(route_total):
        lines.append(f"| {route} | {route_total[route]} |")

    lines.extend(
        [
            "",
            "## ID Checks",
            "",
            f"- Missing task IDs: {', '.join(missing) if missing else 'none'}",
            f"- Duplicate result task IDs: {', '.join(duplicate) if duplicate else 'none'}",
            f"- Extra result task IDs: {', '.join(extra) if extra else 'none'}",
            f"- Malformed answers: {', '.join(malformed) if malformed else 'none'}",
            "",
            "## Failure Details",
            "",
        ]
    )
    lines.extend(failures if failures else ["No failures."])
    lines.append("")

    return "\n".join(lines), 0 if not failures and not missing and not duplicate and not extra else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score local evaluation outputs.")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--expected", default=str(DEFAULT_EXPECTED))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = validate_tasks(load_json(Path(args.tasks)))
    expected = load_json(Path(args.expected))
    results = validate_results(load_json(Path(args.results)))

    report, exit_code = build_report(tasks, expected, results)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(report)

    print(report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
