"""Audit local eval results with richer deterministic checks."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EVAL = ROOT / "eval"
for path in (SRC, EVAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from report_usage import _aggregate, _load_usage, _token_value  # noqa: E402
from score_outputs import extract_numbers, normalize, word_count  # noqa: E402
from minima.validators import validate_results, validate_tasks  # noqa: E402


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _contains(text: str, value: str) -> bool:
    return normalize(value) in normalize(text)


def _contains_label(text: str, label: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(normalize(label)) + r"(?!\w)"
    return re.search(pattern, normalize(text)) is not None


def _sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    parts = [part for part in re.split(r"[.!?]+(?:\s+|$)", stripped) if part.strip()]
    return max(1, len(parts))


def _bullet_count(text: str) -> int:
    return sum(
        1
        for line in text.splitlines()
        if re.match(r"\s*(?:[-*]|\d+[.)]|[•?])\s+", line)
    )


def _extract_code(answer: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", answer, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return answer.strip()


def _numeric_values(text: str) -> list[float]:
    values = extract_numbers(text)
    for numerator, denominator in re.findall(r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)", text):
        denom_value = float(denominator)
        if denom_value:
            values.append(float(numerator) / denom_value)
    return values


def _python_exec(answer: str, check: dict[str, Any]) -> list[str]:
    code = _extract_code(answer)
    tests = str(check.get("tests", ""))
    function = str(check.get("function", ""))
    if function and f"def {function}" not in code:
        return [f"missing function definition {function}"]

    script = (
        "namespace = {}\n"
        "code = " + repr(code) + "\n"
        "exec(code, namespace)\n"
        "globals().update(namespace)\n"
        + tests
        + "\n"
    )
    with tempfile.TemporaryDirectory(prefix="minima_audit_") as tmpdir:
        script_path = Path(tmpdir) / "check.py"
        script_path.write_text(script, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
            check=False,
        )
    if completed.returncode == 0:
        return []
    detail = (completed.stderr or completed.stdout).strip().splitlines()
    return ["python_exec failed: " + (detail[-1] if detail else f"exit {completed.returncode}")]


def score_answer(answer: str, check: dict[str, Any]) -> tuple[bool, list[str]]:
    kind = str(check.get("type", ""))
    failures: list[str] = []

    if not isinstance(answer, str) or not answer.strip():
        return False, ["empty answer"]

    if kind == "numeric_exact":
        expected = float(check["value"])
        tolerance = float(check.get("tolerance", 0))
        numbers = _numeric_values(answer)
        if any(abs(value - expected) <= tolerance for value in numbers):
            return True, []
        return False, [f"expected numeric value {expected:g}"]

    if kind == "label_with_optional_reason":
        label = str(check["label"])
        if not _contains_label(answer, label):
            failures.append(f"missing label {label}")
        if check.get("require_reason") and word_count(answer) < 4:
            failures.append("expected brief justification")
        return not failures, failures

    if kind == "contains_all":
        for value in check.get("values", []):
            if not _contains(answer, str(value)):
                failures.append(f"missing {value}")
        return not failures, failures

    if kind == "contains_any":
        values = [str(value) for value in check.get("values", [])]
        if any(_contains(answer, value) for value in values):
            return True, []
        return False, ["none of the expected values appeared: " + ", ".join(values)]

    if kind == "regex":
        pattern = str(check["pattern"])
        if re.search(pattern, answer, flags=re.IGNORECASE | re.MULTILINE):
            return True, []
        return False, [f"regex did not match: {pattern}"]

    if kind == "entity_set":
        for entity in check.get("entities", []):
            text = str(entity.get("text", ""))
            entity_type = str(entity.get("type", ""))
            if text and not _contains(answer, text):
                failures.append(f"missing entity {text}")
            if entity_type and not _contains(answer, entity_type):
                failures.append(f"missing type {entity_type} for {text}")
        return not failures, failures

    if kind == "summary_constraints":
        for keyword in check.get("keywords", []):
            if not _contains(answer, str(keyword)):
                failures.append(f"missing keyword {keyword}")
        max_words = check.get("max_words")
        if max_words is not None and word_count(answer) > int(max_words):
            failures.append(f"too many words: {word_count(answer)} > {max_words}")
        sentence_count = check.get("sentence_count")
        if sentence_count is not None and _sentence_count(answer) != int(sentence_count):
            failures.append(f"expected {sentence_count} sentence(s), got {_sentence_count(answer)}")
        bullet_count = check.get("bullet_count")
        if bullet_count is not None and _bullet_count(answer) != int(bullet_count):
            failures.append(f"expected {bullet_count} bullet(s), got {_bullet_count(answer)}")
        return not failures, failures

    if kind == "python_exec":
        failures = _python_exec(answer, check)
        return not failures, failures

    return False, [f"unknown check type {kind}"]


def _usage_by_task(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    records = _load_usage(path)
    usage: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = record.get("task_id")
        if isinstance(task_id, str):
            usage[task_id] = record
    return usage


def build_audit_rows(
    tasks: list[dict[str, str]],
    expected: list[dict[str, Any]],
    results: list[dict[str, str]],
    usage: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_by_id = {item["task_id"]: item for item in expected}
    result_by_id = {item["task_id"]: item for item in results}
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = task["task_id"]
        expected_item = expected_by_id.get(task_id, {})
        result = result_by_id.get(task_id, {"answer": ""})
        answer = result.get("answer", "")
        check = expected_item.get("check", {})
        passed, failures = score_answer(answer, check if isinstance(check, dict) else {})
        usage_item = usage.get(task_id, {})
        rows.append(
            {
                "task_id": task_id,
                "category": expected_item.get("category", "unknown"),
                "difficulty": expected_item.get("difficulty", "unknown"),
                "prompt": task["prompt"],
                "answer": answer,
                "expected": check,
                "passed": passed,
                "failure_reasons": failures,
                "model": usage_item.get("model"),
                "prompt_tokens": usage_item.get("prompt_tokens"),
                "completion_tokens": usage_item.get("completion_tokens"),
                "total_tokens": usage_item.get("total_tokens"),
                "finish_reason": usage_item.get("finish_reason"),
                "notes": expected_item.get("notes", ""),
            }
        )
    return rows


def write_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _token_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    return value if isinstance(value, int) else 0


def build_summary(rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    category_total: Counter[str] = Counter(str(row["category"]) for row in rows)
    category_passed: Counter[str] = Counter(str(row["category"]) for row in rows if row["passed"])
    difficulty_total: Counter[str] = Counter(str(row["difficulty"]) for row in rows)
    difficulty_passed: Counter[str] = Counter(str(row["difficulty"]) for row in rows if row["passed"])

    tokens_by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0})
    tokens_by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0})
    failure_types: Counter[str] = Counter()
    for row in rows:
        category = str(row["category"])
        model = str(row.get("model") or "-")
        tokens_by_category[category]["prompt"] += _token_int(row, "prompt_tokens")
        tokens_by_category[category]["completion"] += _token_int(row, "completion_tokens")
        tokens_by_category[category]["total"] += _token_int(row, "total_tokens")
        tokens_by_model[model]["prompt"] += _token_int(row, "prompt_tokens")
        tokens_by_model[model]["completion"] += _token_int(row, "completion_tokens")
        tokens_by_model[model]["total"] += _token_int(row, "total_tokens")
        if not row["passed"]:
            for reason in row["failure_reasons"]:
                failure_types[str(reason).split(":")[0]] += 1

    lines = [
        "# Audit Summary",
        "",
        f"- Total tasks: {total}",
        f"- Passed: {passed}",
        f"- Failed: {total - passed}",
        f"- Overall pass rate: {(passed / total * 100 if total else 0):.1f}%",
        f"- Total tokens: {sum(_token_int(row, 'total_tokens') for row in rows)}",
        f"- Prompt tokens: {sum(_token_int(row, 'prompt_tokens') for row in rows)}",
        f"- Completion tokens: {sum(_token_int(row, 'completion_tokens') for row in rows)}",
        "",
        "## Pass Rate By Category",
        "",
        "| category | passed | total | pass_rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category in sorted(category_total):
        count = category_total[category]
        ok = category_passed[category]
        lines.append(f"| {category} | {ok} | {count} | {(ok / count * 100 if count else 0):.1f}% |")

    lines.extend(["", "## Pass Rate By Difficulty", "", "| difficulty | passed | total | pass_rate |", "| --- | ---: | ---: | ---: |"])
    for difficulty in sorted(difficulty_total):
        count = difficulty_total[difficulty]
        ok = difficulty_passed[difficulty]
        lines.append(f"| {difficulty} | {ok} | {count} | {(ok / count * 100 if count else 0):.1f}% |")

    lines.extend(["", "## Tokens By Category", "", "| category | prompt | completion | total |", "| --- | ---: | ---: | ---: |"])
    for category in sorted(tokens_by_category):
        row = tokens_by_category[category]
        lines.append(f"| {category} | {row['prompt']} | {row['completion']} | {row['total']} |")

    lines.extend(["", "## Tokens By Model", "", "| model | prompt | completion | total |", "| --- | ---: | ---: | ---: |"])
    for model in sorted(tokens_by_model):
        row = tokens_by_model[model]
        lines.append(f"| {model} | {row['prompt']} | {row['completion']} | {row['total']} |")

    top_tokens = sorted(rows, key=lambda row: _token_int(row, "total_tokens"), reverse=True)[:10]
    lines.extend(["", "## Top Token Tasks", "", "| task_id | category | total_tokens | passed |", "| --- | --- | ---: | --- |"])
    for row in top_tokens:
        lines.append(f"| {row['task_id']} | {row['category']} | {_token_int(row, 'total_tokens')} | {row['passed']} |")

    failed_rows = [row for row in rows if not row["passed"]]
    lines.extend(["", "## Failures", "", "| task_id | category | difficulty | reasons |", "| --- | --- | --- | --- |"])
    for row in failed_rows[:30]:
        reasons = "; ".join(str(reason) for reason in row["failure_reasons"])
        lines.append(f"| {row['task_id']} | {row['category']} | {row['difficulty']} | {reasons} |")
    if len(failed_rows) > 30:
        lines.append(f"| ... | ... | ... | {len(failed_rows) - 30} more failures |")

    lines.extend(["", "## Failure Types", ""])
    if failure_types:
        for reason, count in failure_types.most_common(10):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Category Recommendations", ""])
    for category in sorted(category_total):
        count = category_total[category]
        ok = category_passed[category]
        total_tokens = tokens_by_category[category]["total"]
        rate = ok / count if count else 0.0
        if rate >= 0.95:
            verdict = "stable locally"
        elif rate >= 0.8:
            verdict = "mixed; inspect failures"
        else:
            verdict = "risky locally"
        lines.append(f"- {category}: {verdict}; {ok}/{count}; {total_tokens} tokens")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a local eval run.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--usage")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = validate_tasks(load_json(Path(args.tasks)))
    expected = load_json(Path(args.expected))
    results = validate_results(load_json(Path(args.results)))
    usage = _usage_by_task(Path(args.usage) if args.usage else None)
    rows = build_audit_rows(tasks, expected, results, usage)
    write_audit(Path(args.out), rows)
    summary = build_summary(rows)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
