"""Run router and forced-model stress comparisons with audit reports."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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

from audit_run import build_audit_rows, build_summary, load_json, write_audit  # noqa: E402
from compare_models import _run_forced_model, _slug  # noqa: E402
from minima.validators import validate_results, validate_tasks  # noqa: E402


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _load_usage_by_task(path: Path) -> dict[str, dict[str, Any]]:
    from report_usage import _load_usage

    usage: dict[str, dict[str, Any]] = {}
    for record in _load_usage(path):
        task_id = record.get("task_id")
        if isinstance(task_id, str):
            usage[task_id] = record
    return usage


def _audit(
    label: str,
    tasks: list[dict[str, str]],
    expected: list[dict[str, Any]],
    results_path: Path,
    stderr_path: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    results = validate_results(load_json(results_path))
    rows = build_audit_rows(tasks, expected, results, _load_usage_by_task(stderr_path))
    audit_path = output_dir / f"{label}_audit.jsonl"
    summary_path = output_dir / f"{label}_summary.md"
    write_audit(audit_path, rows)
    summary_path.write_text(build_summary(rows), encoding="utf-8")
    return rows


def _run_router(
    tasks_path: Path,
    output_path: Path,
    stderr_path: Path,
    models: list[str],
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    env["MINIMA_LOG_USAGE"] = "1"
    env["MINIMA_LOG_ROUTING"] = "1"
    env["ALLOWED_MODELS"] = ",".join(models)
    env.pop("MINIMA_ENABLE_LOCAL_SOLVER", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "minima.main",
            "--input",
            str(tasks_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.stdout:
        (stderr_path.parent / "router_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"router run failed with exit code {completed.returncode}")


def _summarize_rows(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    by_category_total: Counter[str] = Counter(str(row["category"]) for row in rows)
    by_category_passed: Counter[str] = Counter(str(row["category"]) for row in rows if row["passed"])
    tokens_by_category: dict[str, int] = defaultdict(int)
    failures: Counter[str] = Counter()
    for row in rows:
        category = str(row["category"])
        value = row.get("total_tokens")
        if isinstance(value, int):
            tokens_by_category[category] += value
        if not row["passed"]:
            for reason in row["failure_reasons"]:
                failures[str(reason).split(":")[0]] += 1
    return {
        "label": label,
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else 0.0,
        "total_tokens": sum(value for value in tokens_by_category.values()),
        "category_total": dict(by_category_total),
        "category_passed": dict(by_category_passed),
        "tokens_by_category": dict(tokens_by_category),
        "failure_types": dict(failures),
    }


def _print_summary(summaries: list[dict[str, Any]]) -> None:
    print("# Stress Compare")
    print()
    print("| run | passed | total | pass_rate | total_tokens |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for summary in summaries:
        print(
            f"| {summary['label']} | {summary['passed']} | {summary['total']} | "
            f"{summary['pass_rate'] * 100:.1f}% | {summary['total_tokens']} |"
        )
    print()

    categories = sorted(
        {
            category
            for summary in summaries
            for category in summary["category_total"]
        }
    )
    print("## Pass Rate By Category")
    print()
    print("| category | " + " | ".join(summary["label"] for summary in summaries) + " |")
    print("| --- | " + " | ".join("---:" for _ in summaries) + " |")
    for category in categories:
        values = []
        for summary in summaries:
            total = summary["category_total"].get(category, 0)
            passed = summary["category_passed"].get(category, 0)
            values.append(f"{passed}/{total} ({(passed / total * 100 if total else 0):.1f}%)")
        print(f"| {category} | " + " | ".join(values) + " |")
    print()

    print("## Tokens By Category")
    print()
    print("| category | " + " | ".join(summary["label"] for summary in summaries) + " |")
    print("| --- | " + " | ".join("---:" for _ in summaries) + " |")
    for category in categories:
        values = [str(summary["tokens_by_category"].get(category, 0)) for summary in summaries]
        print(f"| {category} | " + " | ".join(values) + " |")
    print()

    print("## Failure Types")
    print()
    for summary in summaries:
        failures = ", ".join(
            f"{reason} x{count}" for reason, count in sorted(summary["failure_types"].items())
        ) or "none"
        print(f"- {summary['label']}: {failures}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stress comparisons and audits.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "stress"))
    parser.add_argument("--runs", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.models) != len(args.labels):
        raise SystemExit("--models and --labels must have the same length")
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    tasks_path = Path(args.tasks)
    expected_path = Path(args.expected)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = validate_tasks(load_json(tasks_path))
    expected = load_json(expected_path)
    summaries: list[dict[str, Any]] = []

    router_results = output_dir / "router_results.json"
    router_stderr = output_dir / "router_stderr.txt"
    print("running router baseline", file=sys.stderr)
    _run_router(tasks_path, router_results, router_stderr, list(args.models))
    router_rows = _audit("router", tasks, expected, router_results, router_stderr, output_dir)
    summaries.append(_summarize_rows("router", router_rows))

    for run_index in range(1, args.runs + 1):
        for label, model in zip(args.labels, args.models):
            suffix = _slug(label) if args.runs == 1 else f"{_slug(label)}_run{run_index}"
            results_path = output_dir / f"{suffix}_results.json"
            stderr_path = output_dir / f"{suffix}_stderr.txt"
            print(f"running forced label={label} run={run_index} model={model}", file=sys.stderr)
            _run_forced_model(tasks, model, results_path, stderr_path)
            rows = _audit(suffix, tasks, expected, results_path, stderr_path, output_dir)
            summaries.append(_summarize_rows(suffix, rows))

    _write_json(output_dir / "stress_compare_summary.json", summaries)
    _print_summary(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
