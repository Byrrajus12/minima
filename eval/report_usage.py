"""Summarize Fireworks usage lines emitted by minima."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


MARKER = "minima usage "
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _token_value(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    return value if isinstance(value, int) else 0


def _load_usage(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        raw_content = path.read_bytes()
    except OSError as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc

    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            content = raw_content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        content = raw_content.decode("utf-8", errors="replace")

    lines = content.splitlines()
    line_index = 0
    while line_index < len(lines):
        line_number = line_index + 1
        line = lines[line_index]
        if MARKER not in line:
            line_index += 1
            continue
        raw = line.split(MARKER, 1)[1].strip()
        data: Any | None = None
        consumed_index = line_index
        for candidate_index in range(line_index, min(line_index + 8, len(lines))):
            if candidate_index > line_index:
                continuation = lines[candidate_index].strip()
                if MARKER in continuation or continuation.startswith("minima route"):
                    break
                raw += continuation
            try:
                data = json.loads(raw)
                consumed_index = candidate_index
                break
            except json.JSONDecodeError:
                continue

        if data is None:
            errors.append(f"{path}:{line_number}: malformed usage JSON")
            line_index += 1
            continue

        if isinstance(data, dict):
            records.append(data)
        else:
            errors.append(f"{path}:{line_number}: usage JSON was not an object")
        line_index = consumed_index + 1

    if errors:
        print("Usage parse warnings:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)

    return records


def _aggregate(
    records: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> dict[tuple[str, ...], dict[str, int]]:
    summary: dict[tuple[str, ...], dict[str, int]] = defaultdict(
        lambda: {"count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    for record in records:
        group = tuple(str(record.get(key, "-")) for key in keys)
        row = summary[group]
        row["count"] += 1
        for token_key in TOKEN_KEYS:
            row[token_key] += _token_value(record, token_key)
    return dict(summary)


def _print_table(
    title: str,
    labels: tuple[str, ...],
    rows: dict[tuple[str, ...], dict[str, int]],
) -> None:
    print(f"## {title}")
    print()
    headers = [*labels, "count", "prompt_tokens", "completion_tokens", "total_tokens", "avg_total"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" if index < len(labels) else "---:" for index in range(len(headers))) + " |")
    for group, totals in sorted(rows.items()):
        count = totals["count"]
        avg_total = totals["total_tokens"] / count if count else 0.0
        values = [
            *group,
            str(count),
            str(totals["prompt_tokens"]),
            str(totals["completion_tokens"]),
            str(totals["total_tokens"]),
            f"{avg_total:.1f}",
        ]
        print("| " + " | ".join(values) + " |")
    print()


def _print_top_calls(records: list[dict[str, Any]]) -> None:
    ranked = sorted(
        records,
        key=lambda item: _token_value(item, "total_tokens"),
        reverse=True,
    )[:10]
    print("## Top 10 Calls")
    print()
    print("| rank | task_id | category | model | total_tokens | prompt_tokens | completion_tokens | finish_reason | reasoning_retry |")
    print("| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |")
    for index, record in enumerate(ranked, start=1):
        values = [
            str(index),
            str(record.get("task_id", "-")),
            str(record.get("category", "-")),
            str(record.get("model", "-")),
            str(record.get("total_tokens", "-")),
            str(record.get("prompt_tokens", "-")),
            str(record.get("completion_tokens", "-")),
            str(record.get("finish_reason", "-")),
            str(record.get("reasoning_effort_retry", "-")),
        ]
        print("| " + " | ".join(values) + " |")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize minima usage logs.")
    parser.add_argument("paths", nargs="+", help="stderr files containing minima usage lines")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for raw_path in args.paths:
        records.extend(_load_usage(Path(raw_path)))

    if not records:
        print("No minima usage lines found in the provided file(s).", file=sys.stderr)
        return 1

    prompt_tokens = sum(_token_value(record, "prompt_tokens") for record in records)
    completion_tokens = sum(_token_value(record, "completion_tokens") for record in records)
    total_tokens = sum(_token_value(record, "total_tokens") for record in records)

    print("# Usage Report")
    print()
    print(f"- Calls: {len(records)}")
    print(f"- Prompt tokens: {prompt_tokens}")
    print(f"- Completion tokens: {completion_tokens}")
    print(f"- Total tokens: {total_tokens}")
    print()

    _print_table("By Category", ("category",), _aggregate(records, ("category",)))
    _print_table("By Model", ("model",), _aggregate(records, ("model",)))
    _print_table(
        "By Category And Model",
        ("category", "model"),
        _aggregate(records, ("category", "model")),
    )
    _print_top_calls(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
