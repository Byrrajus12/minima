"""LLM-judge local audit rows without exposing source labels in judge prompts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


SYSTEM_PROMPT = """You are an evaluator for a general-purpose AI agent benchmark.

Your job is to decide whether the submitted answer should receive credit for the task.

You will be given:

* task_id
* category
* difficulty
* the original user prompt
* reference expectations / ground-truth metadata
* the submitted answer

Important judging principles:

1. Judge the answer against the user prompt and expected intent, not against exact wording.
2. The reference expectations are guidance for correctness. They are not always the only acceptable wording.
3. Accept semantically equivalent answers.
4. Do not require exact format unless the prompt requested a specific format.
5. Do require all explicitly requested information.
6. Penalize answers that are incomplete, contradictory, hallucinated, or ignore constraints.
7. Penalize answers that are technically correct but fail a required format or length constraint.
8. Penalize answers that provide code that is syntactically invalid or does not satisfy the requested behavior.
9. Be stricter for tasks with exact constraints such as math, NER labels, code behavior, and summarization length.
10. Be tolerant of concise answers when the task only requires the final result.
11. Do not reward verbose reasoning if the final answer is wrong.
12. Do not penalize harmless extra explanation unless it violates a requested format or introduces wrong information.
13. If the answer is correct but slightly different from the deterministic expected string, mark it as passing.
14. If an answer appears to satisfy only a keyword but not the real intent, mark it as failing.
15. Evaluate the submitted answer independently. Do not assume prior pass/fail status.

Category-specific rubric:

Factual knowledge:

* Pass if the answer accurately addresses the question and includes the key facts requested.
* Fail if it omits a required part, gives a wrong fact, or answers only part of a multi-detail question.
* Concise answers are acceptable.

Mathematical reasoning:

* Pass if the final numeric result is correct and the answer is understandable.
* Accept equivalent numeric forms, such as fractions, decimals, percentages, or units, when appropriate.
* Fail if the final value is wrong, the wrong quantity is answered, or required units/conditions are missing.
* Reasoning steps are helpful but not required unless the prompt asks for them.

Sentiment classification:

* Pass if the label is one of positive, negative, neutral, or mixed and matches the overall sentiment.
* Use "mixed" when substantial positive and negative sentiment are both present.
* If the prompt asks for a reason or the benchmark expectation includes justification, require a brief plausible justification.
* Fail if the label is wrong, missing, or the reason contradicts the label.

Summarization:

* Pass if the answer preserves the key points and follows requested length/format constraints.
* For "exactly one sentence," require one sentence.
* For "exactly N bullets," require N bullets or clearly separated bullet-like items.
* For word limits, be reasonably strict.
* Fail if key requested facts are missing, unsupported details are added, or the format constraint is violated.

Named entity recognition:

* Pass if the required entities are extracted with correct type labels.
* Accept reasonable capitalization and type-name variants such as ORG/organization.
* Fail if important entities are missing, mislabeled, or if extra non-entities create confusion.
* Do not require exact ordering unless requested.

Code debugging:

* Pass if the answer identifies or fixes the bug and provides a corrected implementation that satisfies the stated purpose.
* Accept alternative correct fixes.
* Fail if the proposed code is invalid, still has the bug, omits the required function, or changes the intended behavior.
* A bug explanation is helpful but not required unless the prompt asks for it.

Logical / deductive reasoning:

* Pass if the final answer satisfies all constraints.
* Accept concise answers if the result is clear.
* Fail if any constraint is violated, if the answer is ambiguous, or if it does not answer the asked question.

Code generation:

* Pass if the code implements the requested function and handles the specified edge cases.
* Accept equivalent implementations.
* Fail if code is syntactically invalid, missing the required function, returns the wrong type, ignores edge cases, or includes unrelated code that would break execution.

Return JSON only. No markdown. No extra text.

Schema:
{
"pass": true or false,
"score": integer from 0 to 5,
"severity": "none" | "minor" | "major",
"confidence": "low" | "medium" | "high",
"reason": "one or two concise sentences explaining the judgment",
"expected_missing": ["list any missing required elements"],
"incorrect_or_unsupported": ["list any wrong or unsupported elements"],
"format_issue": "brief description or empty string",
"would_hidden_judge_accept": true or false
}

Scoring guide:
5 = fully correct and follows constraints
4 = correct with minor harmless issue
3 = partially correct but missing a meaningful requirement
2 = mostly wrong, but contains some relevant information
1 = barely relevant
0 = invalid, empty, refusal, or completely wrong

Passing threshold:

* Mark "pass": true for scores 4 or 5.
* Mark "pass": false for scores 0, 1, 2, or 3."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: malformed JSONL: {exc}") from exc
            if not isinstance(item, dict):
                raise SystemExit(f"{path}:{line_number}: row was not a JSON object")
            rows.append(item)
    return rows


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        task_id = row.get("task_id")
        if isinstance(task_id, str) and not row.get("parse_error"):
            existing[task_id] = row
    return existing


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _judge_user_prompt(row: dict[str, Any]) -> str:
    expected_json = json.dumps(row.get("expected", {}), ensure_ascii=True, sort_keys=True)
    return (
        "Evaluate this benchmark answer.\n\n"
        "task_id:\n"
        f"{row.get('task_id', '')}\n\n"
        "category:\n"
        f"{row.get('category', '')}\n\n"
        "difficulty:\n"
        f"{row.get('difficulty', '')}\n\n"
        "original_prompt:\n"
        f"{row.get('prompt', '')}\n\n"
        "reference_expectations:\n"
        f"{expected_json}\n\n"
        "submitted_answer:\n"
        f"{row.get('answer', '')}\n\n"
        "Decide whether the answer should receive credit under the benchmark rubric.\n\n"
        "Do not require exact wording unless the original prompt required exact wording.\n"
        "Do not assume the reference expectation is the only valid phrasing.\n"
        "Judge semantic correctness, completeness, and instruction-following.\n\n"
        "Return JSON only."
    )


def _default_base_url() -> str:
    return os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1").rstrip("/")


def _extract_error_body(exc: urllib.error.HTTPError) -> str:
    detail = exc.read().decode("utf-8", errors="replace").strip()
    return detail[:1000] if detail else "no response body"


def _post_chat(
    row: dict[str, Any],
    judge_model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    sleep_seconds: float,
) -> str:
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY is not set")

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _judge_user_prompt(row)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")
    endpoint = f"{_default_base_url()}/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("judge response did not include choices")
            first = choices[0]
            if not isinstance(first, dict):
                raise RuntimeError("judge choice was not an object")
            message = first.get("message")
            if not isinstance(message, dict):
                raise RuntimeError("judge choice did not include a message")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("judge returned empty content")
            return content.strip()
        except urllib.error.HTTPError as exc:
            last_error = RuntimeError(f"Fireworks HTTP {exc.code}: {_extract_error_body(exc)}")
        except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(sleep_seconds * (attempt + 1))
    raise RuntimeError(str(last_error) if last_error else "judge request failed")


def _parse_judge_json(raw: str) -> tuple[dict[str, Any] | None, str]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None, "no JSON object found"
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return None, f"JSON parse failed: {exc.msg}"
    if not isinstance(data, dict):
        return None, "judge JSON was not an object"
    return data, ""


def _coerce_judge(data: dict[str, Any]) -> dict[str, Any]:
    score = data.get("score")
    if not isinstance(score, int):
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 0
    score = max(0, min(5, score))

    passed = data.get("pass")
    if not isinstance(passed, bool):
        passed = score >= 4

    severity = data.get("severity")
    if severity not in {"none", "minor", "major"}:
        severity = "none" if passed else "major"

    confidence = data.get("confidence")
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"

    hidden = data.get("would_hidden_judge_accept")
    if not isinstance(hidden, bool):
        hidden = passed

    def string_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if value in (None, ""):
            return []
        return [str(value)]

    return {
        "pass": bool(passed),
        "score": score,
        "severity": severity,
        "confidence": confidence,
        "reason": str(data.get("reason", "")),
        "expected_missing": string_list(data.get("expected_missing")),
        "incorrect_or_unsupported": string_list(data.get("incorrect_or_unsupported")),
        "format_issue": str(data.get("format_issue", "")),
        "would_hidden_judge_accept": bool(hidden),
    }


def _judge_one(
    label: str,
    row: dict[str, Any],
    judge_model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    raw = ""
    parse_error = ""
    try:
        raw = _post_chat(row, judge_model, temperature, max_tokens, retries, sleep_seconds)
        parsed, parse_error = _parse_judge_json(raw)
    except Exception as exc:
        parsed = None
        parse_error = str(exc)

    if parsed is None:
        judge = {
            "pass": False,
            "score": 0,
            "severity": "major",
            "confidence": "low",
            "reason": "Judge response could not be parsed or request failed.",
            "expected_missing": [],
            "incorrect_or_unsupported": [],
            "format_issue": "",
            "would_hidden_judge_accept": False,
        }
    else:
        judge = _coerce_judge(parsed)

    return {
        "label": label,
        "task_id": row.get("task_id"),
        "category": row.get("category"),
        "difficulty": row.get("difficulty"),
        "deterministic_passed": row.get("passed"),
        "deterministic_failure_reasons": row.get("failure_reasons", []),
        "llm_pass": judge["pass"],
        "llm_score": judge["score"],
        "llm_severity": judge["severity"],
        "llm_confidence": judge["confidence"],
        "llm_reason": judge["reason"],
        "expected_missing": judge["expected_missing"],
        "incorrect_or_unsupported": judge["incorrect_or_unsupported"],
        "format_issue": judge["format_issue"],
        "would_hidden_judge_accept": judge["would_hidden_judge_accept"],
        "judge_model": judge_model,
        "judge_raw_response": raw,
        "parse_error": parse_error,
        "answer": row.get("answer"),
        "prompt": row.get("prompt"),
        "expected": row.get("expected"),
        "notes": row.get("notes"),
        "source_model": row.get("model"),
        "prompt_tokens": row.get("prompt_tokens"),
        "completion_tokens": row.get("completion_tokens"),
        "total_tokens": row.get("total_tokens"),
        "finish_reason": row.get("finish_reason"),
    }


def _token_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    return value if isinstance(value, int) else 0


def _summarize(label: str, rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    llm_passed = sum(1 for row in rows if row.get("llm_pass") is True)
    det_passed = sum(1 for row in rows if row.get("deterministic_passed") is True)
    agreement = sum(
        1
        for row in rows
        if bool(row.get("llm_pass")) == bool(row.get("deterministic_passed"))
    )
    false_pos = [row for row in rows if row.get("deterministic_passed") is True and row.get("llm_pass") is False]
    false_neg = [row for row in rows if row.get("deterministic_passed") is False and row.get("llm_pass") is True]
    categories = sorted({str(row.get("category")) for row in rows})
    difficulties = sorted({str(row.get("difficulty")) for row in rows})

    lines = [
        f"# Judge Summary: {label}",
        "",
        f"- Total rows judged: {total}",
        f"- LLM pass rate: {(llm_passed / total * 100 if total else 0):.1f}%",
        f"- Deterministic pass rate: {(det_passed / total * 100 if total else 0):.1f}%",
        f"- Agreement rate: {(agreement / total * 100 if total else 0):.1f}%",
        f"- False positives: {len(false_pos)}",
        f"- False negatives: {len(false_neg)}",
        f"- Original answer tokens: {sum(_token_int(row, 'total_tokens') for row in rows)}",
        "",
        "## Pass Rate By Category",
        "",
        "| category | llm_passed | total | pass_rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category in categories:
        items = [row for row in rows if str(row.get("category")) == category]
        ok = sum(1 for row in items if row.get("llm_pass") is True)
        lines.append(f"| {category} | {ok} | {len(items)} | {(ok / len(items) * 100 if items else 0):.1f}% |")

    lines.extend(["", "## Pass Rate By Difficulty", "", "| difficulty | llm_passed | total | pass_rate |", "| --- | ---: | ---: | ---: |"])
    for difficulty in difficulties:
        items = [row for row in rows if str(row.get("difficulty")) == difficulty]
        ok = sum(1 for row in items if row.get("llm_pass") is True)
        lines.append(f"| {difficulty} | {ok} | {len(items)} | {(ok / len(items) * 100 if items else 0):.1f}% |")

    fp_by_cat = Counter(str(row.get("category")) for row in false_pos)
    fn_by_cat = Counter(str(row.get("category")) for row in false_neg)
    lines.extend(["", "## False Positives By Category", ""])
    lines.extend([f"- {cat}: {fp_by_cat[cat]}" for cat in sorted(fp_by_cat)] or ["- none"])
    lines.extend(["", "## False Negatives By Category", ""])
    lines.extend([f"- {cat}: {fn_by_cat[cat]}" for cat in sorted(fn_by_cat)] or ["- none"])

    reason_counter: Counter[str] = Counter()
    failed_by_category: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("llm_pass") is False:
            reason = str(row.get("llm_reason", "")).strip()
            reason_counter[reason[:120] or "no reason"] += 1
            failed_by_category[str(row.get("category"))].append(str(row.get("task_id")))
    lines.extend(["", "## Common LLM Failure Reasons", ""])
    for reason, count in reason_counter.most_common(10):
        lines.append(f"- {count}x {reason}")
    if not reason_counter:
        lines.append("- none")

    lines.extend(["", "## Failed Task IDs By Category", ""])
    for category in sorted(failed_by_category):
        lines.append(f"- {category}: {', '.join(failed_by_category[category])}")
    if not failed_by_category:
        lines.append("- none")

    tokens_by_category: dict[str, int] = defaultdict(int)
    tokens_by_model: dict[str, int] = defaultdict(int)
    for row in rows:
        tokens_by_category[str(row.get("category"))] += _token_int(row, "total_tokens")
        tokens_by_model[str(row.get("source_model") or "-")] += _token_int(row, "total_tokens")
    lines.extend(["", "## Token Usage By Category", ""])
    for category in sorted(tokens_by_category):
        lines.append(f"- {category}: {tokens_by_category[category]}")
    lines.extend(["", "## Token Usage By Model", ""])
    for model in sorted(tokens_by_model):
        lines.append(f"- {model}: {tokens_by_model[model]}")

    lines.extend(["", "## Estimated Official-Risk Categories", ""])
    for category in categories:
        items = [row for row in rows if str(row.get("category")) == category]
        ok = sum(1 for row in items if row.get("llm_pass") is True)
        rate = ok / len(items) if items else 0.0
        if rate >= 0.95:
            verdict = "low local risk"
        elif rate >= 0.85:
            verdict = "medium local risk"
        else:
            verdict = "high local risk"
        lines.append(f"- {category}: {verdict} ({ok}/{len(items)})")

    lines.extend(["", "## Recommendation", ""])
    lines.append("Use categories with high LLM pass rate and low token usage as candidates for routing experiments; avoid categories with judge-visible format or completeness failures.")
    lines.append("")
    return "\n".join(lines)


def _comparison_summary(all_rows: dict[str, list[dict[str, Any]]]) -> str:
    labels = list(all_rows)
    categories = sorted({str(row.get("category")) for rows in all_rows.values() for row in rows})
    lines = ["# Judge Comparison Summary", "", "| label | llm_pass_rate | deterministic_pass_rate | total_tokens | pass_per_1k_tokens |", "| --- | ---: | ---: | ---: | ---: |"]
    for label, rows in all_rows.items():
        total = len(rows)
        llm_ok = sum(1 for row in rows if row.get("llm_pass") is True)
        det_ok = sum(1 for row in rows if row.get("deterministic_passed") is True)
        tokens = sum(_token_int(row, "total_tokens") for row in rows)
        efficiency = llm_ok / (tokens / 1000) if tokens else 0.0
        lines.append(f"| {label} | {(llm_ok / total * 100 if total else 0):.1f}% | {(det_ok / total * 100 if total else 0):.1f}% | {tokens} | {efficiency:.2f} |")

    lines.extend(["", "## LLM Pass Rate By Category", "", "| category | " + " | ".join(labels) + " |", "| --- | " + " | ".join("---:" for _ in labels) + " |"])
    category_rates: dict[str, dict[str, float]] = defaultdict(dict)
    category_tokens: dict[str, dict[str, int]] = defaultdict(dict)
    for category in categories:
        values: list[str] = []
        for label in labels:
            items = [row for row in all_rows[label] if str(row.get("category")) == category]
            ok = sum(1 for row in items if row.get("llm_pass") is True)
            rate = ok / len(items) if items else 0.0
            category_rates[category][label] = rate
            category_tokens[category][label] = sum(_token_int(row, "total_tokens") for row in items)
            values.append(f"{ok}/{len(items)} ({rate * 100:.1f}%)")
        lines.append(f"| {category} | " + " | ".join(values) + " |")

    lines.extend(["", "## Original Answer Tokens By Category", "", "| category | " + " | ".join(labels) + " |", "| --- | " + " | ".join("---:" for _ in labels) + " |"])
    for category in categories:
        lines.append(f"| {category} | " + " | ".join(str(category_tokens[category].get(label, 0)) for label in labels) + " |")

    if "kimi" in all_rows and "router" in all_rows:
        safe: list[str] = []
        loses: list[str] = []
        for category in categories:
            kimi_rate = category_rates[category].get("kimi", 0.0)
            router_rate = category_rates[category].get("router", 0.0)
            kimi_tokens = category_tokens[category].get("kimi", 0)
            router_tokens = category_tokens[category].get("router", 0)
            if kimi_rate >= router_rate and kimi_tokens < router_tokens:
                safe.append(category)
            if kimi_rate < router_rate:
                loses.append(category)
        lines.extend(["", "## Kimi Tradeoff", ""])
        lines.append(f"- Categories where Kimi matches router accuracy and saves tokens: {', '.join(safe) if safe else 'none'}")
        lines.append(f"- Categories where Kimi loses judge accuracy: {', '.join(loses) if loses else 'none'}")

    if "minimax" in all_rows and "kimi" in all_rows:
        safer: list[str] = []
        for category in categories:
            if category_rates[category].get("minimax", 0.0) > category_rates[category].get("kimi", 0.0):
                safer.append(category)
        lines.append(f"- Categories where MiniMax is safer than Kimi: {', '.join(safer) if safer else 'none'}")

    lines.extend(["", "## Summarization Risk", ""])
    if "summarization" in categories:
        lines.append("; ".join(f"{label}: {category_rates['summarization'].get(label, 0.0) * 100:.1f}%" for label in labels))
    else:
        lines.append("No summarization rows.")

    lines.extend(["", "## Recommended Smallest Routing Experiment", ""])
    lines.append("Prefer the smallest Kimi-first change among categories where Kimi matches router judge accuracy and saves tokens. If factual remains in that set, start with factual only.")
    lines.append("")
    return "\n".join(lines)


def _write_summary(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _process_label(
    input_path: Path,
    label: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output_path = output_dir / f"{label}_judge.jsonl"
    existing = _load_existing(output_path) if args.resume else {}
    source_rows = _load_jsonl(input_path)
    if args.limit is not None:
        source_rows = source_rows[: args.limit]
    pending = [row for row in source_rows if str(row.get("task_id")) not in existing]
    completed = dict(existing)

    print(f"{label}: {len(completed)} existing, {len(pending)} pending", file=sys.stderr)

    def judge(row: dict[str, Any]) -> dict[str, Any]:
        return _judge_one(
            label=label,
            row=row,
            judge_model=args.judge_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=args.retries,
            sleep_seconds=args.sleep,
        )

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(judge, row): row for row in pending}
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = future.result()
                task_id = str(result.get("task_id"))
                completed[task_id] = result
                _append_jsonl(output_path, result)
                print(f"{label}: judged {index}/{len(pending)} {task_id}", file=sys.stderr)

    ordered_rows: list[dict[str, Any]] = []
    for row in source_rows:
        task_id = str(row.get("task_id"))
        if task_id in completed:
            ordered_rows.append(completed[task_id])
    _write_jsonl(output_path, ordered_rows)
    _write_summary(output_dir / f"{label}_judge_summary.md", _summarize(label, ordered_rows))
    return ordered_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-judge local audit JSONL files.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "judge"))
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.input) != len(args.labels):
        raise SystemExit("--input and --labels must have the same length")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    for raw_path, label in zip(args.input, args.labels):
        all_rows[label] = _process_label(Path(raw_path), label, output_dir, args)
    _write_summary(output_dir / "comparison_summary.md", _comparison_summary(all_rows))
    print(_comparison_summary(all_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
