"""Run the local evaluation harness."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "eval" / "tasks.json"
DEFAULT_EXPECTED = ROOT / "eval" / "expected.json"
DEFAULT_RESULTS = ROOT / "output" / "eval_results.json"
DEFAULT_REPORT = ROOT / "eval" / "reports" / "latest.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local evaluation.")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--expected", default=str(DEFAULT_EXPECTED))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Use the existing task and expected files.",
    )
    return parser.parse_args()


def run_command(command: list[str], env: dict[str, str]) -> int:
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    return completed.returncode


def main() -> int:
    args = parse_args()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src")

    if not args.skip_generate:
        code = run_command([sys.executable, "eval/generate_tasks.py"], env)
        if code != 0:
            return code

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    code = run_command(
        [
            sys.executable,
            "-m",
            "minima.main",
            "--input",
            args.tasks,
            "--output",
            args.results,
        ],
        env,
    )
    if code != 0:
        return code

    code = run_command([sys.executable, "-m", "json.tool", args.results], env)
    if code != 0:
        return code

    return run_command(
        [
            sys.executable,
            "eval/score_outputs.py",
            "--tasks",
            args.tasks,
            "--expected",
            args.expected,
            "--results",
            args.results,
            "--report",
            args.report,
        ],
        env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
