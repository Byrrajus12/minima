#!/usr/bin/env bash
set -euo pipefail

mkdir -p output
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m minima.main --input eval/sample_tasks.json --output output/results.json
