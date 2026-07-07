#!/usr/bin/env bash
set -euo pipefail

mkdir -p output eval/reports

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m minima.main \
  --input eval/live_mini_tasks.json \
  --output output/live_mini_results.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python eval/score_outputs.py \
  --tasks eval/live_mini_tasks.json \
  --expected eval/live_mini_expected.json \
  --results output/live_mini_results.json \
  --report eval/reports/live_mini_latest.md
