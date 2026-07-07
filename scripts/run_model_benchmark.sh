#!/usr/bin/env bash
set -euo pipefail

mkdir -p eval/reports

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python eval/benchmark_models.py \
  --tasks eval/live_mini_tasks.json \
  --expected eval/live_mini_expected.json
