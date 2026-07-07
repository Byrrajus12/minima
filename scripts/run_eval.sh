#!/usr/bin/env bash
set -euo pipefail

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python eval/generate_tasks.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python eval/run_eval.py --skip-generate
