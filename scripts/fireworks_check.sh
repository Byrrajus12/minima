#!/usr/bin/env bash
set -euo pipefail

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m minima.fireworks_check
