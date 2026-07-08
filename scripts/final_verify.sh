#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-minima:local}"
VERIFY_DIR="${VERIFY_DIR:-$(pwd)/.final_verify}"

require_ignored() {
  local path="$1"
  if git check-ignore -q "$path"; then
    echo "ignored: $path"
  else
    echo "not ignored: $path" >&2
    exit 1
  fi
}

check_shape() {
  local path="$1"
  python - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(data, list):
    raise SystemExit(f"{path}: expected JSON list")
for index, item in enumerate(data):
    if set(item) != {"task_id", "answer"}:
        raise SystemExit(f"{path}: item {index} has keys {sorted(item)}")
print(f"shape ok: {path}")
PY
}

echo "compile"
python -m compileall -q src eval

echo "local sample"
mkdir -p output
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m minima.main \
  --input eval/sample_tasks.json \
  --output output/results.json
python -m json.tool output/results.json >/dev/null
check_shape output/results.json

if [[ -n "${FIREWORKS_API_KEY:-}" && -n "${FIREWORKS_BASE_URL:-}" && -n "${ALLOWED_MODELS:-}" ]]; then
  echo "live mini"
  rm -f output/live_mini_results.json
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m minima.main \
    --input eval/live_mini_tasks.json \
    --output output/live_mini_results.json
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python eval/score_outputs.py \
    --tasks eval/live_mini_tasks.json \
    --expected eval/live_mini_expected.json \
    --results output/live_mini_results.json \
    --report eval/reports/live_mini_latest.md
  python -m json.tool output/live_mini_results.json >/dev/null
  check_shape output/live_mini_results.json
else
  echo "live mini skipped: Fireworks env vars are not fully set"
fi

echo "docker build"
docker build --platform linux/amd64 -t "$IMAGE_NAME" .

echo "docker sample"
rm -rf "$VERIFY_DIR"
mkdir -p "$VERIFY_DIR/input" "$VERIFY_DIR/output"
cp eval/sample_tasks.json "$VERIFY_DIR/input/tasks.json"
HOST_INPUT="$VERIFY_DIR/input"
HOST_OUTPUT="$VERIFY_DIR/output"
if command -v cygpath >/dev/null 2>&1; then
  HOST_INPUT="$(cygpath -w "$HOST_INPUT")"
  HOST_OUTPUT="$(cygpath -w "$HOST_OUTPUT")"
fi
MSYS_NO_PATHCONV=1 docker run --rm \
  -e FIREWORKS_API_KEY="${FIREWORKS_API_KEY:-}" \
  -e FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-}" \
  -e ALLOWED_MODELS="${ALLOWED_MODELS:-}" \
  -v "$HOST_INPUT:/input:ro" \
  -v "$HOST_OUTPUT:/output" \
  "$IMAGE_NAME"
python -m json.tool "$VERIFY_DIR/output/results.json" >/dev/null
check_shape "$VERIFY_DIR/output/results.json"

echo "image contents"
docker run --rm --entrypoint python "$IMAGE_NAME" \
  -c "import pathlib, sys; sys.exit(0 if not pathlib.Path('/app/local_context').exists() else 1)"

echo "ignore checks"
require_ignored ".env"
require_ignored "local_context/keep.txt"
require_ignored "input/tasks.json"
require_ignored "output/results.json"
require_ignored "eval/reports/latest.md"

echo "final verification passed"
