#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-minima:phase2}"
SMOKE_DIR="${SMOKE_DIR:-$(pwd)/.smoke}"

rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR/input" "$SMOKE_DIR/output"
cp eval/sample_tasks.json "$SMOKE_DIR/input/tasks.json"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m minima.main --input eval/sample_tasks.json --output output/results.json
python -m json.tool output/results.json >/dev/null

docker build --platform linux/amd64 -t "$IMAGE_NAME" .
docker run --rm \
  -v "$SMOKE_DIR/input:/input:ro" \
  -v "$SMOKE_DIR/output:/output" \
  "$IMAGE_NAME"
python -m json.tool "$SMOKE_DIR/output/results.json" >/dev/null

if docker run --rm --entrypoint python "$IMAGE_NAME" -c "import pathlib, sys; sys.exit(0 if not pathlib.Path('/app/local_context').exists() else 1)"; then
  echo "local_context not present in image"
else
  echo "local_context was copied into image" >&2
  exit 1
fi

echo "smoke test passed"
