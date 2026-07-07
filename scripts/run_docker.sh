#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-minima:phase2}"
INPUT_DIR="${INPUT_DIR:-$(pwd)/input}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/output}"

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
docker run --rm \
  -e FIREWORKS_API_KEY="${FIREWORKS_API_KEY:-}" \
  -e FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-}" \
  -e ALLOWED_MODELS="${ALLOWED_MODELS:-}" \
  -v "$INPUT_DIR:/input:ro" \
  -v "$OUTPUT_DIR:/output" \
  "$IMAGE_NAME"
