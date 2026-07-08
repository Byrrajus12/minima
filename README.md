# minima

Compact Fireworks model-routing agent for JSON task evaluation.

The container reads `/input/tasks.json`, writes `/output/results.json`, and exits with code 0 when the run succeeds.

## Input

```json
[
  { "task_id": "t1", "prompt": "What is the capital of France?" }
]
```

## Output

```json
[
  { "task_id": "t1", "answer": "Paris" }
]
```

## Configuration

Runtime Fireworks settings are read from environment variables:

- `FIREWORKS_API_KEY`
- `FIREWORKS_BASE_URL`
- `ALLOWED_MODELS`

`ALLOWED_MODELS` is a comma-separated list. The agent selects only from those exact model strings and never hardcodes model IDs.

When none of these variables are set, minima uses a deterministic placeholder answer marked `LOCAL TEST PLACEHOLDER`. This mode is for local JSON IO smoke tests only. If any Fireworks setting is provided, all required settings must be present and real Fireworks calls are used.

## Local Run

```bash
./scripts/run_local.sh
python -m json.tool output/results.json
```

Equivalent explicit command:

```bash
PYTHONPATH=src python -m minima.main --input eval/sample_tasks.json --output output/results.json
```

## Docker Run

```bash
docker build --platform linux/amd64 -t minima:local .
cp eval/sample_tasks.json input/tasks.json
./scripts/run_docker.sh
```

With Fireworks enabled:

```bash
export FIREWORKS_API_KEY="..."
export FIREWORKS_BASE_URL="https://..."
export ALLOWED_MODELS="model-from-allowlist"
./scripts/run_docker.sh
```

## Final Verification

```bash
./scripts/final_verify.sh
```

The verification script runs local JSON checks, Docker checks, ignore checks, and image contents checks without printing secret values. If all Fireworks environment variables are present, it also runs the live mini evaluation.

## Smoke Test

```bash
./scripts/smoke_test.sh
```

The smoke test runs local JSON IO, validates JSON, builds a linux/amd64 Docker image, runs it with mounted `/input` and `/output`, validates Docker output JSON, and checks that `/app/local_context` is absent from the image.

## Fireworks Connectivity Check

Run only after all required Fireworks environment variables are set. The check sends one minimal request through the same client used by the app and does not print secret values.

```bash
./scripts/fireworks_check.sh
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m minima.fireworks_check
```
