#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EVALUATION_PROJECT_CONFIG="$ROOT/data/config/stage1_instruction_quality.json"
export EVALUATION_JUDGE_PROVIDER="${EVALUATION_JUDGE_PROVIDER:-openai}"
export EVALUATION_JUDGE_MODEL="${EVALUATION_JUDGE_MODEL:-gpt-5.4-2026-03-05}"
bash scripts/stage_sync.sh push
python3 -m evaluation.main
bash scripts/stage_sync.sh pull
