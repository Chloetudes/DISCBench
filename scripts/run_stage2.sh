#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EVALUATION_PROJECT_CONFIG="$ROOT/data/config/stage2_generate_replies.json"
bash scripts/stage_sync.sh push
python3 -m evaluation.main
bash scripts/stage_sync.sh pull
