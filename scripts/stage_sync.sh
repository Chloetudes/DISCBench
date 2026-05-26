#!/usr/bin/env bash
# Sync canonical JSONL ↔ staging Excel for evaluation stages.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/scripts:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
ACTION="${1:-}"
case "$ACTION" in
  push) python3 -c "from lib.stage_bridge import push_jsonl_to_staging; push_jsonl_to_staging()" ;;
  pull) python3 -c "from lib.stage_bridge import pull_staging_to_jsonl; pull_staging_to_jsonl()" ;;
  *)
    echo "Usage: bash scripts/stage_sync.sh push|pull"
    exit 1
    ;;
esac
