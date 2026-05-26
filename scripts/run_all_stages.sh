#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "$ROOT/scripts/setup.sh"
bash "$ROOT/scripts/check_setup.sh"
bash "$ROOT/scripts/run_stage1.sh"
bash "$ROOT/scripts/run_stage2.sh"
bash "$ROOT/scripts/run_stage3.sh"
bash "$ROOT/scripts/run_stats.sh"
echo "✓ 全流程完成"
