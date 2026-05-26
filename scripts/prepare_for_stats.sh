#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
_PYDEPS=""
[[ -d "$ROOT/.pydeps" ]] && _PYDEPS="$ROOT/.pydeps:"
export PYTHONPATH="${_PYDEPS}$ROOT/scripts:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache}"

echo "=== [1/3] 题目表补全 ==="
python3 scripts/enrich_questions_table.py

echo ""
echo "=== [2/3] 数据就绪检查 ==="
python3 scripts/verify_data_for_stats.py

echo ""
echo "=== [3/3] 综合统计 ==="
bash scripts/run_stats.sh
