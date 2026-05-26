#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
_PYDEPS=""
[[ -d "$ROOT/.pydeps" ]] && _PYDEPS="$ROOT/.pydeps:"
export PYTHONPATH="${_PYDEPS}$ROOT/scripts:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache}"

python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from lib.paths import DISCBENCH_ROOT, QUESTIONS_JSONL, REPLIES_JSONL

def rel(p):
    try:
        return p.relative_to(DISCBENCH_ROOT).as_posix()
    except ValueError:
        return str(p)

print(f"  题目: {rel(QUESTIONS_JSONL)}")
print(f"  回复: {rel(REPLIES_JSONL)}")
PY
echo ""

python3 scripts/stats/generate_comprehensive_benchmark_stats.py

python3 scripts/stats/generate_benchmark_source_summary_report.py

python3 scripts/stats/generate_paper_benchmark_workbook.py

# 中间汇总表不写库：仅保留论文总册 + charts（PNG 由 comprehensive 脚本生成）
rm -f "$ROOT/output/reports/comprehensive_benchmark_stats.xlsx" \
       "$ROOT/output/reports/benchmark_source_model_summary.xlsx" \
       "$ROOT/output/reports/data_ready_for_stats.xlsx"

echo ""
echo "✅ 统计产出: output/reports/paper_benchmark_tables.xlsx (+ output/reports/charts/*.png)"
