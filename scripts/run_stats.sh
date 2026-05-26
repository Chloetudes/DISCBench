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
from lib.paths import CIF_ROOT, QUESTIONS_JSONL, REPLIES_JSONL

def rel(p):
    try:
        return p.relative_to(CIF_ROOT).as_posix()
    except ValueError:
        return str(p)

print(f"  题目: {rel(QUESTIONS_JSONL)}")
print(f"  回复: {rel(REPLIES_JSONL)}")
PY
echo ""

echo "=== [1/3] 综合统计 ==="
python3 scripts/stats/generate_comprehensive_benchmark_stats.py

echo ""
echo "=== [2/3] 模型均分 / 裁判一致性 ==="
python3 scripts/stats/generate_benchmark_source_summary_report.py

echo ""
echo "=== [3/3] 合并论文用总册 ==="
python3 scripts/stats/generate_paper_benchmark_workbook.py

echo ""
echo "✅ 输出目录: $ROOT/output/reports/"
echo "   📄 论文用总册: output/reports/paper_benchmark_tables.xlsx"
