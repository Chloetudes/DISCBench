#!/usr/bin/env bash
# Stage1 第二轮：1300 题全量指令质量评估 → 题目表 sheet「数据对齐_2」
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EVALUATION_PROJECT_CONFIG="$ROOT/data/config/stage1_instruction_quality_round2.json"
export EVALUATION_JUDGE_PROVIDER="${EVALUATION_JUDGE_PROVIDER:-openai}"
export EVALUATION_JUDGE_MODEL="${EVALUATION_JUDGE_MODEL:-gpt-5.4-2026-03-05}"

echo "=== [1/3] 准备 sheet1 快照 + sheet2 ==="
python3 scripts/prepare_stage1_round2_sheets.py --sync-all

echo ""
echo "=== [2/3] Stage1 Round2 API（1300 题）==="
python3 -m evaluation.main

echo ""
echo "=== [3/3] 双轮难度分取平均写回 sheet1 ==="
python3 scripts/merge_iq_dual_sheet_avg.py --sync-all

echo ""
echo "✅ Round2 完成。请运行: python3 scripts/merge_iq_dual_sheet_avg.py --sync-all && python3 scripts/enrich_questions_table.py --sync-all && bash scripts/run_stats.sh"
