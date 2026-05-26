#!/usr/bin/env bash
# 等待 Stage1 Round2 进程结束后，合并双轮难度分并刷新统计
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PID="${1:?用法: wait_round2_then_merge.sh <stage1_pid>}"
LOG="$ROOT/output/stage1_quality/round2_post.log"

echo "[wait] 等待 Stage1 Round2 PID=$PID ..." | tee -a "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  sleep 60
done

echo "[merge] $(date -Iseconds) 合并双轮难度分" | tee -a "$LOG"
python3 scripts/merge_iq_dual_sheet_avg.py --sync-all 2>&1 | tee -a "$LOG"

echo "[enrich] $(date -Iseconds) 回填题目表衍生列" | tee -a "$LOG"
python3 scripts/enrich_questions_table.py --sync-all 2>&1 | tee -a "$LOG"

echo "[stats] $(date -Iseconds) 重新生成综合统计" | tee -a "$LOG"
bash scripts/run_stats.sh 2>&1 | tee -a "$LOG"

echo "[done] $(date -Iseconds) Round2 全流程完成" | tee -a "$LOG"
