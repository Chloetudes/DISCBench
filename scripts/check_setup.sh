#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OK=0
MISS=0
WARN=0

check() {
  if [[ -e "$1" ]]; then
    echo "  ✓ $2"
    OK=$((OK + 1))
  else
    echo "  ✗ 缺失: $2"
    MISS=$((MISS + 1))
  fi
}

warn() {
  echo "  ⚠ $1"
  WARN=$((WARN + 1))
}

echo "=== DISCBench 项目检查 ==="
echo "项目根: $ROOT"
echo ""

python3 -c "import sys; sys.path.insert(0,'scripts'); from lib.paths import ensure_data_layout; ensure_data_layout(verbose=False)" 2>/dev/null || true

echo "[1] 评测引擎"
check "$ROOT/evaluation/main.py" "evaluation/main.py"
check "$ROOT/clients/openai_client.py" "clients/openai_client.py"

echo ""
echo "[2] API 配置（Stage 1–3 需要；离线统计不需要）"
if [[ -f "$ROOT/config.py" ]]; then
  echo "  ✓ config.py"
  OK=$((OK + 1))
else
  warn "无 config.py → cp config.example.py config.py"
fi

echo ""
echo "[3] 数据存储（JSONL 优先）"
if [[ -f "$ROOT/data/questions.jsonl" ]]; then
  echo "  ✓ data/questions.jsonl"
  OK=$((OK+1))
elif [[ -f "$ROOT/data/论文数据_all.xlsx" ]]; then
  warn "仅有 legacy xlsx → 运行: python3 scripts/consolidate_discbench_data.py"
else
  echo "  ✗ 缺少 data/questions.jsonl"
  MISS=$((MISS+1))
fi
if [[ -f "$ROOT/data/replies.jsonl" ]]; then
  echo "  ✓ data/replies.jsonl"
  OK=$((OK+1))
elif [[ -f "$ROOT/data/replies_compared_all.xlsx" ]]; then
  warn "仅有 legacy replies xlsx → 运行 consolidate_discbench_data.py"
else
  echo "  ✗ 缺少 data/replies.jsonl"
  MISS=$((MISS+1))
fi
if [[ -f "$ROOT/data/DISCbench_data.jsonl" ]]; then
  echo "  ✓ data/DISCbench_data.jsonl (open PK 200)"
  OK=$((OK+1))
else
  warn "无 data/DISCbench_data.jsonl → python3 scripts/export_discbench_open_jsonl.py"
fi
if [[ -f "$ROOT/data/public_benchmark_data.jsonl" ]]; then
  echo "  ✓ data/public_benchmark_data.jsonl (open 4×200)"
  OK=$((OK+1))
else
  warn "无 data/public_benchmark_data.jsonl → python3 scripts/export_public_open_jsonl.py"
fi

echo ""
echo "[4] 模型与提示词"
check "$ROOT/data/models.xlsx" "models.xlsx（仅含论文 Compared 12 模型 + Aimux 路由列）"
check "$ROOT/data/sysprompts/instruction_quality_evaluation.txt" "Stage1 sysprompt"
check "$ROOT/data/sysprompts/reply_evaluation.txt" "Stage3 sysprompt"
[[ -f "$ROOT/data/schema.xlsx" ]] || warn "无 data/schema.xlsx（L1/L2/L3 对齐可选）"

echo ""
echo "[5] Stage 配置"
check "$ROOT/data/config/stage1_instruction_quality.json" "stage1"
check "$ROOT/data/config/stage2_generate_replies.json" "stage2"
check "$ROOT/data/config/stage3_evaluate_replies.json" "stage3"

echo ""
echo "[6] Python 依赖"
cd "$ROOT"
_PYDEPS=""
[[ -d "$ROOT/.pydeps" ]] && _PYDEPS="$ROOT/.pydeps:"
export PYTHONPATH="${_PYDEPS}$ROOT/scripts:$ROOT"
if python3 -c "import pandas, openpyxl, scipy, matplotlib" 2>/dev/null; then
  echo "  ✓ pandas / openpyxl / scipy / matplotlib"
  OK=$((OK + 1))
else
  echo "  ✗ 缺少依赖 → pip3 install -r requirements.txt"
  MISS=$((MISS + 1))
fi

if [[ -f "$ROOT/config.py" ]]; then
  if python3 -c "import evaluation.main" 2>/dev/null; then
    echo "  ✓ Stage 1–3 引擎可导入"
    OK=$((OK + 1))
  else
    warn "Stage 引擎导入失败（检查 config.py）"
  fi
fi

echo ""
echo "=== ${OK} 通过, ${MISS} 缺失, ${WARN} 警告 ==="
[[ $MISS -eq 0 ]] || exit 1
echo "推 GitHub 前: bash scripts/preflight_release.sh"
echo "离线统计: bash scripts/run_stats.sh"
echo "完整准备: bash scripts/prepare_for_stats.sh"
exit 0
