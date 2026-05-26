#!/usr/bin/env bash
# Rebuild canonical JSONL from legacy xlsx + export open-release bundles.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
_PYDEPS=""
[[ -d "$ROOT/.pydeps" ]] && _PYDEPS="$ROOT/.pydeps:"
export PYTHONPATH="${_PYDEPS}$ROOT/scripts:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "=== [1/4] Legacy xlsx -> questions.jsonl + replies.jsonl ==="
python3 scripts/consolidate_discbench_data.py

echo ""
echo "=== [2/4] DISCBench PK 200 -> data/DISCbench_data.jsonl ==="
python3 scripts/export_discbench_open_jsonl.py

echo ""
echo "=== [3/4] Public 4x200 -> data/public_benchmark_data.jsonl ==="
python3 scripts/export_public_open_jsonl.py

echo ""
echo "=== [4/4] Encoding repair (open bundles only) ==="
python3 scripts/fix_open_jsonl_encoding.py --no-bootstrap
python3 scripts/fix_open_jsonl_encoding.py \
  --input data/public_benchmark_data.jsonl --no-bootstrap

echo ""
echo "Open data ready:"
echo "   data/DISCbench_data.jsonl        (200 PK items, 12 models)"
echo "   data/public_benchmark_data.jsonl (800 items, 8 models)"
echo "   data/questions.jsonl + data/replies.jsonl (full bank, stats)"
