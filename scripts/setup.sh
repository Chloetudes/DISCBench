#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
pip3 install -r requirements.txt
if [[ ! -f config.py ]]; then
  cp config.example.py config.py
  echo ">>> 已创建 config.py，请编辑填入 API Key（仅 Stage 1–3 需要）"
fi
mkdir -p output/reports output/stage1_quality data
python3 -c "import sys; sys.path.insert(0,'scripts'); from lib.paths import ensure_data_layout; ensure_data_layout()"
bash scripts/check_setup.sh
