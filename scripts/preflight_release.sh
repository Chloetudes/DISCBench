#!/usr/bin/env bash
# Pre-push checklist: data layout, open JSONL, stats readiness (JSONL-first).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
_PYDEPS=""
[[ -d "$ROOT/.pydeps" ]] && _PYDEPS="$ROOT/.pydeps:"
export PYTHONPATH="${_PYDEPS}$ROOT/scripts:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache}"

echo "========== DISCBench release preflight =========="
echo ""

echo ">>> [1/4] check_setup.sh"
bash scripts/check_setup.sh
echo ""

echo ">>> [2/4] Data + open JSONL audit"
python3 - <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from lib.cif_stats_common import (
    PUBLIC_SOURCES,
    analysis_cohort_qids,
    load_questions,
    load_replies_with_scores,
    ours_pk_qids,
    verify_cohort_completeness,
)
from lib.paths import (
    DISCBENCH_OPEN_JSONL,
    PUBLIC_OPEN_JSONL,
    QUESTIONS_JSONL,
    REPLIES_JSONL,
)

errors = []

def count_lines(p: Path) -> int:
    return sum(1 for line in p.open(encoding="utf-8") if line.strip())

# Canonical flat store
for p, label in [(QUESTIONS_JSONL, "questions"), (REPLIES_JSONL, "replies")]:
    if not p.is_file():
        errors.append(f"missing {p}")
    else:
        print(f"  OK {label}: {p.name} ({count_lines(p)} lines)")

q = load_questions(QUESTIONS_JSONL)
r = load_replies_with_scores(REPLIES_JSONL, q)
cov = verify_cohort_completeness(q, r)
total = cov.loc[cov["数据来源"] == "合计", "回复+评估完整"].iloc[0]
if int(total) != 1000:
    errors.append(f"cohort incomplete: {total}/1000")
else:
    print("  OK PK cohort: 1000/1000 reply+score complete")

# Open nested bundles
for p, expect in [(DISCBENCH_OPEN_JSONL, 200), (PUBLIC_OPEN_JSONL, 800)]:
    n = count_lines(p)
    if n != expect:
        errors.append(f"{p.name}: {n} lines (expected {expect})")
    else:
        print(f"  OK {p.name}: {n} lines")

pk = ours_pk_qids(q)
open_disc = {json.loads(l)["id"] for l in DISCBENCH_OPEN_JSONL.open(encoding="utf-8") if l.strip()}
open_pub = {json.loads(l)["id"] for l in PUBLIC_OPEN_JSONL.open(encoding="utf-8") if l.strip()}
pub_bank = set(q[q["source"].astype(str).isin(PUBLIC_SOURCES)]["qid"].astype(str))

if open_disc != pk:
    errors.append("DISCbench_data.jsonl ids != Ours PK 200")
else:
    print("  OK DISCbench open ids == PK 200")

if open_pub != pub_bank:
    errors.append("public_benchmark_data.jsonl ids != public bank 800")
else:
    print("  OK public open ids == 800 public questions")

repl = 0
for p in (DISCBENCH_OPEN_JSONL, PUBLIC_OPEN_JSONL):
    for line in p.open(encoding="utf-8"):
        rec = json.loads(line)
        for mr in rec.get("model_responses") or []:
            if "\ufffd" in (mr.get("reply") or ""):
                repl += 1
if repl:
    print(f"  WARN replacement char in open replies: {repl} rows (run fix_open_jsonl_encoding.py)")
else:
    print("  OK no U+FFFD in open bundles")

if errors:
    print("\nFAILED:")
    for e in errors:
        print("  -", e)
    sys.exit(1)
print("\n  Data audit passed.")
PY
echo ""

echo ">>> [3/4] verify_data_for_stats.py (console only)"
python3 scripts/verify_data_for_stats.py --no-excel
echo ""

echo ">>> [4/4] run_stats.sh (reproduce paper tables)"
bash scripts/run_stats.sh
test -f output/reports/paper_benchmark_tables.xlsx

echo ""
echo "========== All preflight checks passed =========="
echo "See docs/GITHUB_PUBLISH.md before git push."
