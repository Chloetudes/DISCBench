#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build data/questions.jsonl + data/replies.jsonl from data/DISCbench_data.jsonl (nested schema)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
for p in (str(_ROOT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.cif_stats_common import OURS_SOURCE, safe_str  # noqa: E402
from lib.jsonl_store import save_questions_jsonl, save_replies_jsonl  # noqa: E402
from lib.paths import DISCBENCH_OPEN_JSONL, QUESTIONS_JSONL, REPLIES_JSONL  # noqa: E402


def nested_to_flat(src: Path = DISCBENCH_OPEN_JSONL) -> None:
    q_rows = []
    r_rows = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tax = rec.get("taxonomy") or {}
            inst = rec.get("instruction") or {}
            iq = rec.get("instruction_evaluation") or {}
            cps = rec.get("checkpoints") or {}
            q_rows.append({
                "qid": rec["id"],
                "source": OURS_SOURCE,
                "query": inst.get("query", ""),
                "rubrics": inst.get("rubrics", ""),
                "L1": tax.get("L1", ""),
                "L2": tax.get("L2", ""),
                "L3": tax.get("L3", ""),
                "difficulty_score": iq.get("difficulty_score"),
                "difficulty_tier": iq.get("difficulty_tier", ""),
                "difficulty_level": iq.get("difficulty_level", ""),
                "difficulty_desc": iq.get("difficulty_desc", ""),
                "instruction_quality_parsed_json": json.dumps(
                    iq.get("quality_analysis"), ensure_ascii=False
                )
                if iq.get("quality_analysis")
                else "",
                "checkpoint_n": cps.get("count"),
            })
            for mr in rec.get("model_responses") or []:
                r_rows.append({
                    "qid": rec["id"],
                    "model": mr.get("model"),
                    "reply": mr.get("reply", ""),
                    "source": OURS_SOURCE,
                    "1_score": mr.get("score_round1"),
                    "3_score": mr.get("score_round3"),
                    "2_score": None,
                    "4_score": None,
                })
    qdf = pd.DataFrame(q_rows)
    rdf = pd.DataFrame(r_rows)
    save_questions_jsonl(QUESTIONS_JSONL, qdf)
    save_replies_jsonl(REPLIES_JSONL, rdf)
    print(f"✓ {QUESTIONS_JSONL.name}: {len(qdf)} questions")
    print(f"✓ {REPLIES_JSONL.name}: {len(rdf)} reply rows")


if __name__ == "__main__":
    nested_to_flat()
