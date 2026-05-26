#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export open-source JSONL for the four public comparison benchmarks (4 × 200 items).

Each line: nested schema (instruction, checkpoints, instruction_evaluation, model_responses).
Models: CANONICAL_8 (shared cross-benchmark comparison set).
Scores: GPT-5.4 mean(1_score, 3_score).

Output: data/public_benchmark_data.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
for p in (str(_ROOT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.cif_stats_common import (  # noqa: E402
    PUBLIC_ANALYSIS_N,
    PUBLIC_SOURCES,
    load_questions,
    load_replies_with_scores,
)
from lib.open_jsonl_export import PUBLIC_MODELS, export_nested_jsonl  # noqa: E402
from lib.paths import CIF_ROOT, resolve_questions_path, resolve_replies_path  # noqa: E402

DEFAULT_OUT = CIF_ROOT / "data" / "public_benchmark_data.jsonl"


def _sort_qids(qids: List[str]) -> List[str]:
    return sorted(qids, key=lambda x: float(x) if str(x).replace(".", "").isdigit() else x)


def public_cohort_qid_order(questions) -> List[str]:
    order: List[str] = []
    for src in PUBLIC_SOURCES:
        sub = questions[questions["source"].astype(str) == src]
        qids = _sort_qids(sub["qid"].astype(str).unique().tolist())
        if len(qids) != PUBLIC_ANALYSIS_N:
            raise RuntimeError(
                f"{src}: expected {PUBLIC_ANALYSIS_N} questions, got {len(qids)}"
            )
        order.extend(qids)
    return order


def export_public(
    *,
    out_path: Path = DEFAULT_OUT,
    questions_path: Path | None = None,
    replies_path: Path | None = None,
) -> Dict:
    questions = load_questions(questions_path or resolve_questions_path())
    replies = load_replies_with_scores(replies_path or resolve_replies_path(), questions)
    qid_order = public_cohort_qid_order(questions)
    q_pub = questions[questions["source"].astype(str).isin(PUBLIC_SOURCES)].copy()

    summary = export_nested_jsonl(
        questions=q_pub,
        replies=replies,
        qid_order=qid_order,
        models=PUBLIC_MODELS,
        out_path=out_path,
    )
    summary["pool"] = f"public_{len(PUBLIC_SOURCES)}x{PUBLIC_ANALYSIS_N}"
    summary["sources"] = list(PUBLIC_SOURCES)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--questions", type=Path, default=None)
    ap.add_argument("--replies", type=Path, default=None)
    args = ap.parse_args()

    summary = export_public(
        out_path=args.out,
        questions_path=args.questions,
        replies_path=args.replies,
    )
    print(f"✓ Wrote {summary['n_written']} items → {summary['output']}")
    print(f"  pool: {summary['pool']} ({', '.join(summary['sources'])})")
    print(f"  models: {len(summary['models'])} (incomplete rows: {summary['n_incomplete_model_coverage']})")


if __name__ == "__main__":
    main()
