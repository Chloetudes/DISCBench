#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export DISCBench open-source JSONL (PK cohort — first 200 Ours by qid).

Default: all 200 PK items with available 12-model replies + GPT-5.4 scores.
Optional: --ranked-subset 100 — legacy high-difficulty / stable-judge subset.

Output: data/DISCbench_data.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
for p in (str(_ROOT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.cif_stats_common import (  # noqa: E402
    CANONICAL_12,
    OURS_SOURCE,
    PK_OURS_N,
    load_questions,
    load_replies_with_scores,
    ours_pk_qids,
)
from lib.open_jsonl_export import (  # noqa: E402
    DISCBENCH_MODELS,
    build_nested_record,
    export_nested_jsonl,
    prepare_replies_slice,
)
from lib.paths import CIF_ROOT, DISCBENCH_OPEN_JSONL, resolve_questions_path, resolve_replies_path  # noqa: E402

STABILITY_WEIGHT = 0.5
REQUIRED_MODELS = len(CANONICAL_12)


def _sort_qids(qids: set[str]) -> List[str]:
    return sorted(qids, key=lambda x: float(x) if str(x).replace(".", "").isdigit() else x)


def rank_pk_by_stability(questions: pd.DataFrame, replies: pd.DataFrame) -> pd.DataFrame:
    """Rank PK pool for optional 100-item subset (high difficulty + stable judges)."""
    pk = _sort_qids(ours_pk_qids(questions))
    qsub = questions[questions["qid"].astype(str).isin(pk)].set_index("qid")
    r = prepare_replies_slice(replies, set(pk), models=DISCBENCH_MODELS)

    rows: List[Dict[str, Any]] = []
    for qid in pk:
        if qid not in qsub.index:
            continue
        g = r[r["qid"].astype(str) == str(qid)]
        s1 = pd.to_numeric(g["1_score"], errors="coerce")
        s3 = pd.to_numeric(g["3_score"], errors="coerce")
        mask = s1.notna() & s3.notna()
        if int(mask.sum()) < REQUIRED_MODELS:
            continue
        qrow = qsub.loc[qid]
        if isinstance(qrow, pd.DataFrame):
            qrow = qrow.iloc[0]
        diff = pd.to_numeric(qrow.get("difficulty_score"), errors="coerce")
        if pd.isna(diff):
            continue
        diff_abs = (s1[mask] - s3[mask]).abs()
        rows.append({
            "qid": qid,
            "difficulty_score": float(diff),
            "mean_abs_round13_diff": float(diff_abs.mean()),
            "priority_score": float(diff) - STABILITY_WEIGHT * float(diff_abs.mean()),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["priority_score", "difficulty_score", "mean_abs_round13_diff", "qid"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def export_discbench(
    *,
    out_path: Path = DISCBENCH_OPEN_JSONL,
    questions_path: Path | None = None,
    replies_path: Path | None = None,
    ranked_subset: int | None = None,
) -> Dict[str, Any]:
    questions = load_questions(questions_path or resolve_questions_path())
    replies = load_replies_with_scores(replies_path or resolve_replies_path(), questions)

    pk = _sort_qids(ours_pk_qids(questions))
    if ranked_subset is not None:
        ranked = rank_pk_by_stability(questions, replies)
        if len(ranked) < ranked_subset:
            raise RuntimeError(
                f"Only {len(ranked)} items meet 12-model score criteria; need {ranked_subset}."
            )
        qid_order = ranked.head(ranked_subset)["qid"].astype(str).tolist()
        pool_label = f"PK_{PK_OURS_N}_ranked_top_{ranked_subset}"
    else:
        qid_order = pk
        pool_label = f"PK_{PK_OURS_N}_full"

    q_ours = questions[
        (questions["source"].astype(str) == OURS_SOURCE)
        & (questions["qid"].astype(str).isin(qid_order))
    ]

    summary = export_nested_jsonl(
        questions=q_ours,
        replies=replies,
        qid_order=qid_order,
        models=DISCBENCH_MODELS,
        out_path=out_path,
    )
    summary["pool"] = pool_label
    summary["qid_min"] = qid_order[0] if qid_order else None
    summary["qid_max"] = qid_order[-1] if qid_order else None
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ranked-subset",
        type=int,
        default=None,
        metavar="N",
        help="Export top-N by difficulty/stability (legacy 100-item release)",
    )
    ap.add_argument("--out", type=Path, default=DISCBENCH_OPEN_JSONL)
    ap.add_argument("--questions", type=Path, default=None)
    ap.add_argument("--replies", type=Path, default=None)
    args = ap.parse_args()

    summary = export_discbench(
        out_path=args.out,
        questions_path=args.questions,
        replies_path=args.replies,
        ranked_subset=args.ranked_subset,
    )
    print(f"✓ Wrote {summary['n_written']} items → {summary['output']}")
    print(f"  pool: {summary['pool']}")
    print(f"  models: {len(summary['models'])} (incomplete rows: {summary['n_incomplete_model_coverage']})")
    if summary.get("qid_min"):
        print(f"  qid range: {summary['qid_min']} – {summary['qid_max']}")


if __name__ == "__main__":
    main()
