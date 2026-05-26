#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot: legacy Excel → canonical JSONL + single master workbook (12 models only).

Creates:
  data/questions.jsonl
  data/replies.jsonl
  data/discbench_master.xlsx   (Questions + Replies sheets)

Optional --prune: remove legacy split workbooks and output/questions|replies mirrors.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
for p in (str(_ROOT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.cif_stats_common import OURS_SOURCE, to_discbench_model, CANONICAL_12  # noqa: E402
from lib.jsonl_store import (  # noqa: E402
    load_questions_df,
    load_replies_df,
    questions_df_to_records,
    replies_df_to_records,
    records_to_questions_df,
    records_to_replies_df,
    save_master_xlsx,
    save_questions_jsonl,
    save_replies_jsonl,
)
from lib.paths import (  # noqa: E402
    CIF_ROOT,
    LEGACY_QUESTIONS_XLSX,
    LEGACY_REPLIES_XLSX,
    MASTER_XLSX,
    QUESTIONS_JSONL,
    REPLIES_JSONL,
)


def _load_legacy_questions(path: Path, *, ours_only: bool) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing questions: {path}")
    df = pd.read_excel(path, sheet_name="数据对齐")
    if ours_only:
        df = df[df["source"].astype(str) == OURS_SOURCE].copy()
    return records_to_questions_df(questions_df_to_records(df))


def _load_legacy_replies(path: Path, qids: set) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing replies: {path}")
    df = pd.read_excel(path)
    df = df[df["qid"].astype(str).isin({str(x) for x in qids})]
    df = df[~df["model"].astype(str).str.lower().eq("ref")]
    df = df[~df["model"].astype(str).str.startswith("reply_slot_")]
    df["model"] = df["model"].map(lambda m: to_discbench_model(str(m)) or str(m).strip())
    df = df[df["model"].isin(CANONICAL_12)]
    return records_to_replies_df(replies_df_to_records(df, canonical_only=False))


def consolidate(*, prune: bool = False) -> None:
    q_all = _load_legacy_questions(LEGACY_QUESTIONS_XLSX, ours_only=False)
    q_ours = q_all[q_all["source"].astype(str) == OURS_SOURCE].copy()
    qids = set(q_all["qid"].astype(str))
    r = _load_legacy_replies(LEGACY_REPLIES_XLSX, qids)

    save_questions_jsonl(QUESTIONS_JSONL, q_all)
    save_replies_jsonl(REPLIES_JSONL, r)
    save_master_xlsx(MASTER_XLSX, q_ours, r[r["qid"].astype(str).isin(set(q_ours["qid"].astype(str)))])

    print(f"✓ {QUESTIONS_JSONL.relative_to(CIF_ROOT)}  ({len(q_all)} questions, all sources)")
    print(f"✓ {REPLIES_JSONL.relative_to(CIF_ROOT)}  ({len(r)} reply rows, 12 models)")
    print(f"✓ {MASTER_XLSX.relative_to(CIF_ROOT)}  ({len(q_ours)} Ours + replies)")

    if prune:
        for p in [
            LEGACY_QUESTIONS_XLSX,
            LEGACY_REPLIES_XLSX,
            CIF_ROOT / "output/questions",
            CIF_ROOT / "output/replies",
        ]:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                print(f"  removed dir {p.relative_to(CIF_ROOT)}")
            elif p.is_file():
                p.unlink(missing_ok=True)
                print(f"  removed file {p.relative_to(CIF_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prune", action="store_true", help="Delete legacy xlsx and output mirrors")
    ap.add_argument(
        "--from-jsonl",
        action="store_true",
        help="Rebuild master xlsx from existing questions.jsonl + replies.jsonl only",
    )
    args = ap.parse_args()
    if args.from_jsonl:
        q = load_questions_df(QUESTIONS_JSONL)
        r = load_replies_df(REPLIES_JSONL)
        save_master_xlsx(MASTER_XLSX, q, r)
        print(f"✓ rebuilt {MASTER_XLSX}")
        return
    consolidate(prune=args.prune)


if __name__ == "__main__":
    main()
