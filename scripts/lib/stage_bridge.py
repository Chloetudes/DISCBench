#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sync canonical JSONL ↔ staging Excel for legacy stage modules."""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
_ROOT = _SCRIPTS.parent
for p in (str(_ROOT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.jsonl_store import (  # noqa: E402
    load_questions_df,
    load_replies_df,
    records_to_questions_df,
    records_to_replies_df,
    replies_df_to_records,
    save_questions_jsonl,
    save_replies_jsonl,
    questions_df_to_records,
)
from lib.paths import (  # noqa: E402
    QUESTIONS_JSONL,
    REPLIES_JSONL,
    STAGING_QUESTIONS_XLSX,
    STAGING_REPLIES_XLSX,
)


def push_jsonl_to_staging() -> None:
    """JSONL → staging xlsx (before running evaluation stages)."""
    q = load_questions_df(QUESTIONS_JSONL)
    r = load_replies_df(REPLIES_JSONL)
    STAGING_QUESTIONS_XLSX.parent.mkdir(parents=True, exist_ok=True)
    q.to_excel(STAGING_QUESTIONS_XLSX, sheet_name="数据对齐", index=False)
    r.to_excel(STAGING_REPLIES_XLSX, sheet_name="Replies", index=False)
    print(f"  ✓ staging questions → {STAGING_QUESTIONS_XLSX.relative_to(_ROOT)} ({len(q)} rows)")
    print(f"  ✓ staging replies   → {STAGING_REPLIES_XLSX.relative_to(_ROOT)} ({len(r)} rows)")


def pull_staging_to_jsonl() -> None:
    """staging xlsx → JSONL (after evaluation stages)."""
    import pandas as pd

    if STAGING_QUESTIONS_XLSX.is_file():
        q_raw = pd.read_excel(STAGING_QUESTIONS_XLSX, sheet_name=0)
        save_questions_jsonl(QUESTIONS_JSONL, records_to_questions_df(questions_df_to_records(q_raw)))
        print(f"  ✓ questions jsonl ← staging ({len(q_raw)} rows)")
    if STAGING_REPLIES_XLSX.is_file():
        r_raw = pd.read_excel(STAGING_REPLIES_XLSX, sheet_name=0)
        save_replies_jsonl(
            REPLIES_JSONL,
            records_to_replies_df(replies_df_to_records(r_raw, canonical_only=False)),
        )
        print(f"  ✓ replies jsonl   ← staging ({len(r_raw)} rows)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Sync JSONL ↔ staging Excel for stages")
    ap.add_argument("action", choices=["push", "pull"], help="push=jsonl→xlsx, pull=xlsx→jsonl")
    args = ap.parse_args()
    if args.action == "push":
        push_jsonl_to_staging()
    else:
        pull_staging_to_jsonl()
