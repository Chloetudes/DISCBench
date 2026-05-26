#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical JSONL I/O for DISCBench (questions + replies)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from lib.cif_stats_common import (
    CANONICAL_12,
    OURS_SOURCE,
    assign_tier_bin,
    checkpoint_count_for_row,
    normalize_qid,
    pick_difficulty,
    query_length,
    safe_str,
    to_discbench_model,
)

PathLike = Union[str, Path]

# --- Questions: columns kept in master store ---
QUESTION_COLUMNS = [
    "qid",
    "source",
    "query",
    "rubrics",
    "reference",
    "L1",
    "L2",
    "L3",
    "difficulty_score",
    "difficulty_tier",
    "difficulty_level",
    "difficulty_desc",
    "instruction_quality_raw",
    "instruction_quality_parsed_json",
    "checkpoint_n",
    "constraint_n",
    "query_len",
    "iq_count_教学约束",
    "iq_count_素材约束",
    "iq_count_流程步骤",
    "iq_count_格式输出",
    "iq_count_边界范围",
    "iq_count_数量篇幅",
]

# --- Replies: 12 models only ---
REPLY_COLUMNS = [
    "qid",
    "model",
    "reply",
    "provider",
    "status",
    "1_score",
    "1_details",
    "2_score",
    "2_details",
    "3_score",
    "3_details",
    "source",
]

SCORE_COLS = ["1_score", "2_score", "3_score", "4_score"]
DETAIL_COLS = ["1_details", "2_details", "3_details", "4_details"]


def _path(p: PathLike) -> Path:
    return Path(p)


def _json_safe(val: Any) -> Any:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (np.integer, np.floating)):
        return float(val) if isinstance(val, np.floating) else int(val)
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    return val


def _row_to_json(obj: Dict[str, Any]) -> str:
    clean = {k: _json_safe(v) for k, v in obj.items() if _json_safe(v) is not None or v is None}
    return json.dumps(clean, ensure_ascii=False)


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    p = _path(path)
    if not p.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: PathLike, rows: Iterable[Dict[str, Any]]) -> None:
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(_row_to_json(row) + "\n")


def enrich_question_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Fill derived fields (difficulty tier, checkpoint_n, query_len)."""
    s = pd.Series(row)
    picked = pick_difficulty(s)
    row["difficulty_score"] = picked[0] if pd.notna(picked[0]) else row.get("difficulty_score")
    if row.get("query_len") is None or pd.isna(row.get("query_len")):
        row["query_len"] = query_length(row.get("query"))
    if row.get("checkpoint_n") is None or pd.isna(row.get("checkpoint_n")):
        row["checkpoint_n"] = checkpoint_count_for_row(s)
    tier = assign_tier_bin(pd.Series([row.get("difficulty_score")]))
    row["difficulty_tier"] = str(tier[0]) if len(tier) else row.get("difficulty_tier", "")
    return row


def questions_df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: Dict[str, Any] = {}
        for col in QUESTION_COLUMNS:
            if col in row.index:
                v = row[col]
                if pd.notna(v) if not isinstance(v, str) else bool(str(v).strip()):
                    rec[col] = v if not (isinstance(v, float) and pd.isna(v)) else None
        rec["qid"] = normalize_qid(rec.get("qid", row.get("qid")))
        rec["source"] = safe_str(rec.get("source", OURS_SOURCE)) or OURS_SOURCE
        records.append(enrich_question_row(rec))
    return records


def replies_df_to_records(df: pd.DataFrame, *, canonical_only: bool = True) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        model = to_discbench_model(str(row.get("model", "")))
        if canonical_only and model not in CANONICAL_12:
            continue
        if not model:
            continue
        rec: Dict[str, Any] = {"qid": normalize_qid(row["qid"]), "model": model}
        for col in REPLY_COLUMNS:
            if col in ("qid", "model"):
                continue
            if col in row.index:
                v = row[col]
                if pd.notna(v) if not isinstance(v, str) else bool(str(v).strip()):
                    rec[col] = v
        rec.setdefault("source", OURS_SOURCE)
        records.append(rec)
    return records


def records_to_questions_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=QUESTION_COLUMNS)
    df = pd.DataFrame(records)
    for col in QUESTION_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["qid"] = df["qid"].map(normalize_qid)
    picked = df.apply(pick_difficulty, axis=1, result_type="expand")
    df["difficulty_score"] = picked[0]
    df["difficulty_tier"] = assign_tier_bin(df["difficulty_score"])
    return df[QUESTION_COLUMNS]


def records_to_replies_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=REPLY_COLUMNS)
    df = pd.DataFrame(records)
    for col in REPLY_COLUMNS + DETAIL_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df["qid"] = df["qid"].map(normalize_qid)
    df["model"] = df["model"].astype(str).str.strip()
    return df


def load_questions_df(path: PathLike, *, source: Optional[str] = OURS_SOURCE) -> pd.DataFrame:
    p = _path(path)
    if p.suffix.lower() == ".jsonl":
        records = read_jsonl(p)
        if source:
            records = [r for r in records if safe_str(r.get("source")) == source]
        return records_to_questions_df(records)
    sheet = "Questions" if p.suffix.lower() == ".xlsx" else "数据对齐"
    try:
        df = pd.read_excel(p, sheet_name=sheet)
    except ValueError:
        df = pd.read_excel(p, sheet_name="数据对齐")
    if source and "source" in df.columns:
        df = df[df["source"].astype(str) == source]
    return records_to_questions_df(questions_df_to_records(df))


def load_replies_df(
    path: PathLike,
    *,
    canonical_only: bool = True,
    qids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    p = _path(path)
    if p.suffix.lower() == ".jsonl":
        records = read_jsonl(p)
    elif p.suffix.lower() == ".xlsx":
        try:
            df = pd.read_excel(p, sheet_name="Replies")
        except ValueError:
            df = pd.read_excel(p)
        records = replies_df_to_records(df, canonical_only=False)
    else:
        raise ValueError(f"Unsupported replies format: {p}")
    if canonical_only:
        records = [r for r in records if r.get("model") in CANONICAL_12]
    if qids is not None:
        qset = {normalize_qid(x) for x in qids}
        records = [r for r in records if normalize_qid(r.get("qid")) in qset]
    return records_to_replies_df(records)


def save_questions_jsonl(path: PathLike, df: pd.DataFrame) -> None:
    write_jsonl(path, questions_df_to_records(df))


def save_replies_jsonl(path: PathLike, df: pd.DataFrame) -> None:
    write_jsonl(path, replies_df_to_records(df, canonical_only=False))


def save_master_xlsx(
    path: PathLike,
    questions: pd.DataFrame,
    replies: pd.DataFrame,
) -> None:
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    q_out = records_to_questions_df(questions_df_to_records(questions))
    r_out = records_to_replies_df(replies_df_to_records(replies, canonical_only=False))
    with pd.ExcelWriter(p, engine="openpyxl") as w:
        q_out.to_excel(w, sheet_name="Questions", index=False)
        r_out.to_excel(w, sheet_name="Replies", index=False)


def load_master_xlsx(path: PathLike) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = _path(path)
    q = pd.read_excel(p, sheet_name="Questions")
    r = pd.read_excel(p, sheet_name="Replies")
    return records_to_questions_df(questions_df_to_records(q)), records_to_replies_df(
        replies_df_to_records(r, canonical_only=False)
    )


def merge_nested_export_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert flat staging rows back to DISCbench_data.jsonl-style record (optional)."""
    return record  # export script owns nested format
