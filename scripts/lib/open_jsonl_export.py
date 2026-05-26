#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build nested open-release JSONL records (DISCbench / public benchmark schema)."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from lib.cif_stats_common import (
    CANONICAL_12,
    CANONICAL_8,
    GPT54_SCORE_COLS,
    PRIMARY_SCORE_LABEL,
    gpt54_mean_score,
    normalize_qid,
    safe_str,
    to_discbench_model,
)

TEXT_KEYS = ("query", "rubrics")


def parse_rubrics_items(rubrics: Any) -> List[str]:
    text = safe_str(rubrics).strip()
    if not text:
        return []
    numbered = re.findall(r"(?m)^\s*(\d+)\.\s+(.+)$", text)
    if numbered:
        return [t.strip() for _, t in numbered]
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def parse_quality_json(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = safe_str(raw).strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def tier_label(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val)


def source_display_name(source: str) -> str:
    """Nested JSONL `source` field for open bundles."""
    if source == "Ours":
        return "DISCBench"
    return str(source)


def build_model_responses(
    g: pd.DataFrame,
    *,
    models: Sequence[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if "model_id" not in g.columns:
        g = g.copy()
        g["model_id"] = g["model"].map(to_discbench_model)
    for model in models:
        sub = g[g["model_id"] == model]
        if sub.empty:
            continue
        row = sub.iloc[0]
        s1 = pd.to_numeric(row.get("1_score"), errors="coerce")
        s3 = pd.to_numeric(row.get("3_score"), errors="coerce")
        mean_s = gpt54_mean_score(row)
        entry: Dict[str, Any] = {
            "model": model,
            "reply": safe_str(row.get("reply")),
            "score_round1": None if pd.isna(s1) else float(s1),
            "score_round3": None if pd.isna(s3) else float(s3),
            "score_mean": None if pd.isna(mean_s) else round(float(mean_s), 4),
        }
        out.append(entry)
    return out


def build_nested_record(
    qid: str,
    qrow: pd.Series,
    g: pd.DataFrame,
    *,
    models: Sequence[str],
    cohort_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    quality = parse_quality_json(qrow.get("instruction_quality_parsed_json"))
    rubrics_text = safe_str(qrow.get("rubrics"))
    cp_n = pd.to_numeric(qrow.get("checkpoint_n"), errors="coerce")
    if pd.isna(cp_n):
        cp_n = pd.to_numeric(qrow.get("cp_classify_n_items"), errors="coerce")

    src = safe_str(qrow.get("source"))
    diff = pd.to_numeric(qrow.get("difficulty_score"), errors="coerce")

    record: Dict[str, Any] = {
        "id": normalize_qid(qid),
        "source": source_display_name(src),
        "taxonomy": {
            "L1": safe_str(qrow.get("L1")),
            "L2": safe_str(qrow.get("L2")),
            "L3": safe_str(qrow.get("L3")),
        },
        "instruction": {
            "query": safe_str(qrow.get("query")),
            "rubrics": rubrics_text,
        },
        "checkpoints": {
            "count": int(cp_n) if pd.notna(cp_n) and cp_n > 0 else len(parse_rubrics_items(rubrics_text)),
            "items": parse_rubrics_items(rubrics_text),
        },
        "instruction_evaluation": {
            "difficulty_score": None if pd.isna(diff) else float(diff),
            "difficulty_tier": tier_label(qrow.get("difficulty_tier")),
            "difficulty_level": safe_str(qrow.get("difficulty_level")),
            "difficulty_desc": safe_str(qrow.get("difficulty_desc")),
            "score_rule": PRIMARY_SCORE_LABEL,
            "judge_rounds": list(GPT54_SCORE_COLS),
        },
        "model_responses": build_model_responses(g, models=models),
    }
    if quality is not None:
        record["instruction_evaluation"]["quality_analysis"] = quality
    if cohort_meta:
        record["cohort"] = cohort_meta
    return record


def prepare_replies_slice(
    replies: pd.DataFrame,
    qids: set[str],
    *,
    models: Sequence[str],
) -> pd.DataFrame:
    r = replies[replies["qid"].astype(str).isin({str(x) for x in qids})].copy()
    r["model_id"] = r["model"].map(to_discbench_model)
    return r[r["model_id"].isin(models)]


def export_nested_jsonl(
    *,
    questions: pd.DataFrame,
    replies: pd.DataFrame,
    qid_order: List[str],
    models: Sequence[str],
    out_path: Any,
) -> Dict[str, Any]:
    qsub = questions[questions["qid"].astype(str).isin({str(x) for x in qid_order})].copy()
    qsub = qsub.set_index("qid")
    r = prepare_replies_slice(replies, set(qid_order), models=models)

    records: List[Dict[str, Any]] = []
    n_missing_any = 0
    for idx, qid in enumerate(qid_order, start=1):
        qid = str(qid)
        if qid not in qsub.index:
            continue
        qrow = qsub.loc[qid]
        if isinstance(qrow, pd.DataFrame):
            qrow = qrow.iloc[0]
        g = r[r["qid"].astype(str) == qid]
        n_have = g["model_id"].nunique()
        if n_have < len(models):
            n_missing_any += 1
        meta = {
            "pool_index": idx,
            "n_models": int(n_have),
            "expected_models": len(models),
        }
        records.append(
            build_nested_record(qid, qrow, g, models=models, cohort_meta=meta)
        )

    out_path = __import__("pathlib").Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "output": str(out_path),
        "n_written": len(records),
        "n_incomplete_model_coverage": n_missing_any,
        "models": list(models),
    }


DISCBENCH_MODELS = CANONICAL_12
PUBLIC_MODELS = CANONICAL_8
