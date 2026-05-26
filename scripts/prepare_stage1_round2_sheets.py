#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage1 第二轮前：sheet1 保留首轮难度为 difficulty_score_r1，创建空 IQ 的 sheet2。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
for p in (str(_ROOT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.paths import QUESTION_TABLE_PATHS, QUESTIONS_XLSX

SHEET1 = "数据对齐"
SHEET2 = "数据对齐_2"

IQ_RESULT_COLS = [
    "instruction_quality_raw",
    "difficulty_score",
    "difficulty_score_py",
    "difficulty_score_model",
    "difficulty_score_delta",
    "difficulty_level",
    "difficulty_desc",
    "iq_numerator",
    "iq_denominator",
    "iq_constraint_json",
    "instruction_quality_parsed_json",
    "iq_qualified",
    "iq_filter_reason",
    "iq_parse_ok",
    "iq_status",
    "iq_error",
    "iq_count_教学约束",
    "iq_count_素材约束",
    "iq_count_流程步骤",
    "iq_count_格式输出",
    "iq_count_边界范围",
    "iq_count_数量篇幅",
]

QUESTION_TABLE_COPIES = list(QUESTION_TABLE_PATHS)


def prepare_one(path: Path, sheet1: str = SHEET1, sheet2: str = SHEET2) -> None:
    if not path.is_file():
        print(f"⚠ 跳过不存在: {path}")
        return

    xl = pd.ExcelFile(path)
    sheets = {name: pd.read_excel(path, sheet_name=name) for name in xl.sheet_names}
    if sheet1 not in sheets:
        raise ValueError(f"{path} 缺少 sheet {sheet1!r}")

    s1 = sheets[sheet1].copy()
    if "difficulty_score_r1" not in s1.columns and "difficulty_score" in s1.columns:
        s1["difficulty_score_r1"] = pd.to_numeric(s1["difficulty_score"], errors="coerce")
        print(f"  {path.name}: 已快照 difficulty_score → difficulty_score_r1")

    s2 = s1.copy()
    for col in IQ_RESULT_COLS:
        if col in s2.columns:
            s2[col] = np.nan
    if "difficulty_score_r2" in s2.columns:
        s2 = s2.drop(columns=["difficulty_score_r2"])
    if "difficulty_score_avg" in s2.columns:
        s2 = s2.drop(columns=["difficulty_score_avg"])

    sheets[sheet1] = s1
    sheets[sheet2] = s2

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, sdf in sheets.items():
            sdf.to_excel(w, sheet_name=name, index=False)

    print(f"✓ {path}  [{sheet1}] {len(s1)} 行 | [{sheet2}] 已就绪")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(QUESTIONS_XLSX))
    ap.add_argument("--sync-all", action="store_true")
    args = ap.parse_args()

    paths = [Path(p) for p in QUESTION_TABLE_COPIES] if args.sync_all else [Path(args.questions)]
    for p in paths:
        prepare_one(p)


if __name__ == "__main__":
    main()
