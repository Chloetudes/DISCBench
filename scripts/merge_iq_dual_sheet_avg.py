#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双轮指令质量合并：sheet1（首轮）+ sheet2（第二轮）→ sheet1 写回
  - difficulty_score_r1 / difficulty_score_r2
  - difficulty_score = mean(r1, r2)（后续统计用）
  - difficulty_score_avg 同义备份
"""
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


def merge_one(path: Path, sheet1: str = SHEET1, sheet2: str = SHEET2) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    if sheet2 not in xl.sheet_names:
        raise ValueError(f"{path} 缺少 {sheet2!r}，请先跑 Stage1 Round2")

    sheets = {name: pd.read_excel(path, sheet_name=name) for name in xl.sheet_names}
    s1 = sheets[sheet1].copy()
    s2 = sheets[sheet2].copy()
    s1["qid"] = s1["qid"].astype(str).str.strip()
    s2["qid"] = s2["qid"].astype(str).str.strip()

    r2_map = s2.set_index("qid")["difficulty_score"].to_dict()
    r1 = pd.to_numeric(
        s1.get("difficulty_score_r1", s1.get("difficulty_score")), errors="coerce"
    )
    r2 = s1["qid"].map(lambda q: pd.to_numeric(r2_map.get(q), errors="coerce"))

    s1["difficulty_score_r1"] = r1
    s1["difficulty_score_r2"] = r2
    both = r1.notna() & r2.notna()
    avg = pd.Series(np.nan, index=s1.index, dtype=float)
    avg.loc[both] = ((r1.loc[both] + r2.loc[both]) / 2.0).round(2)
    avg.loc[~both & r1.notna()] = r1.loc[~both & r1.notna()]
    avg.loc[~both & r2.notna()] = r2.loc[~both & r2.notna()]

    s1["difficulty_score_avg"] = avg
    s1["difficulty_score"] = avg
    if "difficulty_score_py" in s1.columns:
        s1["difficulty_score_py"] = avg

    sheets[sheet1] = s1
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, sdf in sheets.items():
            sdf.to_excel(w, sheet_name=name, index=False)

    cov = pd.DataFrame([{
        "文件": str(path),
        "题数": len(s1),
        "r1有效": int(r1.notna().sum()),
        "r2有效": int(r2.notna().sum()),
        "双轮均分有效": int(both.sum()),
        "平均难度(均分后)": round(float(avg.mean()), 2),
    }])
    print(cov.to_string(index=False))
    return s1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(QUESTIONS_XLSX))
    ap.add_argument("--sync-all", action="store_true")
    args = ap.parse_args()

    paths = [Path(p) for p in QUESTION_TABLE_PATHS] if args.sync_all else [Path(args.questions)]
    for p in paths:
        if p.is_file():
            merge_one(p)


if __name__ == "__main__":
    main()
