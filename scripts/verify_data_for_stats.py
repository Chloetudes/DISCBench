#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计前数据就绪检查：题目表补字段、回复 GPT-5.4 分列对齐、cohort 覆盖。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
for p in (str(_ROOT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.cif_stats_common import (
    CANONICAL_8,
    GPT54_JUDGE_MODEL,
    GPT54_SCORE_COLS,
    MIN_MODELS,
    OURS_SOURCE,
    PK_OURS_N,
    PRIMARY_SCORE_LABEL,
    PUBLIC_ANALYSIS_N,
    PUBLIC_SOURCES,
    SOURCE_ORDER,
    analysis_cohort_qids,
    load_questions,
    load_replies_with_scores,
    verify_cohort_completeness,
    resolve_questions_path,
    resolve_replies_path,
    QUESTIONS_SHEET,
)

IQ_COUNT_COLS = [
    "iq_count_教学约束", "iq_count_素材约束", "iq_count_流程步骤",
    "iq_count_格式输出", "iq_count_边界范围", "iq_count_数量篇幅",
]

JUDGE_BY_SCORE_COL = {
    "1_score": GPT54_JUDGE_MODEL,
    "2_score": "Claude Opus 4.6",
    "3_score": GPT54_JUDGE_MODEL,
    "4_score": GPT54_JUDGE_MODEL,
}


def check_questions(q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src in SOURCE_ORDER:
        sub = q[q["source"].astype(str) == src]
        rows.append({
            "数据来源": src,
            "题数": len(sub),
            "query_len": int(sub["query_len"].notna().sum()) if "query_len" in sub.columns else 0,
            "constraint_n": int(sub["constraint_n"].notna().sum()) if "constraint_n" in sub.columns else 0,
            "checkpoint_n": int(sub["checkpoint_n"].notna().sum()) if "checkpoint_n" in sub.columns else 0,
            "L1": int(sub["L1"].notna().sum()) if "L1" in sub.columns else 0,
            "difficulty_score": int(sub["difficulty_score"].notna().sum()),
            "instruction_quality_raw": int(sub["instruction_quality_raw"].notna().sum()) if "instruction_quality_raw" in sub.columns else 0,
            "iq_count": int(sub[IQ_COUNT_COLS[0]].notna().sum()) if IQ_COUNT_COLS[0] in sub.columns else 0,
            "缺失难度": int(sub["difficulty_score"].isna().sum()),
            "缺失L1": int(sub["L1"].isna().sum()) if "L1" in sub.columns else len(sub),
            "缺失考点": int(sub["checkpoint_n"].isna().sum()) if "checkpoint_n" in sub.columns else len(sub),
        })
    return pd.DataFrame(rows)


def check_replies_alignment(replies_path: Path, q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = analysis_cohort_qids(q)
    scored = load_replies_with_scores(replies_path, q)
    scored = scored[scored["qid"].isin(cohort)]

    from lib.jsonl_store import load_replies_df

    if replies_path.suffix.lower() == ".jsonl":
        r = load_replies_df(replies_path, canonical_only=False)
    else:
        r = pd.read_excel(replies_path)
    r["qid"] = r["qid"].astype(str)
    src_map = dict(zip(q["qid"].astype(str), q["source"].astype(str)))
    r["source"] = r["qid"].map(src_map)

    score_cols = ["1_score", "2_score", "3_score", "4_score"]
    for c in score_cols:
        if c in r.columns:
            r[c] = pd.to_numeric(r[c], errors="coerce")

    rows = []
    for src in SOURCE_ORDER:
        sub = scored[scored["source"] == src]
        n_q = sub["qid"].nunique()
        n_ge4 = 0
        for qid, g in sub.groupby("qid"):
            g = g[g["logical_model"].isin(CANONICAL_8)]
            g = g.dropna(subset=["score"])
            if len(g.groupby("model")["score"].first()) >= MIN_MODELS:
                n_ge4 += 1
        row = {
            "数据来源": src,
            "cohort题数": PUBLIC_ANALYSIS_N if src != OURS_SOURCE else PK_OURS_N,
            "有回复题数": n_q,
            "主分列": PRIMARY_SCORE_LABEL,
            "主分裁判": GPT54_JUDGE_MODEL,
            f"≥{MIN_MODELS}模型有主分": n_ge4,
            "主分完整率_%": round(n_ge4 / (PUBLIC_ANALYSIS_N if src != OURS_SOURCE else PK_OURS_N) * 100, 1),
        }
        rsub = r[r["source"] == src]
        for c in score_cols:
            if c in rsub.columns:
                row[c] = int(rsub[c].notna().sum())
        rows.append(row)

    round_rows = []
    for label, col in [
        ("第1轮", "1_score"), ("第2轮_Claude", "2_score"),
        ("第3轮", "3_score"), ("第4轮", "4_score"),
    ]:
        if col not in r.columns:
            continue
        round_rows.append({
            "轮次": label,
            "分列": col,
            "裁判": JUDGE_BY_SCORE_COL.get(col, "?"),
            "有效条数": int(r[col].notna().sum()),
            "GPT54": col in ("1_score", "3_score", "4_score"),
        })

    return pd.DataFrame(rows), pd.DataFrame(round_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(resolve_questions_path()))
    ap.add_argument("--replies", default=str(resolve_replies_path()))
    ap.add_argument(
        "--output",
        default=str(_ROOT / "output/reports/data_ready_for_stats.xlsx"),
        help="Optional Excel audit workbook (skipped if --no-excel).",
    )
    ap.add_argument(
        "--no-excel",
        action="store_true",
        help="Only print readiness tables (no auxiliary xlsx under output/).",
    )
    args = ap.parse_args()

    q = load_questions(Path(args.questions), sheet="数据对齐")
    scored = load_replies_with_scores(Path(args.replies), q)
    coverage = verify_cohort_completeness(q, scored)
    q_check = check_questions(q)
    r_align, rounds = check_replies_alignment(Path(args.replies), q)

    meta = pd.DataFrame([
        {"项": "题目表", "值": args.questions},
        {"项": "回复表", "值": args.replies},
        {"项": "统一裁判", "值": GPT54_JUDGE_MODEL},
        {"项": "统一主分", "值": f"{PRIMARY_SCORE_LABEL}（GPT-5.4 第1轮+第3轮均分）"},
        {"项": "主分原始列", "值": " + ".join(GPT54_SCORE_COLS)},
        {"项": "分析cohort", "值": f"公开4×{PUBLIC_ANALYSIS_N} + Ours前{PK_OURS_N} = 1000"},
        {"项": "2_score说明", "值": "Claude 裁判，仅用于一致性对比，不作主统计"},
    ])

    if not args.no_excel:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            meta.to_excel(w, sheet_name="00_说明", index=False)
            q_check.to_excel(w, sheet_name="01_题目表覆盖", index=False)
            coverage.to_excel(w, sheet_name="02_cohort回复覆盖", index=False)
            r_align.to_excel(w, sheet_name="03_回复分列对齐", index=False)
            rounds.to_excel(w, sheet_name="04_裁判轮次", index=False)
        print(f"✓ {out}")
    print("\n=== 题目表 ===")
    print(q_check.to_string(index=False))
    print("\n=== 回复主分列（GPT-5.4 对齐）===")
    print(r_align[["数据来源", "主分列", "主分裁判", f"≥{MIN_MODELS}模型有主分", "主分完整率_%"]].to_string(index=False))

    q_miss = int(q["difficulty_score"].isna().sum())
    r_ok = r_align[f"≥{MIN_MODELS}模型有主分"].min() >= (PUBLIC_ANALYSIS_N if True else 0)
    cohort_ok = coverage.loc[coverage["数据来源"] == "合计", "回复+评估完整"].iloc[0] == 1000
    if q_miss:
        print(f"\n⚠ 题目表仍有 {q_miss} 题缺 difficulty_score（ComplexBench 需 Stage1 GPT-5.4）")
    if cohort_ok and r_align["主分完整率_%"].min() >= 100:
        print("\n✅ 回复分数 cohort 100% 就绪，可跑 run_stats.sh")
    else:
        print("\n⚠ 回复覆盖未达 100%，请检查后再统计")


if __name__ == "__main__":
    main()
